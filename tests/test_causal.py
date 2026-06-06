"""Causal-eval test, written FIRST (§3.5a). It exercises the patching HARNESS, which is a unit
invariant that holds for ANY SAE weights — so it builds a fresh random TopKSAE in-memory and never
touches a trained .pt artifact. It only needs the live net, so it skips until LEELA_SAE_CHECKPOINT
and LEELA_SAE_RESID_MODULE are set. Then it goes red (NotImplementedError) -> green once you
implement src/eval/causal.py.

INTERP-SAFETY (§3.5 carve-out): the assertion below is the one you MUST sanity-check by hand. It
encodes a no-op invariant — if a patch that changes nothing still reports a delta, the patching
harness is broken and every "causal effect" you later measure is noise. Verify before trusting.
"""

import os

import chess
import pytest
from lczerolens import LczeroBoard

from src.eval import causal

CHECKPOINT = os.environ.get("LEELA_SAE_CHECKPOINT")
RESID_MODULE = os.environ.get("LEELA_SAE_RESID_MODULE")

requires_causal_stack = pytest.mark.skipif(
    not (CHECKPOINT and RESID_MODULE),
    reason="set LEELA_SAE_CHECKPOINT and LEELA_SAE_RESID_MODULE (net + residual module name)",
)

STARTING_FEN = chess.STARTING_FEN


@pytest.fixture(scope="session")
def model():
    from src import hook
    return hook.load_model(CHECKPOINT)


@pytest.fixture(scope="session")
def sae(model):
    # No TRAINED checkpoint: the no-op invariant (scale=1.0 == identity) holds for ANY SAE weights,
    # so a fresh random TopKSAE keeps this test free of build artifacts. Size it to the net's
    # residual width (read off a real activation) so encode/decode shapes match the patched module,
    # and match its device/dtype so the round-trip doesn't trip a type mismatch.
    from src import hook
    from src.sae import TopKSAE
    resid = hook.get_residual_stream(model, STARTING_FEN, RESID_MODULE)   # (64, d_model)
    sae = TopKSAE(d_model=resid.shape[-1], dict_size=4 * resid.shape[-1], k=32)
    return sae.to(device=resid.device, dtype=resid.dtype).eval()


@requires_causal_stack
def test_identity_patch_is_a_noop(model, sae):
    # NO-OP INVARIANT (hand-checkable): scale=1.0 multiplies the feature by 1 -> the activation is
    # rewritten to ITSELF. patch_feature runs both batch rows through the same SAE round-trip and
    # only rows 64: are touched (by *1), so clean (row 0) and dirty (row 1) are identical and the
    # delta must be ~0. A nonzero delta means the harness perturbs the stream when it shouldn't —
    # a silent bug that would fake causal effects everywhere else.
    board = LczeroBoard(STARTING_FEN)
    policy_delta, wdl_delta = causal.patch_feature(
        model, sae, board, feature_idx=0, layer=RESID_MODULE, scale=1.0,
    )
    # patch_feature already returns abs deltas; max over all logits must stay at floor.
    assert float(policy_delta.max()) < 1e-4
    assert float(wdl_delta.max()) < 1e-4

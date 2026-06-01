"""Causal-eval test, written FIRST (§3.5a). Needs BOTH the live net and a trained SAE, so it skips
until LEELA_SAE_CHECKPOINT, LEELA_SAE_RESID_MODULE and LEELA_SAE_SAE_CKPT are set. Then it goes red
(NotImplementedError) -> green once you implement src/eval/causal.py.

INTERP-SAFETY (§3.5 carve-out): the assertion below is the one you MUST sanity-check by hand. It
encodes a no-op invariant — if a patch that changes nothing still reports a delta, the patching
harness is broken and every "causal effect" you later measure is noise. Verify before trusting.
"""

import os

import chess
import pytest

from src.eval import causal

CHECKPOINT = os.environ.get("LEELA_SAE_CHECKPOINT")
RESID_MODULE = os.environ.get("LEELA_SAE_RESID_MODULE")
SAE_CKPT = os.environ.get("LEELA_SAE_SAE_CKPT")

requires_causal_stack = pytest.mark.skipif(
    not (CHECKPOINT and RESID_MODULE and SAE_CKPT),
    reason="set LEELA_SAE_CHECKPOINT, LEELA_SAE_RESID_MODULE, LEELA_SAE_SAE_CKPT (net + trained SAE)",
)

STARTING_FEN = chess.STARTING_FEN


@pytest.fixture(scope="session")
def model():
    from src import hook
    return hook.load_model(CHECKPOINT)


@requires_causal_stack
def test_identity_patch_is_a_noop(model):
    # NO-OP INVARIANT (hand-checkable): patch feature 0 with scale set to whatever value it ALREADY
    # has on this board. That rewrites the activation to itself -> the heads must not move. We assert
    # both deltas are ~0. A nonzero delta here means the harness perturbs the stream even when it
    # shouldn't — a silent bug that would fake causal effects everywhere else.
    board = chess.LczeroBoard(STARTING_FEN) if hasattr(chess, "LczeroBoard") else chess.Board(STARTING_FEN)
    # scale=None is the agreed convention for "set the feature to its current value" (identity patch).
    policy_delta, wdl_delta = causal.patch_feature(model, board, feature_idx=0, layer=RESID_MODULE, scale=None)
    assert float(abs(policy_delta).max()) < 1e-4
    assert float(abs(wdl_delta).max()) < 1e-4

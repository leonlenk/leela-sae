"""Hook tests, written FIRST (§3.5a). INTERP-SAFETY (§3.5 carve-out): read each assertion and
confirm the hand-checkable fact in its comment before you trust a green bar — a wrong test here
silently blesses a false activation site.

These need a real net, and the checkpoint name is UNVERIFIED (DESIGN.md §1: don't guess it).
So they read it from the LEELA_SAE_CHECKPOINT env var and SKIP if unset. Once step 1 confirms
the id, set it and the tests go red (NotImplementedError) -> green as you implement src/hook.py.
"""

import math
import os

import chess
import pytest

from src import hook

CHECKPOINT = os.environ.get("LEELA_SAE_CHECKPOINT")
# A residual-stream module name you discovered via hook.list_residual_modules() in step 1.
RESID_MODULE = os.environ.get("LEELA_SAE_RESID_MODULE")

requires_net = pytest.mark.skipif(
    not CHECKPOINT,
    reason="set LEELA_SAE_CHECKPOINT once step 1 confirms the real HF repo id (don't guess it)",
)


@pytest.fixture(scope="session")
def model():
    return hook.load_model(CHECKPOINT)


# A position with a single, unmissable best move: White mates in one with Qb7#.
# Hand-checkable fact: in FEN "k7/8/1QK5/8/8/8/8/8 w - - 0 1" the black king is on a8,
# the white queen on b6, the white king on c6. Qb6-b7 is mate: the queen covers a7/a8/b8
# and the white king on c6 defends b7, so the king can't capture. Any engine plays Qb6-b7.
MATE_IN_ONE_FEN = "k7/8/1QK5/8/8/8/8/8 w - - 0 1"
MATE_IN_ONE_MOVE = chess.Move.from_uci("b6b7")


@requires_net
def test_predicts_known_forced_move(model):
    # Hand-checkable: Qb7# is the only mate-in-one here, so Leela's policy argmax must be b6b7.
    assert hook.predict_move(model, MATE_IN_ONE_FEN) == MATE_IN_ONE_MOVE


@requires_net
def test_list_residual_modules(model):
    mods = hook.list_residual_modules(model)
    # Hand-checkable: BT3 has 15 encoder blocks (encoder0..encoder14), and the per-block
    # residual site is each block's final LayerNorm "encoderN/ln2". So we expect 15 names.
    assert len(mods) == 15
    # Every candidate must be a block-final LN, and NONE may be a smolgen sub-LN (those are
    # 256/6144-wide attention machinery, not the 768-wide residual stream).
    assert all(n.endswith("/ln2") for n in mods)
    assert all("smolgen" not in n for n in mods)
    # Must come back ordered by block index so picking "layer k" is unambiguous.
    idx = [int(n.split("encoder")[1].split("/")[0]) for n in mods]
    assert idx == sorted(idx) == list(range(15))


@requires_net
def test_residual_stream_is_per_square(model):
    if not RESID_MODULE:
        pytest.skip("set LEELA_SAE_RESID_MODULE to a module name found in step 1")
    resid = hook.get_residual_stream(model, MATE_IN_ONE_FEN, RESID_MODULE)
    # Hand-checkable: a transformer net has 64 square-tokens; shape is (64, d_model), 2-D.
    assert resid.ndim == 2 and resid.shape[0] == 64


@requires_net
def test_history_dependence_is_measured_and_reported(model):
    # DIAGNOSTIC, NOT a gate (DESIGN.md §5 changed note). We now KEEP a history-using net and
    # control for history downstream (eval/history_filter.py), so we must NOT assert the net
    # ignores history. We only confirm the probe actually MEASURES and REPORTS the regime.
    deltas = hook.output_depends_on_history(model, MATE_IN_ONE_FEN)
    # Hand-checkable: the probe reports one logit-shift magnitude per history perturbation it
    # applies while holding the current board fixed — zeroing the planes, and randomizing them.
    assert set(deltas) == {"zero_delta", "randomize_delta"}
    # Hand-checkable: each delta is a finite, non-negative magnitude (max|Δlogit| >= 0). We
    # intentionally do NOT require it to be ~0 — we EXPECT history dependence here, by design.
    assert all(math.isfinite(v) and v >= 0.0 for v in deltas.values())

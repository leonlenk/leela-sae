"""THE canary (DESIGN.md §6c): a feature you believe encodes "white king on g1" must fire on g1
positions and stay quiet otherwise. Written FIRST (§3.5a) but it's an INTEGRATION test — it needs
the net, a trained SAE, the discovered residual module, and the feature index you found via the
correlational eval. It skips until all are set, then goes red -> green.

INTERP-SAFETY (§3.5 carve-out): this is your one hand-labelled ground-truth position. Read the
assertion and confirm by eye that the g1 FENs really have the white king on g1 and the others
don't. If this canary is green but wrong, you will "discover" features that aren't real.
"""

import os

import chess
import pytest

CHECKPOINT = os.environ.get("LEELA_SAE_CHECKPOINT")
RESID_MODULE = os.environ.get("LEELA_SAE_RESID_MODULE")
SAE_CKPT = os.environ.get("LEELA_SAE_SAE_CKPT")
KG1_FEATURE = os.environ.get("LEELA_SAE_KG1_FEATURE")   # the feature index your eval flagged for Kg1

requires_feature_stack = pytest.mark.skipif(
    not (CHECKPOINT and RESID_MODULE and SAE_CKPT and KG1_FEATURE),
    reason="needs net + trained SAE + LEELA_SAE_RESID_MODULE + LEELA_SAE_KG1_FEATURE (found via eval)",
)

# Hand-checkable: in all three KING_ON_G1 FENs the white king sits on g1; in the OTHERS it does not.
KING_ON_G1 = [
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 6 6",   # both sides castled K-side
    "rnbq1rk1/ppp2ppp/3p1n2/4p3/1bP5/2N2NP1/PP1PPPBP/R1BQ1RK1 w - - 0 6",     # white castled K-side
    "5rk1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",                                    # bare-ish, white Kg1
]
KING_NOT_ON_G1 = [
    chess.STARTING_FEN,                                                        # white king on e1
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",     # white king still on e1
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",                                           # white king on e1
]


@pytest.fixture(scope="session")
def stack():
    """Load the net + trained SAE once. Returns (model, sae, module_name)."""
    import torch
    from src import hook
    from src.sae import TopKSAE
    model = hook.load_model(CHECKPOINT)
    sae = torch.load(SAE_CKPT, weights_only=False)   # your training run saves a TopKSAE
    return model, sae, RESID_MODULE


def _feature_activation(stack, fen):
    """Encode the per-square residual stream for `fen` and return the canary feature's peak activation."""
    from src import hook
    model, sae, module_name = stack
    resid = hook.get_residual_stream(model, fen, module_name)   # (64, d_model)
    _, feats, _ = sae(resid)                                    # (recon, feats (64,F), pre)
    return feats[:, int(KG1_FEATURE)].max().item()             # strongest firing across the 64 squares


@requires_feature_stack
def test_canary_fires_iff_white_king_on_g1(stack):
    # Sanity-check the labels themselves first (cheap, no model): the FEN lists must be correct.
    assert all(chess.Board(f).king(chess.WHITE) == chess.G1 for f in KING_ON_G1)
    assert all(chess.Board(f).king(chess.WHITE) != chess.G1 for f in KING_NOT_ON_G1)

    on = [_feature_activation(stack, f) for f in KING_ON_G1]
    off = [_feature_activation(stack, f) for f in KING_NOT_ON_G1]
    # The feature must separate the two sets: its weakest firing on a g1 position should still beat
    # its strongest firing on a non-g1 position. That gap IS "fires iff white king on g1".
    assert min(on) > max(off)

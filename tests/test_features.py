"""THE canary (DESIGN.md §6c): the trained SAE's own-king feature must fire on the side-to-move's king
square and nowhere else, for BOTH colours to move. Integration test: needs the net and a trained
SAE + its eval results. Skips unless LEELA_SAE_CHECKPOINT and LEELA_SAE_SAE_CKPT are set, e.g.

    LEELA_SAE_CHECKPOINT=lczerolens/BT3-768x15x24h-swa-2790000 LEELA_SAE_SAE_CKPT=outputs/sae/L7.pt uv run pytest tests/test_features.py

INTERP-SAFETY: if this goes red after a change to the hook, encoding or token mapping, the pipeline
drifted. Confirm by eye that each FEN's mover's king is on the square named in the comment.
"""

import json
import os
from pathlib import Path

import chess
import pytest

from conftest import CHECKPOINT

SAE_CKPT = os.environ.get("LEELA_SAE_SAE_CKPT")
requires_stack = pytest.mark.skipif(not (CHECKPOINT and SAE_CKPT), reason="needs the net + a trained SAE (LEELA_SAE_SAE_CKPT)")

# (FEN, square of the side-to-move's king) — hand-checkable.
CASES = [
    ("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 6 6", "g1"),   # White castled
    ("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b - - 6 6", "g8"),   # same, Black to move
    (chess.STARTING_FEN, "e1"),
    ("8/8/4k3/8/8/8/3K4/8 b - - 0 1", "e6"),                                           # bare kings, Black
    ("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1", "g1"),
]


@pytest.fixture(scope="module")
def stack():
    from src import hook
    from src.sae import TopKSAE
    model = hook.load_model(CHECKPOINT)
    sae = TopKSAE.load(SAE_CKPT).to(hook.model_device(model))
    results = json.loads(Path("outputs/results", Path(SAE_CKPT).stem, "results.json").read_text())
    feature = next(r["feature"] for r in results["properties"] if r["property"] == "own_king")
    return model, sae, results["module"], feature


@requires_stack
def test_own_king_feature_fires_exactly_on_the_movers_king(stack):
    from src import hook
    from src.viz.features import token_to_square
    model, sae, module, feature = stack
    for fen, king_square in CASES:
        assert chess.Board(fen).king(chess.Board(fen).turn) == chess.parse_square(king_square)   # label sanity
        acts = sae.encode(hook.get_residual_stream(model, fen, module))[0][:, feature]           # (64,)
        firing = {chess.square_name(token_to_square(fen, int(t))) for t in (acts > 0).nonzero().flatten()}
        assert firing == {king_square}, f"{fen}: fires on {firing}, expected {{{king_square}}}"

"""Auto-interp plumbing tests with a fake LLM (no network). They pin the scoring arithmetic and the
board orientation the LLM sees; the label quality itself is what the real run measures."""

import re

import chess
import torch

from src.autointerp import labeler
from src.eval.codes import SparseCodes


def test_render_board_puts_own_pieces_at_the_bottom_for_black():
    txt = labeler.render_board("4k3/8/8/8/8/8/8/R3K3 b - - 0 1")
    rows = txt.splitlines()[1:]
    # Black to move: rank 1 is drawn at the TOP, rank 8 (own king, uppercase) at the bottom.
    assert rows[0].startswith("1 ") and rows[7].startswith("8 ")
    assert "K" in rows[7] and "r" in rows[0] and rows[8].split() == list("hgfedcba")


def _toy(n_boards=40, feature=3, F=8):
    """Feature `feature` fires on token 4 (e1/e8 king square) of the first half of the boards."""
    fens = [chess.STARTING_FEN if i % 2 else "4k3/8/8/8/8/8/8/4K3 w - - 0 1" for i in range(n_boards)]
    idx = torch.zeros(n_boards, 64, 1, dtype=torch.long)
    val = torch.zeros(n_boards, 64, 1)
    idx[: n_boards // 2, 4, 0] = feature
    val[: n_boards // 2, 4, 0] = torch.linspace(5, 1, n_boards // 2)
    return SparseCodes(idx, val, F), fens


def test_scores_are_perfect_for_an_oracle_and_explain_never_sees_detect_boards():
    codes, fens = _toy()
    acts = codes.feature(3)

    def silent(system, prompt, schema):                          # first pass: just learn the boards
        if "predictions" not in schema["properties"]:
            return {"reasoning": "x", "label": "own king on its home square"}
        return {"predictions": []}

    out = labeler.interpret_feature(codes, fens, 3, silent, n_top=5, n_random=3, n_detect=8)
    truth = {b: bool(acts[b].max() > 0) for b in out["detect_boards"]}
    assert sum(truth.values()) == 4 and len(truth) == 8                # balanced held-out set
    assert not set(out["detect_boards"]) & set(out["explain_boards"])  # held out from EXPLAIN

    def perfect(system, prompt, schema):
        if "predictions" not in schema["properties"]:
            return {"reasoning": "x", "label": "king square"}
        return {"predictions": [{"board": n, "fires": truth[b], "square": "e1" if truth[b] else "none"}
                                for n, b in enumerate(out["detect_boards"])]}

    # interpret_feature is deterministic given the seed, so the same boards come back.
    scored = labeler.interpret_feature(codes, fens, 3, perfect, n_top=5, n_random=3, n_detect=8)
    assert scored["detect_boards"] == out["detect_boards"]
    assert scored["balanced_accuracy"] == 1.0 and scored["square_hit_rate"] == 1.0

    def always_yes(system, prompt, schema):
        if "predictions" not in schema["properties"]:
            return {"reasoning": "x", "label": "anything"}
        return {"predictions": [{"board": n, "fires": True, "square": "a8"} for n in range(8)]}

    lazy = labeler.interpret_feature(codes, fens, 3, always_yes, n_top=5, n_random=3, n_detect=8)
    assert lazy["balanced_accuracy"] == 0.5 and lazy["square_hit_rate"] == 0.0   # chance, wrong square

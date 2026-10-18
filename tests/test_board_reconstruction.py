"""Board-reconstruction / coverage tests (Karvonen et al. 2024 metrics). Synthetic codes, no net.
Every expected number is hand-countable."""

import chess
import torch

from src.eval import board_reconstruction as br
from src.eval.codes import SparseCodes


def test_piece_grid_classes_and_black_mirror():
    g = br.piece_grid([chess.STARTING_FEN, "4k3/8/8/8/8/8/8/R3K3 b - - 0 1"])
    # White to move: e1 (token 4) = own king (6); e8 (token 60) = enemy king (12); e4 empty.
    assert g[0, 4] == 6 and g[0, 60] == 12 and g[0, 28] == 0 and g[0, 8] == 1   # a2 own pawn
    # Black to move: own king e8 sits on token 4; the white rook a1 (enemy, class 10) on token 56.
    assert g[1, 4] == 6 and g[1, 56] == 10 and g[1, 60] == 12


def oracle_codes(labels: torch.Tensor) -> SparseCodes:
    """One feature per class (feature c fires with activation 1 iff class c is on the token)."""
    N = labels.shape[0]
    idx = labels.clone()[..., None]                               # (N, 64, 1)
    val = (labels > 0).float()[..., None]
    return SparseCodes(idx, val, br.N_CLASSES)


def test_perfect_features_reconstruct_every_board_exactly():
    fens = [chess.STARTING_FEN, "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5",
            "4k3/8/8/8/8/8/8/R3K3 b - - 0 1"]
    labels = br.piece_grid(fens)
    codes = oracle_codes(labels)
    out = br.evaluate(codes, labels, codes, labels, thresholds=[0.0])
    assert out["reconstruction"]["f1"] == 1.0 and out["reconstruction"]["exact_board_rate"] == 1.0
    assert abs(out["coverage"] - 1.0) < 1e-9


def random_labels(n=300, seed=0):
    """Synthetic boards: an independent random class on every token, so no square has a >95% prior.
    (Assignment is per square — on identical boards even a feature firing everywhere would be
    "precise" for whatever always sits on each square.)"""
    return torch.randint(0, br.N_CLASSES, (n, 64), generator=torch.Generator().manual_seed(seed))


def test_unselective_feature_is_never_assigned_so_boards_score_zero():
    labels = random_labels()
    codes = SparseCodes(torch.zeros(300, 64, 1, dtype=torch.long), torch.ones(300, 64, 1), 1)   # fires everywhere
    out = br.evaluate(codes, labels, codes, labels, thresholds=[0.0])
    assert out["per_threshold"]["0.0"]["n_assigned_pairs"] == 0
    assert out["reconstruction"]["f1"] == 0.0 and out["reconstruction"]["exact_board_rate"] == 0.0


def test_board_f1_counts_wrong_piece_as_fp_and_fn():
    # One board, true: own king on token 4 only. Prediction: own QUEEN on token 4 and own king on token 5.
    labels = torch.zeros(1, 64, dtype=torch.long)
    labels[0, 4] = 6
    pred = torch.zeros(1, 64, br.N_CLASSES, dtype=torch.bool)
    pred[0, 4, 5] = True                                           # queen where the king is: FP + FN
    pred[0, 5, 6] = True                                           # king on an empty square: FP
    s = br.score_boards(pred, labels)
    assert s["f1"] == 0.0 and s["exact_board_rate"] == 0.0         # tp=0, fp=2, fn=1
    pred[0, 4, 6] = True                                           # now also predict the king correctly
    s = br.score_boards(pred, labels)
    assert abs(s["f1"] - 2 / (2 + 2 + 0)) < 1e-9                   # tp=1, fp=2, fn=0 -> F1 = 0.5


def test_threshold_is_relative_to_feature_max():
    # Feature 1 marks own pawns (class 1) with activation 1.0 but also fires weakly (0.05) on empty
    # tokens. At t=0 its precision for "own pawn" is ~0.5 (pawn vs empty equally common) -> not
    # assigned; at t=0.1 the cut is 0.1 * max = 0.1, only pawn firings remain -> precision 1.0.
    labels = random_labels()
    val = torch.where(labels == 1, 1.0, torch.where(labels == 0, 0.05, 0.0))[..., None]
    codes = SparseCodes(torch.ones(300, 64, 1, dtype=torch.long), val, 2)
    fmax = br.feature_max(codes)
    assert not br.assign_features(codes, labels, fmax, 0.0)[:, 1, 1].any()
    assert br.assign_features(codes, labels, fmax, 0.1)[:, 1, 1].all()


def test_strategy_coverage_oracle_check_feature_scores_one():
    # Fool's-mate check (White in check) vs the starting position: a feature firing on the own-king
    # token exactly when in check is a perfect "check" classifier at the board level.
    fens = ["rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", chess.STARTING_FEN] * 5
    per_square, board = br.strategy_labels(fens)
    assert board["check"].tolist() == [True, False] * 5
    idx = torch.zeros(10, 64, 1, dtype=torch.long)
    val = torch.zeros(10, 64, 1)
    val[board["check"], 4, 0] = 1.0                               # e1 = token 4 for White to move
    codes = SparseCodes(idx, val, 1)
    cov = br.strategy_coverage(codes, per_square, board, br.feature_max(codes))
    assert cov["check"] == 1.0
    assert "pin" not in cov                                       # no positives -> not scored

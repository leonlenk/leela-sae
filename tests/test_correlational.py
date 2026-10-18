"""Correlational-eval tests. Synthetic, no net. Every expected number is hand-computable by counting
true/false positives on paper — confirm them before trusting a green bar.
"""

import torch

from src.eval import correlational as corr
from src.eval.codes import SparseCodes


def codes_from_dense(dense: torch.Tensor, k: int) -> SparseCodes:
    """Build SparseCodes from a dense (N, 64, F) activation grid (k nonzeros max per token)."""
    top = torch.topk(dense, k, dim=-1)
    return SparseCodes(top.indices, top.values, dense.shape[-1])


def test_f1_distinguishes_itself_from_precision_recall_accuracy_jaccard():
    # tp=2, fp=1, fn=1, tn=96: precision = recall = 2/3, F1 = 2/3, Jaccard = 2/4 = 0.5,
    # accuracy = 98/100. Only F1 = 0.667 passes this assertion.
    p, r, f1 = corr.f1_from_counts(torch.tensor([2]), torch.tensor([3]), 3)
    assert torch.allclose(f1, torch.tensor([2 / 3], dtype=torch.float64))
    # Asymmetric case so precision != recall: tp=1, fired=4, positives=2 -> P=0.25, R=0.5, F1=1/3.
    p, r, f1 = corr.f1_from_counts(torch.tensor([1]), torch.tensor([4]), 2)
    assert float(p) == 0.25 and float(r) == 0.5 and abs(float(f1) - 1 / 3) < 1e-12


def test_f1_is_zero_not_nan_when_nothing_fires_or_no_positives():
    _, _, f1 = corr.f1_from_counts(torch.tensor([0, 0]), torch.tensor([0, 5]), 0)
    assert f1.tolist() == [0.0, 0.0]


def test_token_f1_all_counts_from_sparse_codes():
    # 1 board, F=3 features, k=1. Feature 0 fires on tokens 0,1,2; feature 1 on token 3; label on 0,1,3.
    dense = torch.zeros(1, 64, 3)
    dense[0, [0, 1, 2], 0] = 1.0
    dense[0, 3, 1] = 2.0
    labels = torch.zeros(1, 64, dtype=torch.bool)
    labels[0, [0, 1, 3]] = True
    p, r, f1 = corr.token_f1_all(codes_from_dense(dense, k=1), labels)
    # feature 0: tp=2 fired=3 pos=3 -> F1 = 4/6; feature 1: tp=1 fired=1 -> F1 = 2/4; feature 2: 0.
    assert torch.allclose(f1, torch.tensor([4 / 6, 0.5, 0.0], dtype=torch.float64))


def test_auroc_hand_cases():
    y = torch.tensor([False, False, True, True])
    assert corr.auroc(torch.tensor([0.1, 0.2, 0.8, 0.9]), y) == 1.0       # perfect separation
    assert corr.auroc(torch.tensor([0.9, 0.8, 0.2, 0.1]), y) == 0.0       # perfectly inverted
    assert corr.auroc(torch.zeros(4), y) == 0.5                           # all tied -> chance
    # One inversion out of the 4 (pos, neg) pairs: scores neg=[0.1, 0.5], pos=[0.4, 0.9] -> 3/4.
    assert corr.auroc(torch.tensor([0.1, 0.5, 0.4, 0.9]), y) == 0.75


def test_board_max_takes_max_over_tokens():
    dense = torch.zeros(2, 64, 4)
    dense[0, 5, 2], dense[0, 9, 2], dense[1, 0, 3] = 1.0, 3.0, 2.0
    bm = codes_from_dense(dense, k=2).board_max()
    assert bm.shape == (2, 4) and bm[0, 2] == 3.0 and bm[1, 3] == 2.0 and bm[0, 3] == 0.0


def test_evaluate_property_reports_held_out_numbers_not_selection_numbers():
    # Feature 0 tracks the label on BOTH splits. Feature 1 matches it perfectly on S (by "chance")
    # but is unrelated on T. With a rare label, feature 1 wins selection only if it beats feature 0
    # on S — so make feature 0 slightly imperfect on S and check T exposes feature 1 as a fluke.
    torch.manual_seed(0)
    N = 50
    y_S = torch.zeros(N, 64, dtype=torch.bool); y_S[:10, 0] = True
    y_T = torch.zeros(N, 64, dtype=torch.bool); y_T[10:20, 0] = True
    d_S = torch.zeros(N, 64, 2); d_T = torch.zeros(N, 64, 2)
    d_S[:9, 0, 0] = 1.0                    # feature 0: 9 of 10 positives on S
    d_T[10:20, 0, 0] = 1.0                 # ... and all 10 on T
    d_S[:10, 0, 1] = 1.0                   # feature 1: all 10 on S
    d_T[30:40, 0, 1] = 1.0                 # ... but fires on the wrong boards on T
    out = corr.evaluate_property(codes_from_dense(d_S, 2), y_S, codes_from_dense(d_T, 2), y_T, n_perm=3)
    assert out["feature"] == 1 and out["selection_f1"] == 1.0   # selection is fooled ...
    assert out["test_f1"] == 0.0                                 # ... and the held-out number says so
    assert out["n_pos_selection"] == 10 and out["n_pos_test"] == 10


def test_board_level_property():
    N = 40
    y = torch.zeros(N, dtype=torch.bool); y[:20] = True
    d = torch.zeros(N, 64, 3)
    d[:20, 7, 2] = 1.0                     # feature 2 fires (somewhere on the board) iff label
    codes = codes_from_dense(d, 1)
    out = corr.evaluate_property(codes, y, codes, y, level="board", n_perm=3)
    assert out["feature"] == 2 and out["test_f1"] == 1.0 and out["test_auroc"] == 1.0


def test_probe_threshold_is_exact_even_for_rare_positives():
    # 3 positives in 10,000 with perfectly separating scores: the best-F1 cut must isolate exactly
    # those 3 (F1 = 1). A percentile grid would land below them and pull in dozens of negatives.
    from src.eval.probes import best_f1_threshold
    scores = torch.linspace(0, 1, 10_000)
    labels = torch.zeros(10_000, dtype=torch.bool)
    labels[-3:] = True
    t = best_f1_threshold(scores, labels)
    assert ((scores > t) == labels).all()


def test_fit_probe_recovers_a_linearly_separable_rare_property():
    # 1% positives defined by a hidden linear rule: a working probe must score near-perfectly on a
    # fresh test split (threshold chosen on the training split only).
    from src.eval.probes import fit_probe
    torch.manual_seed(0)
    w = torch.randn(16)
    X_S, X_T = torch.randn(20_000, 16), torch.randn(20_000, 16)
    cut = torch.quantile(X_S @ w, 0.99)
    out = fit_probe(X_S, X_S @ w > cut, X_T, X_T @ w > cut, steps=300, device="cpu")
    assert out["test_auroc"] > 0.99 and out["test_f1"] > 0.85

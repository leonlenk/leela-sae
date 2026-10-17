"""Correlational eval: which single SAE feature tracks a chess property, measured HONESTLY.

The 2026-10 audit found the original protocol manufactured results: it took the argmax-F1 feature
out of 6144 and reported that same F1 on the same few hundred boards, so properties with a handful
of positives (en passant: 1 board) got near-perfect scores by chance. The protocol here:

  1. SELECT the best feature for property p on a selection split S (argmax F1 over all features).
  2. REPORT that feature's precision / recall / F1 / AUROC on a DISJOINT test split T.
  3. Context for every number:
       * prevalence and the always-fires F1 on T (what a constant predictor gets);
       * a permuted-label null on S: the best F1 any feature reaches when labels are shuffled
         (how much the argmax alone buys you);
       * supervised linear probes (src/eval/probes.py) on the same layer and on the raw input planes,
         so "the SAE has a feature for it" can be compared with "it is linearly decodable at all"
         and "it is already in the input".
  4. Only properties with >= min_positives positives in S are scored.

F1 concept: precision = of the tokens where the feature fires, the fraction with the property;
recall = of the tokens with the property, the fraction where it fires; F1 is their harmonic mean,
so a feature must be BOTH specific and complete to score high.
"""

from __future__ import annotations

import torch

from src.eval.codes import SparseCodes

EPS = 1e-12


def f1_from_counts(tp: torch.Tensor, fired: torch.Tensor, positives) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(precision, recall, F1) from per-feature true positives, fire counts and the positive count."""
    tp, fired = tp.double(), fired.double()
    positives = torch.as_tensor(positives, dtype=torch.float64)
    precision = tp / fired.clamp_min(1)
    recall = tp / positives.clamp_min(1)
    f1 = 2 * tp / (fired + positives).clamp_min(EPS)   # == 2PR/(P+R), and 0 when nothing fires/exists
    return precision, recall, f1


def token_f1_all(codes: SparseCodes, labels: torch.Tensor, thresh: float = 0.0):
    """Per-token F1 of EVERY feature vs a (N, 64) bool label. -> (precision, recall, F1), each (F,).

    A feature "fires" on a token when its activation is > thresh. Computed from the sparse codes with
    bincount, so it costs O(N*64*k), not O(N*64*F).
    """
    idx, val = codes.tokens()                       # (M, k)
    y = labels.reshape(-1).bool()                   # (M,)
    on = val > thresh                               # (M, k)
    fired = torch.bincount(idx[on], minlength=codes.n_features)
    tp = torch.bincount(idx[on & y[:, None]], minlength=codes.n_features)
    return f1_from_counts(tp, fired, y.sum())


def board_f1_all(board_acts: torch.Tensor, labels: torch.Tensor, thresh: float = 0.0):
    """Per-board F1 of every feature: fires on a board if its max over tokens is > thresh."""
    on = (board_acts > thresh).double()             # (N, F)
    y = labels.double()                             # (N,)
    return f1_from_counts(y @ on, on.sum(0), y.sum())


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties get average ranks)."""
    scores, labels = scores.reshape(-1).double(), labels.reshape(-1).bool()
    n_pos = int(labels.sum())
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    _, inverse, counts = torch.unique_consecutive(scores[order], return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0).double()
    ranks = ((ends - counts + 1 + ends) / 2)[inverse]     # average 1-based rank of each sorted score
    rank_sum = ranks[labels[order]].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_feature(f1: torch.Tensor) -> tuple[int, float]:
    best = int(torch.argmax(f1))
    return best, float(f1[best])


def permuted_null(codes: SparseCodes, labels: torch.Tensor, n_perm: int = 20, seed: int = 0, level: str = "square",
                  board_acts: torch.Tensor | None = None) -> torch.Tensor:
    """Best-feature F1 under shuffled labels, n_perm times. Tells you what the argmax over F
    features reaches with NO real relationship (high for rare properties)."""
    gen = torch.Generator().manual_seed(seed)
    flat = labels.reshape(-1)
    out = []
    for _ in range(n_perm):
        shuffled = flat[torch.randperm(flat.numel(), generator=gen)].reshape(labels.shape)
        if level == "square":
            f1 = token_f1_all(codes, shuffled)[2]
        else:
            f1 = board_f1_all(board_acts, shuffled)[2]
        out.append(float(f1.max()))
    return torch.tensor(out)


def evaluate_property(
    codes_S: SparseCodes, y_S: torch.Tensor, codes_T: SparseCodes, y_T: torch.Tensor,
    level: str = "square", n_perm: int = 20, seed: int = 0,
) -> dict:
    """Select on S, report on T. `y_*` are (N, 64) bool for level="square", (N,) for level="board"."""
    y_S, y_T = y_S.bool(), y_T.bool()
    if level == "square":
        f1_S = token_f1_all(codes_S, y_S)[2]
        feat, sel_f1 = best_feature(f1_S)
        null = permuted_null(codes_S, y_S, n_perm, seed, "square")
        scores_T = codes_T.feature(feat)                         # (N, 64)
    else:
        acts_S = codes_S.board_max()
        f1_S = board_f1_all(acts_S, y_S)[2]
        feat, sel_f1 = best_feature(f1_S)
        null = permuted_null(codes_S, y_S, n_perm, seed, "board", board_acts=acts_S)
        scores_T = codes_T.board_max()[:, feat]                  # (N,)

    fires_T = scores_T > 0
    tp = int((fires_T & y_T).sum())
    p, r, f1 = (float(x) for x in f1_from_counts(torch.tensor([tp]), torch.tensor([int(fires_T.sum())]), y_T.sum()))
    prevalence = float(y_T.float().mean())
    return {
        "feature": feat,
        "selection_f1": sel_f1,
        "null_f1_p95": float(torch.quantile(null, 0.95)),
        "test_f1": f1,
        "test_precision": p,
        "test_recall": r,
        "test_auroc": auroc(scores_T, y_T),
        "test_density": float(fires_T.float().mean()),           # how often the feature fires on T
        "test_prevalence": prevalence,
        "always_true_f1": 2 * prevalence / (1 + prevalence),     # F1 of "fires everywhere"
        "n_pos_selection": int(y_S.sum()),
        "n_pos_test": int(y_T.sum()),
    }

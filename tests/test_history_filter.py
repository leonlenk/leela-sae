"""History-filter tests. Synthetic codes, no net: the agreement statistic must equal a dense Pearson
correlation computed the obvious way, and the positive-control labels must point at the last move."""

import chess
import numpy as np
import torch

from src.eval.codes import SparseCodes
from src.eval.history_filter import activation_agreement, last_move_to_labels, survivors


def sparse(dense: torch.Tensor, k: int) -> SparseCodes:
    top = torch.topk(dense, k, dim=-1)
    return SparseCodes(top.indices, top.values, dense.shape[-1])


def dense_of(codes: SparseCodes) -> torch.Tensor:
    return torch.zeros(codes.idx.shape[:2] + (codes.n_features,)).scatter(-1, codes.idx, codes.val)


def test_agreement_equals_dense_pearson():
    torch.manual_seed(0)
    base = torch.relu(torch.randn(6, 64, 12))
    rep = sparse(base, 4)
    real = sparse(base + 0.3 * torch.relu(torch.randn(6, 64, 12)), 4)
    agree = activation_agreement(rep, real)
    x, y = dense_of(rep).reshape(-1, 12), dense_of(real).reshape(-1, 12)
    for f in range(12):
        if x[:, f].std() > 0 and y[:, f].std() > 0:
            ref = torch.corrcoef(torch.stack([x[:, f], y[:, f]]))[0, 1]
            assert abs(float(agree["corr"][f]) - float(ref)) < 1e-4


def test_identical_codes_survive_and_scrambled_ones_do_not():
    torch.manual_seed(1)
    base = torch.relu(torch.randn(8, 64, 10))
    rep = sparse(base, 3)
    same = survivors(activation_agreement(rep, rep), min_fires=5)
    assert bool(same["survives"][same["measurable"]].all())                 # corr == 1 everywhere
    shuffled = sparse(base[torch.randperm(8)], 3)                           # same features, other boards
    diff = survivors(activation_agreement(rep, shuffled), min_fires=5)
    assert diff["survives"].sum() < same["survives"].sum()


def test_last_move_labels_point_at_destination_in_token_order():
    # 1.e4 e5: last move e7e5, White to move -> token == square e5 (36).
    lab = last_move_to_labels(["e2e4 e7e5"])
    assert np.flatnonzero(lab[0]).tolist() == [chess.E5]
    # 1.e4: last move e2e4, Black to move -> board mirrored, square e4 (28) is token 28 ^ 56 = 36.
    lab = last_move_to_labels(["e2e4"])
    assert np.flatnonzero(lab[0]).tolist() == [chess.square_mirror(chess.E4)]

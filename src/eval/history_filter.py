"""History-robustness filter: does a feature track the BOARD, or the move history? (DESIGN.md §4, check 3)

Leela's input carries 7 previous positions. Everything in this project trains and evaluates on the
canonical history-free encoding (REP: the current board copied into every history slot — lc0's own
convention for a bare FEN). A feature can still be an artifact of that regime, or depend on
history-like cues, so we re-encode the SAME positions with their REAL game history (Lichess move
stacks) and compare each feature's activation on every token:

    agreement_f = Pearson correlation over tokens of f's activation under REP vs REAL.

A feature SURVIVES if it fires often enough to measure and agreement_f >= min_corr. The current board
is identical in both encodings, so a feature that tracks the board should barely move.

Why not the original idea (fill history planes with U(0,1) noise, or zeros)? Both are wildly off the
net's training distribution: the audit measured SAE reconstruction collapsing (FVU 0.17 -> 10.4) and
even the trivially real piece-identity features failing. Real histories are on-distribution.

Positive control: the destination square of the LAST move is only knowable from history. Features
that track it under REAL should fail the filter — if they passed, the filter would have no power.
"""

from __future__ import annotations

import chess
import numpy as np
import torch

from src.eval.board_props import token_order
from src.eval.codes import SparseCodes


def activation_agreement(rep: SparseCodes, real: SparseCodes) -> dict[str, torch.Tensor]:
    """Per-feature Pearson correlation (over all tokens) of activations under REP vs REAL, plus the
    number of tokens each feature fires on in each condition. Computed from sparse codes:
    a feature absent from a token's top-k has activation 0 there."""
    F = rep.n_features
    xi, xv = rep.tokens()                                    # (M, k)
    yi, yv = real.tokens()
    M = xi.shape[0]
    sx = torch.zeros(F, dtype=torch.float64).index_add_(0, xi.flatten(), xv.flatten().double())
    sy = torch.zeros(F, dtype=torch.float64).index_add_(0, yi.flatten(), yv.flatten().double())
    sxx = torch.zeros(F, dtype=torch.float64).index_add_(0, xi.flatten(), (xv.double() ** 2).flatten())
    syy = torch.zeros(F, dtype=torch.float64).index_add_(0, yi.flatten(), (yv.double() ** 2).flatten())
    sxy = torch.zeros(F, dtype=torch.float64)
    for s in range(0, M, 65536):                             # x*y is nonzero only where both fire
        a, av, b, bv = xi[s:s + 65536], xv[s:s + 65536], yi[s:s + 65536], yv[s:s + 65536]
        match = a[:, :, None] == b[:, None, :]               # (m, k, k)
        prod = (av[:, :, None] * bv[:, None, :] * match).sum(-1).double()   # (m, k), aligned with a
        sxy.index_add_(0, a.flatten(), prod.flatten())
    cov = M * sxy - sx * sy
    var = (M * sxx - sx ** 2).clamp_min(0) * (M * syy - sy ** 2).clamp_min(0)
    corr = torch.where(var > 0, cov / var.sqrt().clamp_min(1e-30), torch.full_like(cov, float("nan")))
    fires_rep = torch.bincount(xi[xv > 0], minlength=F)
    fires_real = torch.bincount(yi[yv > 0], minlength=F)
    return {"corr": corr.float(), "fires_rep": fires_rep, "fires_real": fires_real}


def survivors(agreement: dict[str, torch.Tensor], min_fires: int = 20, min_corr: float = 0.9) -> dict[str, torch.Tensor]:
    """Boolean masks over features: `measurable` (fires >= min_fires times in either condition) and
    `survives` (measurable AND REP/REAL correlation >= min_corr)."""
    measurable = torch.maximum(agreement["fires_rep"], agreement["fires_real"]) >= min_fires
    survives = measurable & (torch.nan_to_num(agreement["corr"], nan=-1.0) >= min_corr)
    return {"measurable": measurable, "survives": survives}


def last_move_to_labels(histories: list[str]) -> np.ndarray:
    """(N, 64) bool in TOKEN order: the destination square of the move that produced each position
    (the positive-control property; only visible with real history)."""
    out = np.zeros((len(histories), 64), dtype=bool)
    for i, h in enumerate(histories):
        moves = h.split()
        board = chess.Board()
        for uci in moves:
            board.push_uci(uci)
        last_to = chess.Move.from_uci(moves[-1]).to_square
        perm = token_order(board)                            # token t covers square perm[t]
        out[i, int(np.flatnonzero(perm == last_to)[0])] = True
    return out

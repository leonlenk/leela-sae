"""Feature mining for the "believe one feature" check: where does feature j fire hardest?

The inverse of correlational.py. No hypothesis: pick a feature, rank every (board, token) by its
activation, and read the shared chess concept off the top hits (src/viz/features renders them).
Everything here works from precomputed SparseCodes (src/eval/codes.py), so mining a feature or the
whole dictionary costs no extra net passes.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from src.eval.codes import SparseCodes

Activation = Tuple[str, int, float]  # (fen, token_index, activation) — feeds src/viz/features.py


def top_activating(feature_acts: torch.Tensor, fens: Sequence[str], k: int) -> List[Activation]:
    """Top-k (fen, token, activation) for ONE feature, strongest first, from a (N, 64) grid.

    Flat index f <-> board f // 64, token f % 64 (same row layout as the activation cache). Only
    STRICTLY positive activations are returned: a rare feature with fewer than k firings yields a
    shorter list rather than being padded with zero-activation rows at arbitrary tokens.
    """
    flat = torch.flatten(feature_acts)                       # (N*64,)
    top = torch.topk(flat, min(k, flat.numel()))
    return [(fens[int(i) // 64], int(i) % 64, float(v))
            for v, i in zip(top.values.tolist(), top.indices.tolist()) if v > 0]


def feature_density(feature_acts: torch.Tensor) -> torch.Tensor:
    """(M, F) dense activations -> (F,) fraction of tokens on which each feature is nonzero.

    0 -> dead (never in any token's top-k); tiny but > 0 -> rare, often the most interpretable;
    large -> fires almost everywhere, usually not a clean concept.
    """
    return (feature_acts != 0).float().mean(dim=0)


def density_from_codes(codes: SparseCodes) -> torch.Tensor:
    """(F,) firing density for every feature from sparse codes (same definition as feature_density)."""
    idx, val = codes.tokens()
    return torch.bincount(idx[val > 0], minlength=codes.n_features).float() / idx.shape[0]


def mine_feature(codes: SparseCodes, fens: Sequence[str], feature: int, k: int = 20) -> List[Activation]:
    """Top-k firings of one feature across the boards the codes were computed on."""
    return top_activating(codes.feature(feature), fens, k)

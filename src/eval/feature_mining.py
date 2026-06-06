"""Feature mining for the "believe one feature" check (🔒 — you write the bodies; scaffold per §3.5).

This is the INVERSE of correlational.py. There you ask "given a board property, which feature tracks
it?". Here you have NO hypothesis: you pick one feature, find the positions where it fires hardest,
and read the shared chess concept off them by eye (DESIGN.md §4 step 5). Discovery, not testing —
so you must NOT pre-pick positions; you mine the FENs you already cached and let them tell you.

Two pieces:
  * top_activating  — PURE ranking (no net). The learning core; covered by tests/test_feature_mining.
  * mine_feature    — net glue that builds the (N, 64) activation grid, then calls top_activating.
"""

from __future__ import annotations

from src import hook

from typing import List, Sequence, Tuple

import torch

Activation = Tuple[str, int, float]  # (fen, square_index, activation) — feeds src/viz/features.py


def top_activating(feature_acts: torch.Tensor, fens: Sequence[str], k: int) -> List[Activation]:
    """Top-k (fen, square, activation) for ONE feature across all boards.

    `feature_acts`: (N, 64) — one feature's activation on every TOKEN of every board. Row i lines
    up with fens[i]; column t is Leela token t (NOT necessarily python-chess square t — for black to
    move the board is vertically flipped; src/viz/features.token_to_square does that conversion).
    Returns a list of length k, sorted by activation DESCENDING, each item (fens[board], token, act).

    The crux is the index mapping: torch.topk over the flattened (N*64,) vector gives you flat
    indices, and flat index f ↔ board f // 64, square f % 64 — exactly the row↔(board,square) split
    src/data/cache.py uses. Get that division right and the rendered square will match the firing.
    """
    flattened_feats = torch.flatten(feature_acts)          # (N*64,)
    topk = torch.topk(flattened_feats, k)
    results: List[Activation] = []
    for value, flat_idx in zip(topk.values.tolist(), topk.indices.tolist()):
        board = int(flat_idx) // 64                         # which FEN this hit came from
        square = int(flat_idx) % 64                         # which of its 64 tokens (cache.py split)
        results.append((fens[board], square, float(value)))
    return results


def feature_density(feature_acts: torch.Tensor) -> torch.Tensor:
    """Per-feature firing density: (M, F) tokens -> (F,) fraction of tokens each feature fired on.

    "Density" == of all M tokens, on what fraction was this feature NONZERO (i.e. survived the SAE's
    top-k)? It's the cheap liveness signal, and the whole point of this function:
        0.0            -> DEAD: never in any token's top-k. Skip it.
        tiny but > 0   -> rare: often the interesting, near-monosemantic features.
        large (~1.0)   -> fires almost everywhere: usually junk, not worth eyeballing.
    Magnitudes are irrelevant here — only fired-vs-not. Pure tensor logic, no net, so it's unit-tested
    in isolation exactly like top_activating (tests/test_feature_mining.py).
    """
    fired = feature_acts != 0          # (M, F) bool — True wherever the feature survived top-k
    return fired.float().mean(dim=0)   # (F,)  mean of 0/1 over tokens == firing fraction per feature


def feature_liveness(
    model,
    fens: Sequence[str],
    sae,
    module_name: str,
) -> torch.Tensor:
    """ONE pass over `fens` -> (F,) firing density for EVERY SAE feature at once (find live features).

    The cheap inverse of guessing indices: mine_feature runs a FULL net pass to inspect a SINGLE
    feature column, so scanning the dictionary that way is one pass per feature (infeasible for F).
    Here we run the net once per board, encode ALL F features, and reduce each board to a (F,) density.

    Every FEN is exactly 64 tokens, so each board carries EQUAL weight — averaging the per-board
    densities therefore equals the true overall token fraction, and we never have to hold the giant
    (N, 64, F) activation grid in memory (only (N, F), which is ~50 MB at F=6144, N=2000).

    Returns (F,); in the notebook: `density = feature_liveness(...)`, then sort / threshold it to pick
    non-dead features instead of hand-guessing indices. density == 0 is dead.
    """
    sae = sae.to(next(model.parameters()).device)
    per_board = []
    with torch.no_grad():
        for fen in fens:
            resid = hook.get_residual_stream(model, fen, module_name)   # (64, D) residual stream
            _, feats, _ = sae(resid)                                    # feats: (64, F) post-top-k
            per_board.append(feature_density(feats))                    # (F,) density for THIS board
    return torch.stack(per_board).mean(dim=0)                           # (N, F) -> (F,) overall density


def mine_feature(
    model,
    fens: Sequence[str],
    sae,
    module_name: str,
    feature: int,
    k: int = 20,
) -> List[Activation]:
    """Run the net over `fens`, pull ONE SAE feature's per-square activations, return its top-k hits.

    Glue only — the ranking lives in top_activating. For each FEN we capture the (64, d_model)
    residual stream, encode it through the SAE to per-square features (64, F), and slice column
    `feature` to a (64,) vector. Stack those into the (N, 64) grid top_activating expects.

    NOTE on token order (CHECKED, 2026-06): Leela's 64 tokens are rank-major a1..h8 for WHITE to
    move (token == python-chess square), but VERTICALLY FLIPPED for black to move (token == square
    mirror). So return the raw token index here; the chess-square conversion lives in the viz layer
    (src/viz/features.token_to_square), which knows the fen. Don't convert in this function.
    """

    sae = sae.to(next(model.parameters()).device)
    per_board = []
    with torch.no_grad():
        for fen in fens:
            resid = hook.get_residual_stream(model, fen, module_name)
            _, feats, _ = sae(resid)
            per_board.append(feats[:, feature])
    feature_acts = torch.stack(per_board)
    return top_activating(feature_acts, fens, k)

"""Tests for the "believe one feature" mining core (DESIGN.md §4 step 5).

The net-dependent glue (mine_feature) is an integration concern; the LEARNING core is the pure
ranking step — given one feature's activation on every (board, square), return the top-k and map
each flat hit back to (board #, square #). That mapping is the same row r ↔ (r // 64, r % 64) idea
as src/data/cache.py, and it's the thing that's easy to get subtly wrong, so we pin it here with
hand-made tensors (no model required — this test runs immediately, red until you implement).
"""

import torch

from src.eval.feature_mining import feature_density, top_activating


def test_top_activating_ranks_and_maps_back():
    # Fabricate feature_acts: (N=3 boards, 64 squares). Plant three known firings of decreasing
    # strength on KNOWN (board, square) cells; everything else is zero.
    acts = torch.zeros(3, 64)
    acts[0, 63] = 9.0   # board 0, square 63 (h8) — strongest
    acts[1, 0] = 7.0    # board 1, square  0 (a1) — middle
    acts[2, 10] = 5.0   # board 2, square 10 (c2) — weakest of the three

    fens = ["FEN_A", "FEN_B", "FEN_C"]  # stand-ins; only the index mapping is under test here
    out = top_activating(acts, fens, k=3)

    # Must come back sorted by activation, strongest first, each carrying the RIGHT board + square.
    assert out == [
        ("FEN_A", 63, 9.0),
        ("FEN_B", 0, 7.0),
        ("FEN_C", 10, 5.0),
    ]


def test_top_activating_respects_k():
    acts = torch.zeros(2, 64)
    acts[0, 5] = 3.0
    acts[1, 9] = 4.0
    out = top_activating(acts, ["x", "y"], k=1)
    # k=1 keeps only the single strongest (board 1, square 9).
    assert len(out) == 1
    assert out[0] == ("y", 9, 4.0)


def test_feature_density_counts_firing_fraction():
    # Fabricate a (M=4 tokens, F=3 features) activation matrix. "Density" is purely a per-COLUMN
    # question: of the 4 tokens (rows), on how many is this feature nonzero? Magnitudes don't matter,
    # only fired-vs-not — so plant a known firing pattern and read the fractions straight off.
    feats = torch.tensor(
        [
            [0.0, 5.0, 0.0],   # token 0: feature 1 fires
            [1.0, 0.0, 0.0],   # token 1: feature 0 fires
            [0.0, 2.0, 0.0],   # token 2: feature 1 fires
            [3.0, 9.0, 0.0],   # token 3: features 0 and 1 fire
        ]
    )
    # feature 0 -> 2/4 tokens, feature 1 -> 3/4 tokens, feature 2 -> never (DEAD).
    out = feature_density(feats)
    assert out.shape == (3,)
    assert torch.allclose(out, torch.tensor([0.5, 0.75, 0.0]))


def test_feature_density_flags_dead_features():
    # Only column 2 ever fires; the other four are dead (density exactly 0.0) -> this is the signal
    # the notebook uses to skip indices like the hand-picked [0, 1, 5, 20, ...] guesses.
    feats = torch.zeros(10, 5)
    feats[:, 2] = 1.0          # feature 2 fires on every token
    out = feature_density(feats)
    assert out[2] == 1.0       # fires everywhere -> density 1.0
    assert int((out == 0.0).sum()) == 4   # the remaining four are dead

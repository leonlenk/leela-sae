"""Correlational-eval tests, written FIRST (§3.5a). Synthetic, no net — runs red now, green once you
implement src/eval/correlational.py. INTERP-SAFETY: F1 is the number your "finding" rests on; the
assertions below are hand-computable by counting tp/fp/fn on paper. Confirm them.
"""

import torch

from src.eval import correlational


def test_best_feature_picks_the_cleanest_column():
    # Column 1 matches the labels exactly; columns 0 and 2 do not. Best feature index must be 1.
    labels = torch.tensor([True, False, True])
    fires_matrix = torch.tensor([
        [True,  True,  False],
        [True,  False, True ],
        [False, True,  True ],
    ])
    idx, f1 = correlational.best_feature_for_property(fires_matrix, labels)
    assert idx == 1 and f1 == 1.0

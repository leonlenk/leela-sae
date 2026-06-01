"""Correlational-eval tests, written FIRST (§3.5a). Synthetic, no net — runs red now, green once you
implement src/eval/correlational.py. INTERP-SAFETY: F1 is the number your "finding" rests on; the
assertions below are hand-computable by counting tp/fp/fn on paper. Confirm them.
"""

import torch

from src.eval import correlational


def test_perfectly_correlated_feature_has_f1_one():
    # Feature fires exactly when the property is true -> no false alarms, no misses -> F1 == 1.0.
    labels = torch.tensor([True, False, True, False, True])
    fires = labels.clone()
    assert correlational.feature_property_f1(fires, labels) == 1.0


def test_never_firing_feature_has_f1_zero():
    # Feature never fires but the property is sometimes true -> recall 0, precision 0/0 -> F1 == 0.0.
    labels = torch.tensor([True, False, True, False, True])
    fires = torch.zeros(5, dtype=torch.bool)
    assert correlational.feature_property_f1(fires, labels) == 0.0


def test_known_f1_value_by_hand():
    # Hand-computed: labels true at idx {0,1,2}; feature fires at idx {0,1,3}.
    # tp=2 (0,1), fp=1 (3), fn=1 (2). precision=2/3, recall=2/3, F1 = 2/3 ≈ 0.6667.
    labels = torch.tensor([True, True, True, False])
    fires = torch.tensor([True, True, False, True])
    f1 = correlational.feature_property_f1(fires, labels)
    assert abs(f1 - (2 / 3)) < 1e-6


def test_binarize_threshold():
    acts = torch.tensor([[-1.0, 0.0, 0.5, 2.0]])
    # Default threshold 0.0: strictly positive activations count as "fired".
    assert correlational.binarize(acts).tolist() == [[False, False, True, True]]


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

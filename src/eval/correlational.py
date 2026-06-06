"""Correlational eval: F1 / coverage of SAE features vs board properties (🔒 — scaffold per §3.5).

This answers "does feature j track property p?" across many positions. High F1 is NECESSARY but
NOT sufficient for a feature to be "real" — it must also pass the causal test (DESIGN.md §4).
"""

import torch

def best_feature_for_property(feature_fires_matrix, property_labels, eps = 1e-8):
    """Scan all features, return (best_feature_index, best_f1) for one property.

    `feature_fires_matrix`: (N, F) bool. This is how you find the candidate feature that most
    cleanly encodes a given chess property before spending causal-eval budget on it.
    """

    property_labels = property_labels.to(feature_fires_matrix.device)
    labels = property_labels.unsqueeze(-1)                               # (N,) -> (N, 1) broadcasts over F
    tp = (feature_fires_matrix & labels).sum(dim=0)                      # (F,) per-feature true positives
    fp = (feature_fires_matrix & ~labels).sum(dim=0)                     # (F,) false positives
    fn = (~feature_fires_matrix & labels).sum(dim=0)                     # (F,) false negatives (misses)
    precision = tp / (tp + fp + eps)                                     # (F,)
    recall = tp / (tp + fn + eps)                                        # (F,)
    f1 = 2 * precision * recall / (precision + recall + eps)             # (F,) eps in DENOM guards 0/0
    best = f1.argmax()                                                   # scalar index of best feature
    return int(best), float(f1[best])

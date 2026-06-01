"""Correlational eval: F1 / coverage of SAE features vs board properties (🔒 — scaffold per §3.5).

This answers "does feature j track property p?" across many positions. High F1 is NECESSARY but
NOT sufficient for a feature to be "real" — it must also pass the causal test (DESIGN.md §4).
"""

import torch


def binarize(feature_acts, threshold: float = 0.0):
    """Turn continuous feature activations into a fired/not-fired boolean. (N, F) -> (N, F) bool.

    A feature "fires" on a position when its activation exceeds `threshold`. We compare this
    boolean firing pattern against the boolean board-property labels.
    """
    # TODO (1): return (feature_acts > threshold).
    raise NotImplementedError


def feature_property_f1(feature_fires, property_labels):
    """F1 of one feature's firing pattern against one boolean property over N positions.

    `feature_fires`: (N,) bool — did feature j fire on each position.
    `property_labels`: (N,) bool — is property p true on each position.
    F1 = 2*precision*recall / (precision+recall). Concept refresher (so the formula isn't magic):
      precision = of positions where the feature fired, how many had the property (no false alarms);
      recall    = of positions with the property, how many made the feature fire (no misses).
    Returns a float in [0, 1]; define F1 = 0.0 when precision+recall == 0 (degenerate).
    """
    # TODO (1): tp = (feature_fires & property_labels).sum()                 # true positives
    # TODO (2): fp = (feature_fires & ~property_labels).sum()                # false positives
    # TODO (3): fn = (~feature_fires & property_labels).sum()               # false negatives (misses)
    # TODO (4): precision = tp / (tp + fp); recall = tp / (tp + fn)  (guard divide-by-zero -> 0)
    # TODO (5): return 2*precision*recall / (precision+recall), or 0.0 if the denominator is 0.
    raise NotImplementedError


def best_feature_for_property(feature_fires_matrix, property_labels):
    """Scan all features, return (best_feature_index, best_f1) for one property.

    `feature_fires_matrix`: (N, F) bool. This is how you find the candidate feature that most
    cleanly encodes a given chess property before spending causal-eval budget on it.
    """
    # TODO (1): compute feature_property_f1 for every column j of the matrix.
    # TODO (2): return the argmax index and its F1.
    raise NotImplementedError

"""Causal eval: patch/ablate an SAE feature and measure the policy/WDL delta (🔒 — scaffold per §3.5).

This is the ground truth that separates a real feature from a sparse code that merely reconstructs.
The heads (policy, wdl) come for free from model(board) — no extra wiring (DESIGN.md §1)."""


def patch_feature(model, board, feature_idx, layer, scale=0.0):
    """Ablate/scale one SAE feature at `layer` and measure the effect on the heads.

    Returns (policy_delta, wdl_delta). A *real* feature moves the policy/WDL heads
    when you remove it; a feature that only helps reconstruction won't.
    """
    # TODO (1): run model(board) once, cache output["policy"], output["wdl"].
    # TODO (2): with an nnsight trace, at module `layer`: encode the activation
    #           with the SAE, set feature `feature_idx` to `scale`, decode back,
    #           and substitute it into the stream.
    # TODO (3): re-read policy/wdl from the patched run.
    # TODO (4): return the deltas vs the cached clean run.
    raise NotImplementedError

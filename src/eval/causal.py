"""Causal eval: patch/ablate an SAE feature and measure the policy/WDL delta (🔒 — scaffold per §3.5).

This is the ground truth that separates a real feature from a sparse code that merely reconstructs.
The heads (policy, wdl) come for free from model(board) — no extra wiring (DESIGN.md §1)."""

import torch
from nnsight import NNsight
from lczerolens import LczeroModel, LczeroBoard
from lczerolens.board import InputEncoding

from src.sae import TopKSAE


def patch_feature(
    base_model: LczeroModel,
    sae: TopKSAE,
    board: LczeroBoard,
    feature_idx: int,
    layer: str,
    scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ablate/scale one SAE feature at `layer` and measure the effect on the heads.

    Returns (policy_delta, wdl_delta). A *real* feature moves the policy/WDL heads
    when you remove it; a feature that only helps reconstruction won't.

    concept — WHAT this measures: the delta is the change in *Leela's* policy/WDL
    output, NOT the SAE's reconstruction error. We corrupt ONE feature, push the
    patched activation through the rest of the net, and read the heads. The clean
    baseline is the FULL SAE round-trip (all features), so the SAE's own
    reconstruction error is shared by both runs and cancels in the subtraction.
    """
    device = next(base_model.parameters()).device
    sae = sae.to(device)
    one_board = board.to_input_tensor(input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED).to(device)
    model_input = torch.stack([one_board, one_board], dim=0)   # (2, 112, 8, 8)

    nn_model = NNsight(base_model)
    resid = nn_model.get(layer)
    with torch.no_grad():
        with nn_model.trace(model_input):
            act = resid.output                  # (2*64, d_model) — net collapses batch & square
            feats, _ = sae.encode(act)          # (2*64, F)
            # Dirty board == batch row 1 == the SECOND block of 64 squares. Ablate only there;
            # rows 0:64 (clean board) keep all features so it stays a faithful round-trip.
            feats[64:, feature_idx] *= scale    # (64, ) <- one feature, dirty board's squares
            resid.output = sae.decode(feats)    # (2*64, d_model)
            policy = nn_model.output["policy"].save()   # (2, 1858)
            wdl = nn_model.output["wdl"].save()         # (2, 3)

    # clean == row 0, dirty == row 1. abs delta = magnitude of the head shift from the ablation.
    policy_delta = (policy[1] - policy[0]).abs()   # (1858,)
    wdl_delta = (wdl[1] - wdl[0]).abs()            # (3,)
    return policy_delta, wdl_delta

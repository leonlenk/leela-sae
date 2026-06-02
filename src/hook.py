"""Load the Leela net, reproduce a move, and pull the per-square residual stream (🔒 — scaffold per §3.5).

Hooking stack (DESIGN.md §1): lczerolens loads the net (`LczeroModel.from_hf`), nnsight does the
capture/patch. The exact residual-stream module name is NOT known in advance — you read it off
`print(model)` in step 1. Nothing here may include or depend on the 8-position history stack.
"""

import torch
from nnsight import NNsight
from lczerolens import LczeroModel
from lczerolens import LczeroBoard
from lczerolens.board import InputEncoding  

def load_model(checkpoint: str = "lczerolens/BT3-768x15x24h-swa-2790000", filename: str ="model.onnx"):
    """Load a BT-series (transformer) Leela net from HuggingFace into a differentiable graph.

    `checkpoint` is an HF repo id like "lczerolens/<name>". Do NOT hardcode a guess —
    confirm the real id by loading it (DESIGN.md §1: the model card may be empty).
    """
    return LczeroModel.from_hf(repo_id=checkpoint, filename=filename)


def predict_move(model: LczeroModel, fen: str):
    """Return the move Leela's policy head prefers for `fen` (its argmax over LEGAL moves).

    This is the step-1 gate: this move must match Leela's actual move on known positions.
    """
    board = LczeroBoard(fen)
    # Encode with history = the current board repeated across all 8 snapshots. A bare FEN has no
    # move stack, so lczerolens's default `model(board)` ZEROS the history planes (13:104) — that
    # is off-distribution for this history-using net and yields garbage (flat policy, ~100% draw).
    # Repeating the current board is the standard convention for scoring a single static position.
    board_input = board.to_input_tensor(input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED)
    output = model(board_input)
    legal_indices = board.get_legal_indices()
    legal_policy = output["policy"][0].gather(0, legal_indices)
    best_move_index = legal_indices[legal_policy.argmax()]
    best_legal_move = board.decode_move(int(best_move_index))
    return best_legal_move


def list_residual_modules(model: LczeroModel) -> list[str]:
    """Return candidate module names for the per-square residual stream, read off the model.

    On a BT3 transformer these are attention/MLP block outputs (names differ from the conv
    SE-ResNet example in the docs). You eyeball `print(model)` and pick the post-block site.
    """
    names: list[str] = []
    for name, _module in model.named_modules():
        # Keep only the 15 numbered encoder blocks' final LayerNorm: "encoderN/ln2".
        # Exclude (verified by reading off BT3, 2026-06):
        #   - smolgen sub-LayerNorms ("encoderN/smolgen/ln1|ln2"): 256/6144-wide attention-weight
        #     machinery, NOT the 768-wide residual stream.
        #   - "attn_body/ln2": the input-embedding stack's LN that runs BEFORE encoder0. It IS a
        #     residual site ("layer 0"), but it has no "encoderN" index, so it breaks the sort and
        #     muddies "layer k" indexing. Requiring "encoder" in the name drops it cleanly.
        if name.endswith("/ln2") and "encoder" in name and "smolgen" not in name:
            names.append(name)  # e.g. "module.encoder7/ln2"
    names.sort(key=lambda n: int(n.split("encoder")[1].split("/")[0]))
    return names


def get_residual_stream(model: LczeroModel, fen: str, module_name: str) -> torch.Tensor:
    """Capture the residual-stream activation at `module_name` for one position.

    Expected shape (B=1, 64, d_model): per-square == 64 "tokens" (a transformer net, NOT the
    conv `[B, C, 8, 8]` layout). Squeeze the batch dim before returning -> (64, d_model).
    """
    board = LczeroBoard(fen)
    # Same fix as predict_move: feed REPEATED-history planes, not the raw board (whose default
    # encoding zeros history and pushes the net off-distribution). The residual stream we capture
    # MUST come from the same in-distribution input the policy/value heads see, or every feature
    # we later "find" is keyed off a broken activation.
    board_input = board.to_input_tensor(input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED)
    nn_model = NNsight(model)
    resid_module = nn_model.get(module_name)
    with nn_model.trace(board_input):
        resid = resid_module.output.save()  # (1, 64, d_model) batched / (64, d_model) for one board

    return resid.squeeze(0)


def output_depends_on_history(model: LczeroModel, fen: str) -> dict[str, float]:
    """DIAGNOSTIC (not a gate): how much does perturbing ONLY the history planes move the policy?

    DESIGN.md §1/§5 (changed semantics): we KEEP a history-using net and control for history
    downstream (hold history fixed when patching; filter features by history-randomization in
    eval/history_filter.py). So this no longer pass/fails the net — it just *records the regime*.
    We EXPECT nonzero deltas (empirically ~0.7 zeroing, ~3.3 randomizing on the look-ahead net).

    Returns the magnitude of the policy-logit shift under two history perturbations, holding the
    current board (planes 0:13) FIXED so any shift is attributable to history:
        "zero_delta"      — history planes set to 0       (erase history)
        "randomize_delta" — history planes set to U(0,1)  (scramble history)
    Magnitude = max absolute change across the 1858 policy logits.
    """
    board = LczeroBoard(fen)
    # 112-plane input with all 8 history snapshots == the current board (no real move history).
    # This is the clean baseline; the current board lives in planes 0:13, history in 13:104,
    # castling/side-to-move/rule50 aux in 104:112.                          # (112, 8, 8)
    t_repeated = board.to_input_tensor(input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED)
    clean = model(t_repeated)["policy"][0]                                   # (1858,) baseline logits

    # Perturb ONLY planes 13:104 (history); 0:13 (current board) and 104:112 (aux) stay FIXED.
    t_zeroed = t_repeated.clone()
    t_zeroed[13:104] = 0.0
    zero_delta = (model(t_zeroed)["policy"][0] - clean).abs().max().item()   # max |Δlogit|

    t_randomized = t_repeated.clone()
    t_randomized[13:104] = torch.rand_like(t_randomized[13:104])
    randomize_delta = (model(t_randomized)["policy"][0] - clean).abs().max().item()

    return {"zero_delta": zero_delta, "randomize_delta": randomize_delta}

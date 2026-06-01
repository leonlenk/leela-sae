"""Load the Leela net, reproduce a move, and pull the per-square residual stream (🔒 — scaffold per §3.5).

Hooking stack (DESIGN.md §1): lczerolens loads the net (`LczeroModel.from_hf`), nnsight does the
capture/patch. The exact residual-stream module name is NOT known in advance — you read it off
`print(model)` in step 1. Nothing here may include or depend on the 8-position history stack.
"""

import torch


def load_model(checkpoint: str):
    """Load a BT-series (transformer) Leela net from HuggingFace into a differentiable graph.

    `checkpoint` is an HF repo id like "lczerolens/<name>". Do NOT hardcode a guess —
    confirm the real id by loading it (DESIGN.md §1: the model card may be empty).
    """
    # TODO (1): from lczerolens import LczeroModel
    # TODO (2): return LczeroModel.from_hf(checkpoint)   # loads the ONNX, not the .pt (see §1 caveat)
    raise NotImplementedError


def predict_move(model, fen: str):
    """Return the move Leela's policy head prefers for `fen` (its argmax over LEGAL moves).

    This is the step-1 gate: this move must match Leela's actual move on known positions.
    """
    # TODO (1): from lczerolens import LczeroBoard; board = LczeroBoard(fen)
    # TODO (2): output = model(board)                    # TensorDict: output["policy"] is (1, 1858)
    # TODO (3): legal = board.get_legal_indices()        # mask: policy includes illegal moves too
    # TODO (4): pick the legal index with the highest policy logit (mask out illegals first).
    # TODO (5): return board.decode_move(<that index>)   # index -> chess.Move
    raise NotImplementedError


def list_residual_modules(model) -> list[str]:
    """Return candidate module names for the per-square residual stream, read off the model.

    On a BT3 transformer these are attention/MLP block outputs (names differ from the conv
    SE-ResNet example in the docs). You eyeball `print(model)` and pick the post-block site.
    """
    # TODO (1): walk model.named_modules() and collect names that look like per-block outputs.
    # TODO (2): return them so step 1 can print and you can choose the layer index by hand.
    raise NotImplementedError


def get_residual_stream(model, fen: str, module_name: str) -> torch.Tensor:
    """Capture the residual-stream activation at `module_name` for one position.

    Expected shape (B=1, 64, d_model): per-square == 64 "tokens" (a transformer net, NOT the
    conv `[B, C, 8, 8]` layout). Squeeze the batch dim before returning -> (64, d_model).
    """
    # TODO (1): from nnsight import ... ; build LczeroBoard(fen).
    # TODO (2): open an nnsight trace on `model` for this input.
    # TODO (3): inside the trace, grab the output of the submodule named `module_name`
    #           and call .save() so it survives outside the trace.            # (1, 64, d_model)
    # TODO (4): run the trace, then return the saved tensor squeezed to (64, d_model).
    raise NotImplementedError


def output_depends_on_history(model, fen: str) -> bool:
    """Verification probe: does changing ONLY the history planes change the output?

    Returns True if perturbing the 8-position history (planes that are NOT the current board)
    moves policy/wdl. For the Jenner current-board finetune this must be False — that's how you
    confirm the net you loaded actually ignores history (DESIGN.md §1, step-1 check #2).
    """
    # TODO (1): run model on the position to get baseline output["policy"]/["wdl"].
    # TODO (2): build a second input that differs ONLY in the history planes (input is (1,112,8,8);
    #           the current board is a known subset of those 112 planes — perturb the rest).
    # TODO (3): run again; return whether policy/wdl changed beyond a small tolerance.
    raise NotImplementedError

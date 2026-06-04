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

def load_model(checkpoint: str = "lczerolens/BT3-768x15x24h-swa-2790000", filename: str ="model.onnx", device: str | None = None):
    """Load a BT-series (transformer) Leela net from HuggingFace into a differentiable graph.

    `checkpoint` is an HF repo id like "lczerolens/<name>". Do NOT hardcode a guess —
    confirm the real id by loading it (DESIGN.md §1: the model card may be empty).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LczeroModel.from_hf(repo_id=checkpoint, filename=filename)
    return model.to(device)


def predict_move(model: LczeroModel, fen: str):
    """Return the move Leela's policy head prefers for `fen` (its argmax over LEGAL moves).

    This is the step-1 gate: this move must match Leela's actual move on known positions.
    """
    board = LczeroBoard(fen)
    device = next(model.parameters()).device
    # Encode with history = the current board repeated across all 8 snapshots. A bare FEN has no
    # move stack, so lczerolens's default `model(board)` ZEROS the history planes (13:104) — that
    # is off-distribution for this history-using net and yields garbage (flat policy, ~100% draw).
    # Repeating the current board is the standard convention for scoring a single static position.
    board_input = board.to_input_tensor(input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED).to(device)
    output = model(board_input)
    legal_indices = board.get_legal_indices().to(device)
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
        # ln2 is the last module before the skip, smolgen also has ln2
        # but is an addition to attention and does not come before residual
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
    device = next(model.parameters()).device
    board_input = board.to_input_tensor(input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED).to(device)
    nn_model = NNsight(model)
    resid_module = nn_model.get(module_name)
    with nn_model.trace(board_input):
        resid = resid_module.output.save()  # (1, 64, d_model) batched / (64, d_model) for one board

    return resid.squeeze(0)


def get_residual_stream_batch(
    model: LczeroModel, fens: list[str], module_name: str
) -> torch.Tensor:
    """Capture the residual stream for MANY positions in ONE forward pass -> (B, 64, d_model).

    This is the speed fix for caching (src/data/cache.py). The single-board get_residual_stream
    above runs one net pass per FEN and rebuilds NNsight every call; cache_activations calls a
    capture_fn once per chunk, so the way to go fast is to push the whole chunk through together.
    Wire this in as the capture_fn (no batched_capture adapter needed): the cacher already speaks
    the (list[str]) -> (B, 64, d_model) protocol.

    Two invariants this MUST preserve, or every downstream feature is keyed off broken numbers:
      * SAME input encoding as get_residual_stream — INPUT_CLASSICAL_112_PLANE_REPEATED, NOT the
        raw board (whose default zeros the history planes and goes off-distribution).
      * SAME activation site — module_name's output, unchanged. Never swap the layer here.
    Unlike get_residual_stream this KEEPS the batch dim: row b corresponds to fens[b].
    """

    per_board = [
        LczeroBoard(fen).to_input_tensor(
            input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED
        )
        for fen in fens
    ]                                              # B x (112, 8, 8)

    device = next(model.parameters()).device
    batched_input = torch.stack(per_board, dim=0).to(device)  # (B, 112, 8, 8)

    nn_model = NNsight(model)
    resid_module = nn_model.get(module_name)

    with nn_model.trace(batched_input):
        resid = resid_module.output.save()         # (B*64, d_model) — net collapses batch+square!

    return resid.reshape(len(fens), 64, resid.shape[-1])  # (B, 64, d_model)


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
    device = next(model.parameters()).device
    # 112-plane input with all 8 history snapshots == the current board (no real move history).
    # This is the clean baseline; the current board lives in planes 0:13, history in 13:104,
    t_repeated = board.to_input_tensor(input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED).to(device)
    clean = model(t_repeated)["policy"][0]                                   # (1858,) baseline logits

    # Perturb ONLY planes 13:104 (history); 0:13 (current board) and 104:112 (aux) stay FIXED.
    t_zeroed = t_repeated.clone()
    t_zeroed[13:104] = 0.0
    zero_delta = (model(t_zeroed)["policy"][0] - clean).abs().max().item()   # max |Δlogit|

    t_randomized = t_repeated.clone()
    t_randomized[13:104] = torch.rand_like(t_randomized[13:104])
    randomize_delta = (model(t_randomized)["policy"][0] - clean).abs().max().item()

    return {"zero_delta": zero_delta, "randomize_delta": randomize_delta}

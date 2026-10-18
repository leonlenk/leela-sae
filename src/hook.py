"""Load the Leela net, encode positions, read its heads, and capture / patch the per-square residual stream.

Facts about the net that this module relies on — each was verified empirically in the 2026-10 audit
(probe scripts are summarised in RESULTS.md):

  * ARCHITECTURE. BT3 encoder blocks are post-LayerNorm with a DeepNorm-style α≈0.427:
        x1 = LN1(x + α·MHA(x)),   x2 = LN2(x1 + α·FFN(x1)).
    So `encoderN/ln2`'s output IS the residual stream that enters block N+1 (its only consumers are
    block N+1's Q/K/V, smolgen and skip connection). Patching it replaces block N+1's input exactly.
  * TOKENS. The net flattens (B, 112, 8, 8) -> (B*64, C) with token t = rank*8 + file, in the
    SIDE-TO-MOVE frame: for White token t == python-chess square t; for Black the board is flipped
    vertically (ranks only), so token t == chess.square_mirror(t). See src/viz/features.token_to_square.
    Note every token mixes in the whole board from layer 0 on (a dense board-level embedding term).
  * POLICY HEAD. output["policy"] is RAW logits over lc0's 1858 move slots, INCLUDING illegal moves.
    lc0 masks illegal moves out of its training loss, so those logits are untrained junk: on real
    positions ~94% of a full 1858-way softmax lands on illegal slots. ALWAYS restrict to legal moves
    (legal_log_probs) before a softmax, an argmax or a KL.
  * CASTLING. lc0 encodes castling as king-takes-own-rook (e1h1 / e1a1). python-chess, and
    lczerolens's get_legal_indices, use e1g1 / e1c1 — whose slots the net never puts mass on. Use
    policy_index(), which remaps castling, never board.get_legal_indices().
  * PROMOTIONS. lczerolens's decode_move drops the promotion piece and can return an illegal Move, so
    we never decode indices; we score each legal Move and return the Move object itself.
  * VALUE HEAD. output["wdl"] is already softmaxed: probabilities [W, D, L] from the side to move.
  * INPUT. Classical 112 planes: 8 history slots × 13 planes (own P N B R Q K, enemy P N B R Q K,
    repetition), then castling (us O-O-O, us O-O, them O-O-O, them O-O), black-to-move, the raw
    rule-50 halfmove count, a zero plane and an all-ones plane. There is NO en-passant plane — lc0
    infers ep from history — so under the history-free encoding below the net cannot see ep at all.

HISTORY (DESIGN.md §1, CLAUDE.md): the canonical encoding everywhere is `encode_fens`, which builds a
board from a bare FEN (no move stack) so all 8 history slots hold the CURRENT board. That is lc0's
own convention for a position given as a FEN, and it makes every activation a function of the
current board only. Real move history enters ONLY through the explicit `encode_boards_with_history`,
used by src/eval/history_filter.py as a controlled perturbation.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterator, Sequence

import chess
import torch
from lczerolens import LczeroBoard, LczeroModel
from lczerolens.board import InputEncoding

DEFAULT_CHECKPOINT = "lczerolens/BT3-768x15x24h-swa-2790000"
ENCODING = InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED
HISTORY_PLANES = slice(13, 104)   # history slots 1..7 (slot 0 = planes 0:13 is the current board)


# --------------------------------------------------------------------------------------------------
# Loading + encoding
# --------------------------------------------------------------------------------------------------

def load_model(checkpoint: str = DEFAULT_CHECKPOINT, filename: str = "model.onnx", device: str | None = None):
    """Load a BT-series (transformer) Leela net from the HF Hub (ONNX -> torch), frozen, in eval mode."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LczeroModel.from_hf(repo_id=checkpoint, filename=filename).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)   # we only ever read / patch activations, never train the net
    return model


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def encode_fens(fens: Sequence[str]) -> torch.Tensor:
    """Canonical, history-free encoding: (B, 112, 8, 8), current board copied into all 8 history slots."""
    return torch.stack([LczeroBoard(fen).to_input_tensor(input_encoding=ENCODING) for fen in fens])


def encode_boards_with_history(boards: Sequence[LczeroBoard]) -> torch.Tensor:
    """EXPLICIT real-history encoding: (B, 112, 8, 8) built from each board's move stack (up to 7
    previous positions; if the stack is shorter, the earliest position is repeated). Only the
    history filter should call this."""
    return torch.stack([b.to_input_tensor(input_encoding=ENCODING) for b in boards])


# --------------------------------------------------------------------------------------------------
# Policy head: legal moves only
# --------------------------------------------------------------------------------------------------

def policy_index(board: chess.Board, move: chess.Move) -> int:
    """lc0 policy slot of a LEGAL move, with castling remapped to lc0's king-takes-rook convention."""
    if board.is_castling(move):
        rook_file = 7 if board.is_kingside_castling(move) else 0
        move = chess.Move(move.from_square, chess.square(rook_file, chess.square_rank(move.from_square)))
    return LczeroBoard.encode_move(move, board.turn)


def legal_moves_and_indices(board: chess.Board) -> tuple[list[chess.Move], torch.Tensor]:
    """All legal moves and their lc0 policy slots, in matching order. -> (list[Move], (n_legal,) long)."""
    moves = list(board.legal_moves)
    return moves, torch.tensor([policy_index(board, m) for m in moves], dtype=torch.long)


def legal_log_probs(policy_logits: torch.Tensor, legal_idx: torch.Tensor) -> torch.Tensor:
    """log-softmax restricted to the legal slots. (1858,) logits, (n_legal,) idx -> (n_legal,) log-probs."""
    return torch.log_softmax(policy_logits[legal_idx.to(policy_logits.device)], dim=-1)


def legal_kl(clean_logits: torch.Tensor, patched_logits: torch.Tensor, legal_idx: torch.Tensor) -> torch.Tensor:
    """KL(clean || patched) over legal moves only, for ONE position. Scalar, in nats."""
    lp = legal_log_probs(clean_logits, legal_idx)
    lq = legal_log_probs(patched_logits, legal_idx)
    return (lp.exp() * (lp - lq)).sum()


def predict_move(model: LczeroModel, fen: str) -> chess.Move:
    """Leela's policy-head move for `fen` (argmax over LEGAL moves; no search). Returns a legal Move."""
    board = LczeroBoard(fen)
    moves, idx = legal_moves_and_indices(board)
    policy = model(encode_fens([fen]).to(model_device(model)))["policy"][0]   # (1858,) raw logits
    return moves[int(policy[idx.to(policy.device)].argmax())]


# --------------------------------------------------------------------------------------------------
# Residual stream: capture + patch (plain torch forward hooks; cross-checked against nnsight in tests)
# --------------------------------------------------------------------------------------------------

def list_residual_modules(model: LczeroModel) -> list[str]:
    """Per-block residual-stream sites `module.encoderN/ln2`, ordered by N (see module docstring).
    smolgen has its own ln2 inside the attention sublayer; it is not a residual site, so skip it."""
    names = [n for n, _ in model.named_modules() if n.endswith("/ln2") and "encoder" in n and "smolgen" not in n]
    return sorted(names, key=lambda n: int(n.split("encoder")[1].split("/")[0]))


def _module(model, name: str) -> torch.nn.Module:
    return dict(model.named_modules())[name]


@contextlib.contextmanager
def _hooks(model, fns: dict[str, Callable[[torch.Tensor], torch.Tensor | None]]) -> Iterator[None]:
    handles = [_module(model, n).register_forward_hook(lambda m, i, o, f=f: f(o)) for n, f in fns.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run_with_cache(model, inputs: torch.Tensor, modules: Sequence[str]) -> tuple[dict[str, torch.Tensor], dict]:
    """One forward pass; returns ({module: (B, 64, D) activation}, {"policy": (B,1858), "wdl": (B,3)}).

    The net reshapes its batch to (B*64, D) internally, so rows [b*64:(b+1)*64] belong to board b.
    """
    B = inputs.shape[0]
    store: dict[str, torch.Tensor] = {}

    def saver(name):
        def f(o):
            store[name] = o.detach().reshape(B, 64, -1)
            return None   # None = leave the output untouched
        return f

    with _hooks(model, {n: saver(n) for n in modules}):
        out = model(inputs.to(model_device(model)))
    return store, {"policy": out["policy"], "wdl": out["wdl"]}


@torch.no_grad()
def run_patched(model, inputs: torch.Tensor, module: str, fn: Callable[[torch.Tensor], torch.Tensor]) -> dict:
    """Forward pass with `module`'s output replaced by fn(act), act being (B, 64, D). Returns heads."""
    B = inputs.shape[0]

    def patch(o):
        new = fn(o.reshape(B, 64, -1))
        return new.reshape(o.shape).to(o.dtype)

    with _hooks(model, {module: patch}):
        out = model(inputs.to(model_device(model)))
    return {"policy": out["policy"], "wdl": out["wdl"]}


def get_residual_stream_batch(model: LczeroModel, fens: Sequence[str], module_name: str) -> torch.Tensor:
    """(B, 64, d_model) residual stream at `module_name` for many FENs in one pass (canonical encoding)."""
    acts, _ = run_with_cache(model, encode_fens(fens), [module_name])
    return acts[module_name]


def get_residual_stream(model: LczeroModel, fen: str, module_name: str) -> torch.Tensor:
    """(64, d_model) residual stream at `module_name` for one FEN (canonical encoding)."""
    return get_residual_stream_batch(model, [fen], module_name)[0]


# --------------------------------------------------------------------------------------------------
# History diagnostic
# --------------------------------------------------------------------------------------------------

@torch.no_grad()
def output_depends_on_history(model: LczeroModel, fen: str, seed: int = 0) -> dict[str, float]:
    """DIAGNOSTIC (not a gate): how much does perturbing ONLY the history planes move the policy?

    Baseline = canonical encoding (history slots == current board). We then overwrite planes 13:104
    with zeros, or with U(0,1) noise, holding the current board (0:13) and aux planes (104:112) fixed.
    Returns legal-move KL(baseline || perturbed) for each, plus whether the top legal move changed.
    Both perturbations are far off-distribution — this records the regime, it is NOT a feature filter
    (src/eval/history_filter.py compares against REAL game histories instead).
    """
    board = LczeroBoard(fen)
    _, idx = legal_moves_and_indices(board)
    x = encode_fens([fen])[0]
    assert x.ndim == 3 and x.shape[0] == 112, "plane slicing below assumes an unbatched (112, 8, 8) tensor"

    zeroed = x.clone()
    zeroed[HISTORY_PLANES] = 0.0
    noised = x.clone()
    gen = torch.Generator().manual_seed(seed)
    noised[HISTORY_PLANES] = torch.rand(noised[HISTORY_PLANES].shape, generator=gen)

    pol = model(torch.stack([x, zeroed, noised]).to(model_device(model)))["policy"]   # (3, 1858)
    top = [int(pol[i][idx.to(pol.device)].argmax()) for i in range(3)]
    return {
        "zero_kl": float(legal_kl(pol[0], pol[1], idx)),
        "randomize_kl": float(legal_kl(pol[0], pol[2], idx)),
        "zero_top1_changed": float(top[1] != top[0]),
        "randomize_top1_changed": float(top[2] != top[0]),
    }


@torch.no_grad()
def randomize_weights(model, seed: int = 0, mode: str = "permute", min_numel: int = 64, std: float = 0.02):
    """Untrained-net controls (cf. Karvonen et al. 2024's "SAE on a random-weights model"), IN PLACE.

    mode="permute": randomly permute the entries of every floating-point weight tensor with
        >= min_numel entries. Keeps each tensor's value distribution (trained-scale weights) but
        destroys everything learned. In this post-LN net that drives strong token mixing: by layer 7
        all 64 tokens collapse to (nearly) one vector, so no token says where a piece is, though
        that shared vector still carries a coarse summary of the board (scripts/14).
    mode="fresh": a conventional fresh init — weight matrices ~ N(0, std), vectors (biases) 0,
        LayerNorm gains 1. Small weights make each block close to the identity, so the input board
        passes through nearly untouched (a linear probe still reads every square at layer 7).
    The ONNX conversion stores most weights as buffers and LayerNorm affine params as parameters;
    tensors with < min_numel entries (scalars, scale constants) are left alone in both modes.
    """
    gen = torch.Generator().manual_seed(seed)
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if not t.is_floating_point() or t.numel() < min_numel:
            continue
        if mode == "permute":
            flat = t.reshape(-1)
            t.copy_(flat[torch.randperm(flat.numel(), generator=gen).to(flat.device)].reshape(t.shape))
        elif mode == "fresh":
            if t.ndim == 1:
                t.fill_(1.0 if (name.endswith(".weight") and "ln" in name) else 0.0)
            else:
                t.copy_(torch.randn(t.shape, generator=gen).to(t.device) * std)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return model

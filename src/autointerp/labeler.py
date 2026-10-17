"""Auto-interp: an LLM labels SAE features from their top activations, then is SCORED on held-out boards.

  1. EXPLAIN — show the LLM the boards where feature f fires hardest (plus a few random weaker
     firings, so the label covers the feature's range, not just its peak), with the firing squares
     and their activations, and ask for a short label.
  2. DETECT — in a fresh call, give ONLY the label and a shuffled set of held-out boards (half where f
     fires, half where it doesn't). The LLM predicts, per board, whether f fires and on which
     square. A label is only as good as its predictions: we report board-level balanced accuracy
     and, on boards where f fires, whether the predicted square is one f actually fires on.

Boards are shown in Leela's frame: "own" = side to move. LLM calls go through the Claude Code CLI in
headless mode (`claude -p`), so no API key is needed; any callable prompt -> JSON dict can be passed
as `llm` instead (e.g. an Anthropic SDK wrapper).
"""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
from typing import Callable, Sequence

import chess
import torch

from src.eval.codes import SparseCodes
from src.viz.features import describe_firing_piece, token_to_square

LLM = Callable[[str, str, dict], dict]   # (system, prompt, json_schema) -> parsed JSON

SYSTEM = (
    "You are an expert chess player helping interpret a neural network (Leela Chess Zero). "
    "The network represents each board as 64 square-tokens, from the point of view of the side to "
    "move: 'own' pieces belong to the side to move, 'enemy' pieces to the opponent. A 'feature' is "
    "one learned direction in its internal representation; it activates on particular squares of "
    "particular positions. Be concrete and chess-specific."
)

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "label": {"type": "string", "description": "<= 15 words: what the feature detects on the square where it fires"},
    },
    "required": ["reasoning", "label"],
}

DETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "board": {"type": "integer"},
                    "fires": {"type": "boolean"},
                    "square": {"type": "string", "description": "most likely firing square like 'e4', or 'none'"},
                },
                "required": ["board", "fires", "square"],
            },
        }
    },
    "required": ["predictions"],
}


def claude_cli(model: str = "sonnet", timeout: int = 300) -> LLM:
    """LLM backend via the Claude Code CLI (`claude -p`, your logged-in account): our own system
    prompt, no tools, no MCP servers, structured JSON output. Runs from an empty temp dir so no
    project CLAUDE.md leaks into the labeler's context."""
    workdir = tempfile.mkdtemp(prefix="autointerp_")

    def call(system: str, prompt: str, schema: dict) -> dict:
        cmd = ["claude", "-p", "--model", model, "--tools", "", "--strict-mcp-config", "--no-session-persistence",
               "--output-format", "json", "--system-prompt", system, "--json-schema", json.dumps(schema)]
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=workdir)
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {(proc.stderr or proc.stdout)[:500]}")
        out = json.loads(proc.stdout)
        if out.get("is_error"):
            raise RuntimeError(f"claude CLI error: {out.get('result', '')[:500]}")
        structured = out.get("structured_output")
        return structured if structured is not None else json.loads(out["result"])
    return call


def render_board(fen: str) -> str:
    """ASCII board from the SIDE TO MOVE's view (own pieces at the bottom), with coordinates.
    Uppercase = own, lowercase = enemy — matching the net's own/enemy frame."""
    board = chess.Board(fen)
    us = board.turn
    ranks = range(7, -1, -1) if us == chess.WHITE else range(8)
    files = range(8) if us == chess.WHITE else range(7, -1, -1)
    lines = []
    for r in ranks:
        row = []
        for f in files:
            p = board.piece_at(chess.square(f, r))
            ch = "." if p is None else (p.symbol().upper() if p.color == us else p.symbol().lower())
            row.append(ch)
        lines.append(f"{r + 1} " + " ".join(row))
    lines.append("  " + " ".join(chess.FILE_NAMES[f] for f in files))
    side = "White" if us == chess.WHITE else "Black"
    return f"Side to move: {side} (own = {side}; uppercase = own, lowercase = enemy)\n" + "\n".join(lines)


def _firing_text(fen: str, acts: torch.Tensor, top: int = 3) -> str:
    """'e4 (own pawn) 3.21; d5 (empty) 1.10' for the strongest firing tokens of one board."""
    order = torch.argsort(acts, descending=True)[:top]
    parts = [f"{chess.square_name(token_to_square(fen, int(t)))} ({describe_firing_piece(fen, int(t))}) {float(acts[t]):.2f}"
             for t in order if acts[t] > 0]
    return "; ".join(parts)


def explain_prompt(fens: Sequence[str], acts: torch.Tensor, boards: Sequence[int]) -> str:
    blocks = []
    for n, b in enumerate(boards, 1):
        blocks.append(f"Example {n}: FEN {fens[b]}\n{render_board(fens[b])}\nFeature fires on: {_firing_text(fens[b], acts[b])}")
    return (
        "Below are chess positions where one feature of the network is active, strongest first, "
        "followed by a few weaker activations. For each, the squares where it fires are listed with "
        "the piece on that square (own/enemy relative to the side to move) and the activation.\n\n"
        + "\n\n".join(blocks)
        + "\n\nWhat single chess concept does this feature detect on the squares where it fires? "
        "Consider piece identity, attacks/defence, legal moves, king safety, pawn structure, tactics. "
        "Give a short, specific, testable label."
    )


def detect_prompt(label: str, fens: Sequence[str], boards: Sequence[int]) -> str:
    blocks = [f"Board {n}: FEN {fens[b]}\n{render_board(fens[b])}" for n, b in enumerate(boards)]
    return (
        f"A feature of the network has been described as:\n\n    \"{label}\"\n\n"
        "For each board below, predict whether the feature fires anywhere on it, and if so the single "
        "square where it most likely fires most strongly (algebraic, e.g. 'e4'). Roughly half of the "
        "boards contain the feature.\n\n" + "\n\n".join(blocks)
    )


def interpret_feature(codes: SparseCodes, fens: Sequence[str], feature: int, llm: LLM,
                      n_top: int = 10, n_random: int = 5, n_detect: int = 10, seed: int = 0) -> dict:
    """Explain + detect for one feature. Returns label, scores and the boards used."""
    rng = random.Random(f"{seed}:{feature}")
    acts = codes.feature(feature)                                   # (N, 64)
    peak = acts.amax(1)                                             # (N,)
    active = torch.nonzero(peak > 0).flatten().tolist()
    if len(active) < n_top + n_random + n_detect // 2:
        return {"feature": feature, "skipped": f"only {len(active)} active boards"}
    by_peak = sorted(active, key=lambda b: -float(peak[b]))
    top = by_peak[:n_top]
    rest = by_peak[n_top:]
    rng.shuffle(rest)
    weak = rest[:n_random]
    held_pos = rest[n_random:n_random + n_detect // 2]              # never shown during EXPLAIN
    inactive = torch.nonzero(peak == 0).flatten().tolist()
    held_neg = rng.sample(inactive, min(n_detect - len(held_pos), len(inactive)))

    exp = llm(SYSTEM, explain_prompt(fens, acts, top + weak), LABEL_SCHEMA)
    detect_boards = held_pos + held_neg
    rng.shuffle(detect_boards)
    det = llm(SYSTEM, detect_prompt(exp["label"], fens, detect_boards), DETECT_SCHEMA)

    preds = {p["board"]: p for p in det["predictions"]}
    tp = fn = tn = fp = hits = 0
    for n, b in enumerate(detect_boards):
        pred = preds.get(n, {"fires": False, "square": "none"})
        truth = peak[b] > 0
        if truth:
            tp += pred["fires"]
            fn += not pred["fires"]
            firing = {chess.square_name(token_to_square(fens[b], int(t))) for t in torch.nonzero(acts[b] > 0).flatten()}
            hits += pred["square"].strip().lower() in firing
        else:
            tn += not pred["fires"]
            fp += pred["fires"]
    n_pos, n_neg = tp + fn, tn + fp
    return {
        "feature": feature,
        "label": exp["label"],
        "reasoning": exp["reasoning"],
        "balanced_accuracy": 0.5 * (tp / max(n_pos, 1) + tn / max(n_neg, 1)),
        "square_hit_rate": hits / max(n_pos, 1),
        "n_detect_pos": n_pos,
        "n_detect_neg": n_neg,
        "explain_boards": top + weak,
        "detect_boards": detect_boards,
    }

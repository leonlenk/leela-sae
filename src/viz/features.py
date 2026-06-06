"""Eyeball helpers for the "believe one feature" check (agent-fillable 🤖).

These do the BORING part of DESIGN.md §4 step 5: once you have a list of top-activating
(fen, square, activation) triples for one SAE feature, render each board with the firing square
marked so you can read the shared chess concept off them by eye. No SAE logic lives here — this is
pure presentation, fed by src/eval/feature_mining.py.

Two renderers, same data:
  * render_square()        -> ASCII (always works, used in tests / plain terminals)
  * board_svg() + show_top_activations() -> real chess.com-style SVG boards for the notebook.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import chess
import chess.svg

# One "result" is what src/eval/feature_mining.top_activating returns: which board, which of its 64
# TOKENS fired, and how hard. The int is a Leela residual-stream TOKEN index (0..63), NOT yet a
# python-chess square — token_to_square() does that conversion (it depends on whose move it is).
Activation = Tuple[str, int, float]  # (fen, token_index, activation)

# chess.com board palette (light/dark squares), plus the yellow they use to highlight a square.
# chess.svg lets us override the theme via its `colors` dict; `fill` paints individual squares.
CHESSCOM_COLORS = {
    "square light": "#ebecd0",   # cream
    "square dark": "#739552",    # green
    "margin": "#262421",         # dark border behind the coordinates
    "coord": "#ffffff",
}
HIGHLIGHT = "#f6c343cc"          # warm amber w/ alpha — the "this square fired" marker


def token_to_square(fen: str, token: int) -> int:
    """Map a Leela residual-stream token index (0..63) to the python-chess square it covers.

    Leela orients the board from the SIDE-TO-MOVE's view. For WHITE to move the 64 tokens are
    rank-major a1..h8, so token index == python-chess square. For BLACK to move lczerolens
    vertically flips the board (board.py to_config_tensor: `.flip(1)`), so token t covers the
    rank-mirrored square — chess.square_mirror(t) == t ^ 56. Verified empirically with a Kg1 probe:
    white Kg1 -> token 6 (=G1); flip to black-to-move -> token 62 (=square_mirror(6)).
    """
    return token if chess.Board(fen).turn == chess.WHITE else chess.square_mirror(token)


def describe_firing_piece(fen: str, token: int) -> str:
    """Human label for the piece on the firing square, in Leela's SIDE-TO-MOVE frame ("own"/"enemy").

    Leela sees the board from the mover's view, so the *meaningful* split is own-vs-enemy, not
    white-vs-black: a side-to-move-relative feature ("enemy pawn in my half") lands on a black pawn
    when White moves and a white pawn when Black moves — same concept, opposite absolute colour. We
    compare piece.color to board.turn to collapse that flip. Returns e.g. "enemy pawn", or "empty".
    """
    board = chess.Board(fen)
    square = token_to_square(fen, token)              # token index -> real square (mirror if black)
    piece = board.piece_at(square)
    if piece is None:
        return "empty"
    side = "own" if piece.color == board.turn else "enemy"   # relative to whoever is to move
    return f"{side} {chess.piece_name(piece.piece_type)}"


def board_svg(fen: str, token: int, size: int = 320, relative: bool = False) -> str:
    """Render `fen` as a chess.com-styled SVG with the firing TOKEN's square highlighted amber.

    Returns the raw SVG markup as a string. `token` is a Leela residual token index; we convert it
    to the real python-chess square via token_to_square so the highlight is correct for both colours
    to move.

    `relative` picks the viewing frame — and it matters for reading Leela features:
      * False (default): fixed White orientation. Squares line up across boards (a passed pawn on b5
        is on b5 everywhere), but absolute piece COLOUR flips with whose move it is, so a single
        side-to-move-relative feature looks like it fires on "both colours".
      * True: orient from the SIDE TO MOVE (board.turn). Now "my pawns" sit at the bottom in every
        board, so an us/them feature reads as ONE concept instead of smearing across white & black.
    """
    board = chess.Board(fen)
    square = token_to_square(fen, token)   # token index -> real square (mirror if black to move)
    orientation = board.turn if relative else chess.WHITE
    return chess.svg.board(
        board,
        orientation=orientation,           # side-to-move view (relative) or fixed White (absolute)
        fill={square: HIGHLIGHT},          # paint just the firing square
        colors=CHESSCOM_COLORS,
        size=size,
        coordinates=True,
    )


class _Gallery:
    """Tiny holder so Jupyter renders our HTML inline (it calls `_repr_html_`); `str()` for the rest."""

    def __init__(self, html: str) -> None:
        self.html = html

    def _repr_html_(self) -> str:   # noqa: D401 — Jupyter display protocol hook
        return self.html

    def __str__(self) -> str:
        return self.html


def render_square(fen: str, token: int, marker: str = "*") -> str:
    """ASCII-render the board for `fen` with the firing TOKEN's square highlighted in [brackets].

    `token` is a Leela residual-stream token index (what feature_mining returns); we convert it to
    the real python-chess square via token_to_square so the mark is correct for BOTH colours to move.
    """
    board = chess.Board(fen)
    square = token_to_square(fen, token)   # token index -> real square (mirror if black to move)
    sq_name = chess.square_name(square)

    lines = [f"{fen}", f"  feature fires on {sq_name} (marked):"]
    # Ranks print top (8) to bottom (1); files a..h left to right — i.e. White's view.
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            ch = piece.symbol() if piece else "."        # P/N/B... for white, lowercase for black
            cells.append(f"[{ch}]" if sq == square else f" {ch} ")
        lines.append(f"{rank + 1} " + "".join(cells))
    lines.append("   " + "  ".join("abcdefgh"))
    return "\n".join(lines)


def show_top_activations(
    feature: int,
    results: Iterable[Activation],
    marker: str = "*",
    size: int = 300,
    relative: bool = False,
):
    """Lay the top-activating positions out as a grid of chess.com-style boards for the notebook.

    Returns a `_Gallery` whose `_repr_html_` makes Jupyter draw it inline. Each card shows the rank,
    the activation, the firing square + side to move, and the SVG board with that square highlighted.

    `relative=True` flips every board to the SIDE-TO-MOVE's view and labels the firing piece
    own/enemy instead of white/black (see board_svg / describe_firing_piece). Use it when a feature
    looks like it fires on "both colours" — that's usually one us/them concept seen in the absolute
    frame, and the relative view collapses it back to a single readable concept.
    """
    results = list(results)
    cards = []
    for rank, (fen, token, act) in enumerate(results, start=1):
        square = token_to_square(fen, token)
        sq_name = chess.square_name(square)
        turn = "white to move" if chess.Board(fen).turn == chess.WHITE else "black to move"
        # In the relative view, colour is meaningless (we flipped the board) — show own/enemy + piece.
        subtitle = describe_firing_piece(fen, token) if relative else turn
        svg = board_svg(fen, token, size=size, relative=relative)
        # One card: a header line (rank / activation / square) above the board SVG.
        cards.append(
            "<div style='display:inline-block;margin:8px;text-align:center;"
            "font-family:system-ui,sans-serif;vertical-align:top'>"
            f"<div style='font-weight:600;color:#312e2b'>#{rank} &middot; act {act:.3f}</div>"
            f"<div style='font-size:12px;color:#666;margin-bottom:4px'>fires on "
            f"<b>{sq_name}</b> &middot; {subtitle}</div>"
            f"{svg}</div>"
        )

    frame = "side-to-move view" if relative else "white's view"
    header = (
        f"<h3 style='font-family:system-ui,sans-serif;color:#312e2b'>"
        f"feature #{feature}: top {len(results)} activating positions "
        f"<span style='font-size:13px;font-weight:400;color:#666'>({frame})</span></h3>"
    )
    return _Gallery(header + "<div>" + "".join(cards) + "</div>")

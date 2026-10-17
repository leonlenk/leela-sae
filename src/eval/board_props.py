"""Ground-truth chess properties for the correlational eval (python-chess; side-to-move relative).

INTERP-SAFETY: this file IS the ground truth. A wrong label here manufactures a fake "finding", so
every property has a hand-checkable test in tests/test_board_props.py.

Two kinds of property:
  * PER-SQUARE — returned as a (64,) bool array in LEELA TOKEN ORDER (index t == residual token t),
    so labels line up with activations with no further mapping. For White to move token t is square
    t; for Black to move the board is mirrored vertically (token t is chess.square_mirror(t)).
  * BOARD-LEVEL — one bool per position.

Every property is in Leela's frame: "own" = the side to move, "enemy" = the other side.

Each family is tagged with where its information comes from, because that decides what a match
means:
  * "input"   — literally an input plane (piece identity, castling rights). The residual stream
                carries these forward from layer 0; an SAE feature for them is a sanity check, not
                a discovery.
  * "derived" — a function of the board the net has to COMPUTE (attacks, pins, hanging pieces,
                legal destinations, checks, passed pawns, ...). These are the interesting ones.
  * "hidden"  — not visible under the canonical history-free encoding at all (en passant: there is
                no ep plane; lc0 infers it from history). Only meaningful with real history.
"""

from __future__ import annotations

import chess
import numpy as np

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_PIECES = [(chess.PAWN, "pawn"), (chess.KNIGHT, "knight"), (chess.BISHOP, "bishop"),
           (chess.ROOK, "rook"), (chess.QUEEN, "queen"), (chess.KING, "king")]

SQUARE_FAMILIES: dict[str, str] = {
    **{f"{side}_{name}": "input" for side in ("own", "enemy") for _, name in _PIECES},
    "empty": "input",
    "attacked_by_own": "derived",          # some own piece attacks this square
    "attacked_by_enemy": "derived",
    "own_hanging": "derived",              # own non-king piece, attacked by enemy, defended by nobody
    "enemy_hanging": "derived",            # enemy non-king piece we attack and they don't defend
    "own_en_prise": "derived",             # own piece attacked by a cheaper enemy piece, or hanging
    "enemy_en_prise": "derived",
    "own_pinned": "derived",               # own piece absolutely pinned to own king
    "enemy_pinned": "derived",
    "legal_from": "derived",               # own piece with at least one legal move
    "legal_to": "derived",                 # destination of at least one own legal move
    "capture_target": "derived",           # enemy piece we can legally capture
    "check_square": "derived",             # some own legal move landing here gives check
    "mate_square": "derived",              # some own legal move landing here gives checkmate
    "own_passed_pawn": "derived",
    "enemy_passed_pawn": "derived",
    "own_king_zone": "derived",            # own king's square and its neighbours
    "enemy_king_zone": "derived",
    "own_king_in_check": "derived",        # the own king's square, when in check
}

BOARD_FAMILIES: dict[str, str] = {
    "own_can_castle_kingside": "input",
    "own_can_castle_queenside": "input",
    "enemy_can_castle_kingside": "input",
    "enemy_can_castle_queenside": "input",
    "is_check": "derived",
    "has_mate_in_one": "derived",
    "material_ahead": "derived",           # own material >= enemy + 2 (P=1 N=B=3 R=5 Q=9)
    "material_behind": "derived",
    "queens_on": "derived",                # at least one queen on the board
    "endgame": "derived",                  # <= 12 pieces in total (incl. kings)
    "legal_en_passant": "hidden",          # an ep capture is legal right now
}


def token_order(fen_or_board) -> np.ndarray:
    """perm[t] = python-chess square covered by Leela token t (identity for White, mirror for Black)."""
    board = fen_or_board if isinstance(fen_or_board, chess.Board) else chess.Board(fen_or_board)
    t = np.arange(64)
    return t if board.turn == chess.WHITE else t ^ 56   # square_mirror(t) == t ^ 56


def _passed_pawns(board: chess.Board, color: bool) -> chess.SquareSet:
    """Pawns of `color` with no enemy pawn ahead of them on their own or an adjacent file."""
    enemy_pawns = board.pieces(chess.PAWN, not color)
    passed = chess.SquareSet()
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = range(r + 1, 8) if color == chess.WHITE else range(0, r)
        blockers = [chess.square(ff, rr) for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead]
        if not any(b in enemy_pawns for b in blockers):
            passed.add(sq)
    return passed


def square_properties(board: chess.Board) -> dict[str, np.ndarray]:
    """{family: (64,) bool in TOKEN order} for every family in SQUARE_FAMILIES."""
    us, them = board.turn, not board.turn
    by_square = {name: np.zeros(64, dtype=bool) for name in SQUARE_FAMILIES}

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        att_us, att_them = board.attackers_mask(us, sq), board.attackers_mask(them, sq)
        by_square["attacked_by_own"][sq] = att_us != 0
        by_square["attacked_by_enemy"][sq] = att_them != 0
        if piece is None:
            by_square["empty"][sq] = True
            continue
        side = "own" if piece.color == us else "enemy"
        by_square[f"{side}_{chess.piece_name(piece.piece_type)}"][sq] = True
        if piece.piece_type == chess.KING:
            continue
        attackers, defenders = (att_them, att_us) if piece.color == us else (att_us, att_them)
        hanging = attackers != 0 and defenders == 0
        # Cheapest NON-king attacker: a king can only take an undefended piece, which `hanging`
        # already covers — so a king next to a defended piece must not make it "en prise".
        cheapest = min((PIECE_VALUES[board.piece_type_at(a)] for a in chess.SquareSet(attackers)
                        if board.piece_type_at(a) != chess.KING), default=None)
        by_square[f"{side}_hanging"][sq] = hanging
        by_square[f"{side}_en_prise"][sq] = hanging or (
            cheapest is not None and cheapest < PIECE_VALUES[piece.piece_type])
        by_square[f"{side}_pinned"][sq] = board.is_pinned(piece.color, sq)

    for move in board.legal_moves:
        by_square["legal_from"][move.from_square] = True
        by_square["legal_to"][move.to_square] = True
        if board.is_capture(move) and not board.is_en_passant(move):
            by_square["capture_target"][move.to_square] = True
        if board.gives_check(move):
            by_square["check_square"][move.to_square] = True
            board.push(move)
            if board.is_checkmate():
                by_square["mate_square"][move.to_square] = True
            board.pop()

    for color, side in ((us, "own"), (them, "enemy")):
        for sq in _passed_pawns(board, color):
            by_square[f"{side}_passed_pawn"][sq] = True
        king = board.king(color)
        if king is not None:
            by_square[f"{side}_king_zone"][king] = True
            for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king]):
                by_square[f"{side}_king_zone"][sq] = True
    if board.is_check():
        by_square["own_king_in_check"][board.king(us)] = True

    perm = token_order(board)
    return {name: arr[perm] for name, arr in by_square.items()}   # absolute squares -> token order


def board_properties(board: chess.Board) -> dict[str, bool]:
    """{family: bool} for every family in BOARD_FAMILIES."""
    us, them = board.turn, not board.turn
    material = sum((1 if p.color == us else -1) * PIECE_VALUES[p.piece_type] for p in board.piece_map().values())
    has_mate = False
    for move in board.legal_moves:
        if board.gives_check(move):
            board.push(move)
            has_mate = board.is_checkmate()
            board.pop()
            if has_mate:
                break
    return {
        "own_can_castle_kingside": board.has_kingside_castling_rights(us),
        "own_can_castle_queenside": board.has_queenside_castling_rights(us),
        "enemy_can_castle_kingside": board.has_kingside_castling_rights(them),
        "enemy_can_castle_queenside": board.has_queenside_castling_rights(them),
        "is_check": board.is_check(),
        "has_mate_in_one": has_mate,
        "material_ahead": material >= 2,
        "material_behind": material <= -2,
        "queens_on": bool(board.pieces_mask(chess.QUEEN, chess.WHITE) | board.pieces_mask(chess.QUEEN, chess.BLACK)),
        "endgame": chess.popcount(board.occupied) <= 12,
        "legal_en_passant": board.has_legal_en_passant(),
    }


def label_arrays(fens) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Labels for many positions: ({family: (N, 64) bool, token order}, {family: (N,) bool})."""
    sq = {name: np.zeros((len(fens), 64), dtype=bool) for name in SQUARE_FAMILIES}
    bd = {name: np.zeros(len(fens), dtype=bool) for name in BOARD_FAMILIES}
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        for name, arr in square_properties(board).items():
            sq[name][i] = arr
        for name, val in board_properties(board).items():
            bd[name][i] = val
    return sq, bd

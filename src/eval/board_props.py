"""Feature → python-chess board-property mapping (🔒 — scaffold per §3.5; the chess-specific result lives here).

Hand-write the first dozen properties yourself to fix the PATTERN before any agent enumerates the
~1000 (DESIGN.md §3). Each property is a deterministic fact about the position that a feature might
track. INTERP-SAFETY: this is ground truth — a wrong property here manufactures a fake "finding".
"""

import chess

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def board_properties(board: "chess.Board") -> dict:
    """Compute ground-truth properties for one position.

    Returns a flat dict of {property_name: bool|int}. The matching test pins a handful of these by
    name on known FENs — keep these key names stable or update the test alongside.

    Start with this dozen (each TODO is one property; add per-square occupancy last, it's the big one):
    """
    props: dict = {}

    us, them = board.turn, not board.turn
    rel = (lambda sq: sq) if us == chess.WHITE else chess.square_mirror   # absolute sq -> side-to-move view

    props["is_check"] = board.is_check()

    props["own_king_square"] = board.king(us)
    props["enemy_king_square"] = board.king(them)

    props["own_king_castled_kingside"] = (board.king(us) == rel(chess.G1))   # g1 for white, g8 for black

    props["own_can_castle_kingside"] = board.has_kingside_castling_rights(us)
    props["enemy_can_castle_kingside"] = board.has_kingside_castling_rights(them)

    props["en_passant_available"] = (board.ep_square is not None)

    props["own_queen_count"] = len(board.pieces(chess.QUEEN, us))
    props["enemy_queen_count"] = len(board.pieces(chess.QUEEN, them))

    material = 0
    for sq, piece in board.piece_map().items():
        value = _PIECE_VALUES[piece.piece_type]
        material += value if piece.color == chess.WHITE else -value
    props["material_diff"] = material
    props["material_diff_relative"] = material if us == chess.WHITE else -material   # own minus enemy

    props["e4_occupied_by_own_pawn"] = (board.piece_at(rel(chess.E4)) == chess.Piece(chess.PAWN, us))

    for sq in chess.SQUARES:                            # 0..63, a1..h8
        name = chess.square_name(sq)                   # "a1".."h8"
        piece = board.piece_at(sq)                     # Piece | None
        props[f"{name}_occupied"] = (piece is not None)
        for color, color_name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            # leela doesn't have white v black only own vs enemy
            rel_name = "own" if color == board.turn else "enemy"
            for ptype in chess.PIECE_TYPES:            # PAWN..KING (1..6)
                piece_name = chess.piece_name(ptype)
                match = (
                    piece is not None
                    and piece.color == color
                    and piece.piece_type == ptype
                )
                props[f"{name}_{rel_name}_{piece_name}"] = match      # e.g. "e4_enemy_pawn"

    return props

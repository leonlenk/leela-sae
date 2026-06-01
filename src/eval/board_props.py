"""Feature → python-chess board-property mapping (🔒 — scaffold per §3.5; the chess-specific result lives here).

Hand-write the first dozen properties yourself to fix the PATTERN before any agent enumerates the
~1000 (DESIGN.md §3). Each property is a deterministic fact about the position that a feature might
track. INTERP-SAFETY: this is ground truth — a wrong property here manufactures a fake "finding".
"""

import chess


def board_properties(board: "chess.Board") -> dict:
    """Compute ground-truth properties for one position.

    Returns a flat dict of {property_name: bool|int}. The matching test pins a handful of these by
    name on known FENs — keep these key names stable or update the test alongside.

    Start with this dozen (each TODO is one property; add per-square occupancy last, it's the big one):
    """
    props: dict = {}
    # TODO (1): props["side_to_move_white"] = (board.turn == chess.WHITE)
    # TODO (2): props["is_check"] = board.is_check()
    # TODO (3): props["white_king_square"] = board.king(chess.WHITE)        # 0..63 (square index)
    # TODO (4): props["white_king_on_g1"] = (board.king(chess.WHITE) == chess.G1)  # the canary property
    # TODO (5): props["black_king_on_g8"] = (board.king(chess.BLACK) == chess.G8)
    # TODO (6): props["white_can_castle_kingside"] = board.has_kingside_castling_rights(chess.WHITE)
    # TODO (7): props["en_passant_available"] = (board.ep_square is not None)
    # TODO (8): props["white_queen_count"] = len(board.pieces(chess.QUEEN, chess.WHITE))
    # TODO (9): props["material_diff"] = <white material> - <black material> in centipawns/points
    # TODO (10): props["e4_occupied_by_white_pawn"] = (board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE))
    # TODO (11): per-square occupancy — for sq in chess.SQUARES, encode piece-type/colour (the ~1000-way
    #            enumeration; do this LAST, once the dozen above feel mechanical).
    # TODO (12): return props
    raise NotImplementedError

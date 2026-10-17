"""Ground-truth property tests. Pure python-chess, no net. INTERP-SAFETY: each assertion's comment is
the hand-checkable chess fact it encodes — paste the FEN into any board viewer to confirm it.

Per-square labels come back in LEELA TOKEN ORDER: for White to move token t == square t; for Black
to move token t == chess.square_mirror(t). The helper `at` converts a square name to its token.
"""

import chess
import numpy as np

from src.eval.board_props import (BOARD_FAMILIES, SQUARE_FAMILIES, board_properties, label_arrays,
                                  square_properties, token_order)


def at(board: chess.Board, square_name: str) -> int:
    """Token index that covers `square_name` for this side to move."""
    sq = chess.parse_square(square_name)
    return sq if board.turn == chess.WHITE else chess.square_mirror(sq)


def squares_where(board, props, family):
    """Set of square NAMES where the (token-ordered) family is True."""
    perm = token_order(board)
    return {chess.square_name(perm[t]) for t in np.flatnonzero(props[family])}


def test_every_family_present_with_token_shape():
    b = chess.Board()
    sq = square_properties(b)
    assert set(sq) == set(SQUARE_FAMILIES) and all(v.shape == (64,) and v.dtype == bool for v in sq.values())
    assert set(board_properties(b)) == set(BOARD_FAMILIES)


def test_piece_labels_are_side_to_move_relative_and_token_ordered():
    # Black to move: OWN king = black king on e8, ENEMY rook = white rook on a1.
    b = chess.Board("4k3/8/8/8/8/8/8/R3K3 b - - 0 1")
    p = square_properties(b)
    assert squares_where(b, p, "own_king") == {"e8"} and squares_where(b, p, "enemy_rook") == {"a1"}
    # The black king on e8 is token 4 for Black to move (board flipped vertically).
    assert p["own_king"][4] and at(b, "e8") == 4


def test_starting_position_moves_and_attacks():
    b = chess.Board()
    p = square_properties(b)
    # Hand-checkable: 16 pawn pushes land on ranks 3-4; knights reach a3/c3/f3/h3 (already in rank 3).
    assert squares_where(b, p, "legal_to") == {f + r for f in "abcdefgh" for r in "34"}
    assert squares_where(b, p, "legal_from") == {f + "2" for f in "abcdefgh"} | {"b1", "g1"}
    # Every rank-3 square is covered by a white pawn; nothing reaches rank 4 yet.
    assert {f + "3" for f in "abcdefgh"} <= squares_where(b, p, "attacked_by_own")
    assert not squares_where(b, p, "attacked_by_own") & {f + "4" for f in "abcdefgh"}
    assert squares_where(b, p, "own_hanging") == set() and squares_where(b, p, "check_square") == set()


def test_hanging_and_capture_target():
    # White Qd1 sees straight up the open d-file to the black knight on d5; nothing defends d5.
    b = chess.Board("4k3/8/8/3n4/8/8/8/3QK3 w - - 0 1")
    p = square_properties(b)
    assert squares_where(b, p, "enemy_hanging") == {"d5"}
    assert "d5" in squares_where(b, p, "capture_target")
    assert squares_where(b, p, "own_hanging") == set()      # the knight doesn't reach d1 or e1


def test_en_prise_by_a_cheaper_attacker_even_when_defended():
    # White pawn c4 attacks the black queen d5, which the e6 pawn defends: not hanging, but en prise.
    b = chess.Board("4k3/8/4p3/3q4/2P5/8/8/4K3 w - - 0 1")
    p = square_properties(b)
    assert "d5" in squares_where(b, p, "enemy_en_prise") and "d5" not in squares_where(b, p, "enemy_hanging")


def test_king_next_to_defended_piece_is_not_en_prise():
    # White Ke1 touches the black knight d2, but the c3 pawn defends it: the king can't take it.
    b = chess.Board("4k3/8/8/8/8/2p5/3n4/4K3 w - - 0 1")
    p = square_properties(b)
    assert "d2" not in squares_where(b, p, "enemy_en_prise") | squares_where(b, p, "enemy_hanging")


def test_absolute_pin_blocks_the_pinned_piece():
    # Black Re7 pins the white Ne2 to the white Ke1: pinned, and it has no legal move.
    b = chess.Board("4k3/4r3/8/8/8/8/4N3/4K3 w - - 0 1")
    p = square_properties(b)
    assert squares_where(b, p, "own_pinned") == {"e2"} and "e2" not in squares_where(b, p, "legal_from")


def test_check_and_mate_squares():
    # Ra1-a8+ is the only check (king e8 escapes to d7/e7/f7, so no mate).
    b = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    p = square_properties(b)
    assert squares_where(b, p, "check_square") == {"a8"} and squares_where(b, p, "mate_square") == set()
    assert board_properties(b)["has_mate_in_one"] is False
    # Back-rank mate: the f7/g7/h7 pawns box in the g8 king, so Ra8# mates.
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert squares_where(b, square_properties(b), "mate_square") == {"a8"}
    assert board_properties(b)["has_mate_in_one"] is True


def test_king_in_check_marks_the_king_square_only():
    # Fool's-mate check: black Qh4 checks the white Ke1.
    b = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert squares_where(b, square_properties(b), "own_king_in_check") == {"e1"}
    assert board_properties(b)["is_check"] is True
    assert squares_where(chess.Board(), square_properties(chess.Board()), "own_king_in_check") == set()


def test_passed_pawns_for_both_colours():
    # White d5: the only black pawn (c2) is behind it -> passed. Black c2: no white pawn on b/c/d
    # below rank 2 -> passed.
    b = chess.Board("4k3/8/8/3P4/8/8/2p5/4K3 w - - 0 1")
    p = square_properties(b)
    assert squares_where(b, p, "own_passed_pawn") == {"d5"} and squares_where(b, p, "enemy_passed_pawn") == {"c2"}
    # Add a black pawn on e6: it sits on an adjacent file ahead of d5, so d5 is no longer passed.
    b = chess.Board("4k3/8/4p3/3P4/8/8/2p5/4K3 w - - 0 1")
    assert "d5" not in squares_where(b, square_properties(b), "own_passed_pawn")


def test_king_zone():
    # White Ka1: its zone is a1 plus the three neighbours a2, b1, b2.
    b = chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")
    assert squares_where(b, square_properties(b), "own_king_zone") == {"a1", "a2", "b1", "b2"}


def test_board_level_castling_material_phase():
    start = board_properties(chess.Board())
    assert start["own_can_castle_kingside"] and start["enemy_can_castle_queenside"]
    assert not start["material_ahead"] and not start["endgame"] and start["queens_on"]
    # White to move with an extra rook (5 points up), 3 pieces total.
    up = board_properties(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"))
    assert up["material_ahead"] and not up["material_behind"] and up["endgame"] and not up["queens_on"]
    # Same board, Black to move: now the mover is the one 5 points down.
    assert board_properties(chess.Board("4k3/8/8/8/8/8/8/R3K3 b - - 0 1"))["material_behind"]


def test_legal_en_passant_uses_legality_not_the_raw_ep_square():
    # After 1.e4 the FEN ep square is e3, but no black pawn can capture there.
    b = chess.Board()
    b.push_san("e4")
    assert board_properties(b)["legal_en_passant"] is False
    # White pawn e5, black just played d7-d5: exd6 e.p. is legal.
    assert board_properties(chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"))["legal_en_passant"] is True


def test_label_arrays_stack_per_position():
    fens = [chess.STARTING_FEN, "4k3/8/8/8/8/8/8/R3K3 b - - 0 1"]
    sq, bd = label_arrays(fens)
    assert sq["own_king"].shape == (2, 64) and bd["is_check"].shape == (2,)
    assert sq["own_king"][0, 4] and sq["own_king"][1, 4]   # e1 for White, e8 (token 4) for Black

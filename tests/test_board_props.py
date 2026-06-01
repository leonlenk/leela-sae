"""Board-property tests, written FIRST (§3.5a). Pure python-chess, no net — runs red now, green once
you implement src/eval/board_props.py. INTERP-SAFETY (§3.5 carve-out): each assertion states the
hand-checkable chess fact it encodes — confirm it on a real board before trusting a green bar.
"""

import chess

from src.eval.board_props import board_properties

# Position after 1.e4 e5 2.Nf3 Nc6 3.Bc4 then White castles: White king lands on g1.
# Hand-checkable: paste this FEN into any board viewer — the white king is on g1.
WHITE_CASTLED_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4"
STARTING_FEN = chess.STARTING_FEN


def test_white_king_on_g1_true_after_castling():
    props = board_properties(chess.Board(WHITE_CASTLED_FEN))
    # Hand-checkable fact: White has castled kingside here, so the king IS on g1.
    assert props["white_king_on_g1"] is True


def test_white_king_on_g1_false_at_start():
    props = board_properties(chess.Board(STARTING_FEN))
    # Hand-checkable fact: in the starting position the white king is on e1, not g1.
    assert props["white_king_on_g1"] is False


def test_side_to_move_flag():
    # Starting position: White to move.
    assert board_properties(chess.Board(STARTING_FEN))["side_to_move_white"] is True
    # After 1.e4 it is Black to move.
    after_e4 = chess.Board(STARTING_FEN)
    after_e4.push_san("e4")
    assert board_properties(after_e4)["side_to_move_white"] is False


def test_is_check_detects_a_check():
    # Fool's-mate-ish check: black queen on h4 gives check to the white king on e1.
    # Hand-checkable: "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3" is check.
    check_board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board_properties(check_board)["is_check"] is True
    assert board_properties(chess.Board(STARTING_FEN))["is_check"] is False

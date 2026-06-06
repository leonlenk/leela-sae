"""Board-property tests, written FIRST (§3.5a). Pure python-chess, no net — runs red now, green once
you implement src/eval/board_props.py. INTERP-SAFETY (§3.5 carve-out): each assertion states the
hand-checkable chess fact it encodes — confirm it on a real board before trusting a green bar.
"""

import chess

from src.eval.board_props import board_properties

# Position after 1.e4 e5 2.Nf3 Nc6 3.Bc4 then White castles: White king lands on g1.
# NOTE: it is BLACK to move here ("... b kq ..."), so in Leela's side-to-move frame the white
# king is the ENEMY king. Hand-checkable: paste the FEN into a viewer — white king is on g1.
WHITE_CASTLED_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4"
STARTING_FEN = chess.STARTING_FEN


def test_own_king_castled_kingside_when_mover_has_castled():
    # board_props is side-to-move RELATIVE: own_king_castled_kingside means "the player TO MOVE has
    # their king on its kingside-castle square (g1 for White, g8 for Black)". So we need a position
    # where the mover is the one who castled. Build it by pushing real SAN: White castles on move 4,
    # then Black replies, leaving WHITE to move with its king on g1.
    board = chess.Board()
    for mv in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "Nf6"]:
        board.push_san(mv)
    # Hand-checkable: after 4.O-O the white king is on g1; after 4...Nf6 it is White's turn again,
    # so the side-to-move (White) is the castled player.
    assert board.turn is chess.WHITE
    assert board_properties(board)["own_king_castled_kingside"] is True


def test_own_king_castled_kingside_false_at_start():
    # Hand-checkable: in the starting position the side-to-move (White) king is on e1, not g1.
    assert board_properties(chess.Board(STARTING_FEN))["own_king_castled_kingside"] is False


def test_persquare_piece_labels_are_relative_to_side_to_move():
    # This replaces the old absolute `white_king_on_g1` / `side_to_move_white` keys: the own/enemy
    # per-square labels ARE the side-to-move information now. Same physical white king on g1 reads as
    # OWN when White moves and ENEMY when Black moves.
    castled = board_properties(chess.Board(WHITE_CASTLED_FEN))   # Black to move
    # Hand-checkable: white king sits on g1; Black is to move, so to the mover it is the ENEMY king.
    assert castled["g1_enemy_king"] is True
    assert castled["g1_own_king"] is False

    start = board_properties(chess.Board(STARTING_FEN))          # White to move
    # Hand-checkable: White to move -> own king on e1; g1 holds White's (own) knight, not a king.
    assert start["e1_own_king"] is True
    assert start["g1_own_king"] is False
    assert start["g1_own_knight"] is True


def test_color_labels_flip_when_the_turn_flips():
    # The conceptual check the old `side_to_move_white` flag was groping at: who is "own" vs "enemy"
    # tracks whose move it is. The SAME white e-pawn flips label across one ply.
    start = board_properties(chess.Board(STARTING_FEN))          # White to move
    # Hand-checkable: White to move -> White's e2 pawn is OWN.
    assert start["e2_own_pawn"] is True

    after_e4 = chess.Board(STARTING_FEN)
    after_e4.push_san("e4")                                      # now Black to move
    props = board_properties(after_e4)
    # Hand-checkable: after 1.e4 it is Black's move, so that same white pawn (now on e4) is ENEMY.
    assert props["e4_enemy_pawn"] is True


def test_is_check_detects_a_check():
    # Fool's-mate-ish check: black queen on h4 gives check to the white king on e1.
    # Hand-checkable: "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3" is check.
    check_board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board_properties(check_board)["is_check"] is True
    assert board_properties(chess.Board(STARTING_FEN))["is_check"] is False

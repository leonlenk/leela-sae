"""Hook tests. INTERP-SAFETY: each assertion's comment states the hand-checkable fact it encodes — a
wrong test here silently blesses a false activation site or a mis-read policy head.

The first block needs no net (encoding / move-index plumbing); the rest run on the real net and skip
unless LEELA_SAE_CHECKPOINT is set (see conftest.py).
"""

import chess
import torch
from lczerolens import LczeroBoard

from conftest import requires_net
from src import hook
from src.viz.features import token_to_square

RUY_LOPEZ = "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5"
MATE_IN_ONE_FEN = "k7/8/1QK5/8/8/8/8/8 w - - 0 1"
PROMO_WHITE = "8/P6k/8/8/8/8/8/K7 w - - 0 1"
PROMO_BLACK = "k7/8/8/8/8/8/p6K/8 b - - 0 1"
# Black to move, white king on g1 / black king on g8 (for the token-orientation checks).
BLACK_TO_MOVE = "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b - - 2 6"


# ------------------------------------------------------------------------------ no net needed

def test_castling_uses_lc0_king_takes_rook_slot():
    # lc0 encodes O-O as e1h1 and O-O-O as e1a1 (king onto its own rook). python-chess calls them
    # e1g1 / e1c1. policy_index must map the python-chess move onto lc0's slot.
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    oo, ooo = chess.Move.from_uci("e1g1"), chess.Move.from_uci("e1c1")
    assert hook.policy_index(b, oo) == LczeroBoard.encode_move(chess.Move.from_uci("e1h1"), chess.WHITE)
    assert hook.policy_index(b, ooo) == LczeroBoard.encode_move(chess.Move.from_uci("e1a1"), chess.WHITE)
    # Black castles from e8; in the side-to-move frame that is the same e1h1 slot as White's O-O.
    b.turn = chess.BLACK
    assert hook.policy_index(b, chess.Move.from_uci("e8g8")) == LczeroBoard.encode_move(
        chess.Move.from_uci("e8h8"), chess.BLACK)


def test_legal_indices_are_unique_and_cover_every_legal_move():
    b = LczeroBoard(RUY_LOPEZ)
    moves, idx = hook.legal_moves_and_indices(b)
    assert len(moves) == b.legal_moves.count() == len(idx)
    assert len(set(idx.tolist())) == len(idx)          # no two legal moves share a slot


def test_legal_log_probs_normalise_over_legal_moves_only():
    logits = torch.randn(1858)
    idx = torch.tensor([3, 17, 400])
    lp = hook.legal_log_probs(logits, idx)
    assert lp.shape == (3,) and torch.isclose(lp.exp().sum(), torch.tensor(1.0))


def test_legal_kl_is_zero_for_identical_and_positive_for_different_logits():
    idx = torch.tensor([0, 5, 9])
    a = torch.randn(1858)
    assert float(hook.legal_kl(a, a, idx)) == 0.0
    b = a.clone()
    b[5] += 2.0                                          # move mass onto one legal move
    assert float(hook.legal_kl(a, b, idx)) > 0.0
    c = a.clone()
    c[1000] += 50.0                                      # change an ILLEGAL slot only
    assert float(hook.legal_kl(a, c, idx)) == 0.0        # -> invisible to the legal KL, by design


def test_canonical_encoding_repeats_current_board_and_has_no_history():
    # encode_fens builds boards from bare FENs (no move stack), so every history slot is a copy of the
    # current board. (The real-history path is exercised by tests/test_lichess.py and the history
    # filter; this test only pins the canonical encoding.)
    x = hook.encode_fens([RUY_LOPEZ])[0]                 # (112, 8, 8)
    for slot in range(1, 8):
        assert torch.equal(x[slot * 13:(slot + 1) * 13 - 1], x[0:12])   # piece planes repeat


def test_input_planes_put_square_s_at_token_s_in_side_to_move_frame():
    # Checks the INPUT layout only: plane cell (row, col) is square row*8+col in the side-to-move frame.
    # That the net's internal tokens follow the same order (its first op transposes (112,8,8) ->
    # (8,8,112) and flattens) is checked end to end by the canary in tests/test_features.py.
    # Plane 5 = OWN king, plane 11 = ENEMY king. Hand-checkable: white K on g1 (sq 6), black k on g8.
    white_to_move = hook.encode_fens(["6k1/8/8/8/8/8/8/6K1 w - - 0 1"])[0]
    tok_of = lambda plane: int(plane.flatten().nonzero())
    assert tok_of(white_to_move[5]) == 6 == token_to_square("6k1/8/8/8/8/8/8/6K1 w - - 0 1", 6)
    # Black to move: OWN king is now the black king on g8 (sq 62). Board flipped -> token 6, and
    # token_to_square maps token 6 back to g8.
    fen_b = "6k1/8/8/8/8/8/8/6K1 b - - 0 1"
    black_to_move = hook.encode_fens([fen_b])[0]
    assert tok_of(black_to_move[5]) == 6 and token_to_square(fen_b, 6) == chess.G8
    assert tok_of(black_to_move[11]) == 62 and token_to_square(fen_b, 62) == chess.G1


# ------------------------------------------------------------------------------ real net

@requires_net
def test_predicts_known_forced_move(model):
    # Hand-checkable: Qb6-b7 is the only mate-in-one (queen covers a7/a8/b8, Kc6 guards b7).
    assert hook.predict_move(model, MATE_IN_ONE_FEN) == chess.Move.from_uci("b6b7")


@requires_net
def test_predict_move_can_castle(model):
    # Ruy Lopez after 4...Nf6: 5.O-O is the main line. Before the castling fix the policy never put
    # mass on e1g1's (wrong) slot, so predict_move could not return O-O at all.
    assert hook.predict_move(model, RUY_LOPEZ) == chess.Move.from_uci("e1g1")


@requires_net
def test_predict_move_returns_legal_promotions(model):
    # Lone pawn on the 7th, nothing in the way: promoting to a queen is the obvious move. The old
    # decode path returned "a7a8" with no promotion piece — not a legal move.
    for fen, uci in [(PROMO_WHITE, "a7a8q"), (PROMO_BLACK, "a2a1q")]:
        mv = hook.predict_move(model, fen)
        assert mv in chess.Board(fen).legal_moves and mv.uci() == uci


@requires_net
def test_list_residual_modules(model):
    mods = hook.list_residual_modules(model)
    # BT3 has 15 encoder blocks; the per-block residual site is each block's final LN, "encoderN/ln2".
    assert [int(n.split("encoder")[1].split("/")[0]) for n in mods] == list(range(15))
    assert all(n.endswith("/ln2") and "smolgen" not in n for n in mods)


@requires_net
def test_residual_stream_is_per_square(model, resid_module):
    resid = hook.get_residual_stream(model, MATE_IN_ONE_FEN, resid_module)
    assert resid.shape == (64, 768)                      # 64 square-tokens x d_model=768
    assert not resid.requires_grad                       # captured under no_grad, graph not kept alive


@requires_net
def test_capture_matches_nnsight(model, resid_module):
    # Independent capture path (DESIGN.md's original nnsight hooking) must give the same numbers. Both
    # sides use the same (B*64, D) -> (B, 64, D) reshape, so this guards the hook mechanics, not the
    # board/token split — test_batch_matches_one_by_one guards that. Uses both colours to move.
    from nnsight import NNsight
    fens = [RUY_LOPEZ, BLACK_TO_MOVE]
    ours = hook.get_residual_stream_batch(model, fens, resid_module)
    nn_model = NNsight(model)
    with nn_model.trace(hook.encode_fens(fens).to(hook.model_device(model))):
        ref = nn_model.get(resid_module).output.save()
    assert torch.allclose(ours, ref.reshape(2, 64, -1), atol=1e-5)


@requires_net
def test_batch_matches_one_by_one(model, resid_module):
    fens = [MATE_IN_ONE_FEN, BLACK_TO_MOVE, "8/1p6/p2R1b2/P1B2k1p/1P3p1P/2r2b1K/5N2/8 w - - 0 1"]
    batched = hook.get_residual_stream_batch(model, fens, resid_module)
    assert batched.shape == (3, 64, 768)
    for b, fen in enumerate(fens):
        assert torch.allclose(batched[b], hook.get_residual_stream(model, fen, resid_module), atol=1e-4)


@requires_net
def test_identity_patch_is_exact_and_zero_patch_is_not(model, resid_module):
    x = hook.encode_fens([RUY_LOPEZ, BLACK_TO_MOVE])
    _, clean = hook.run_with_cache(model, x, [resid_module])
    same = hook.run_patched(model, x, resid_module, lambda a: a)
    assert torch.equal(same["policy"], clean["policy"]) and torch.equal(same["wdl"], clean["wdl"])
    # Power check: the patch must actually reach the heads, or "identity is exact" proves nothing.
    zeroed = hook.run_patched(model, x, resid_module, torch.zeros_like)
    assert (zeroed["policy"] - clean["policy"]).abs().max() > 0.1


@requires_net
def test_wdl_head_is_probabilities(model):
    wdl = model(hook.encode_fens([RUY_LOPEZ, PROMO_WHITE]).to(hook.model_device(model)))["wdl"]
    assert torch.allclose(wdl.sum(-1), torch.ones(2, device=wdl.device), atol=1e-5)
    # Hand-checkable: K+P vs K with the pawn about to queen is winning for the side to move.
    assert wdl[1, 0] > 0.5


@requires_net
def test_history_dependence_is_measured_and_reported(model):
    # Diagnostic, not a gate: we only require that both perturbations are measured, finite, and that
    # zeroing the history (wildly off-distribution) moves the policy at all.
    d = hook.output_depends_on_history(model, RUY_LOPEZ)
    assert set(d) == {"zero_kl", "randomize_kl", "zero_top1_changed", "randomize_top1_changed"}
    assert d["zero_kl"] > 0 and d["randomize_kl"] > 0


def test_promotion_slots_match_lc0_including_knight_underpromotion():
    # lc0 has explicit slots for q/r/b promotions; a KNIGHT promotion uses the plain from-to slot.
    b = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    plain = LczeroBoard.encode_move(chess.Move.from_uci("a7a8"), chess.WHITE)
    assert hook.policy_index(b, chess.Move.from_uci("a7a8n")) == plain
    slots = {p: hook.policy_index(b, chess.Move.from_uci("a7a8" + p)) for p in "qrbn"}
    assert len(set(slots.values())) == 4                 # four promotions, four distinct slots

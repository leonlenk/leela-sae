"""Tests for the Lichess corpus builder (src/data/lichess.py). No network: synthetic parquet shards."""

import chess
import pandas as pd

from src.data import lichess

# A short real game (Scholar's-mate-ish, 7 plies, ends in mate) and a longer quiet one.
MATE_GAME = "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0"
QUIET_GAME = (
    "1. d4 { [%clk 0:03:00] } 1... d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 "
    "7. Bh4 b6 8. cxd5 exd5 9. Bd3 Bb7 10. O-O c5 1/2-1/2"
)


def _shard(tmp_path, rows):
    df = pd.DataFrame(rows, columns=lichess._META_COLS)
    p = tmp_path / f"shard{len(list(tmp_path.iterdir()))}.parquet"
    df.to_parquet(p)
    return str(p)


def _row(site, movetext, w=2100, b=2100, tc="180+0", wt=None, bt=None):
    return [site, w, b, wt, bt, tc, movetext]


def test_position_key_ignores_move_counters():
    a = chess.Board("8/8/8/8/8/8/4k3/4K3 w - - 0 1")
    b = chess.Board("8/8/8/8/8/8/4k3/4K3 w - - 37 80")
    assert lichess.position_key(a) == lichess.position_key(b)


def test_sampled_position_matches_replayed_history():
    rows = lichess._sample_game(("g1", QUIET_GAME, 2100, 2100, 3, 6, 0))
    assert len(rows) == 3
    for r in rows:
        # The stored FEN must be exactly the board reached by replaying the stored UCI history,
        # and the history length must equal the ply.
        replay = lichess.board_with_history(r["uci_history"])
        assert replay.fen() == r["fen"]
        assert len(r["uci_history"].split()) == r["ply"] >= 6


def test_terminal_positions_are_dropped():
    # The mate game has 7 plies; with min_ply=6 the candidates are plies 6 and 7. Ply 7 is
    # checkmate (no legal moves) and must never be emitted.
    rows = lichess._sample_game(("g2", MATE_GAME, 2100, 2100, 5, 6, 0))
    assert [r["ply"] for r in rows] == [6]


def test_filters_elo_bullet_and_bots(tmp_path):
    p = _shard(tmp_path, [
        _row("keep", QUIET_GAME),
        _row("low_elo", QUIET_GAME, w=1500),
        _row("bullet", QUIET_GAME, tc="60+0"),
        _row("bot", QUIET_GAME, bt="BOT"),
    ])
    corpus = lichess.build_corpus([p], positions_per_game=1, processes=1)
    assert set(corpus["game_id"]) == {"keep"}


def test_heldout_positions_never_duplicate_train(tmp_path):
    # Many copies of the same game under different ids: every sampled position key collides.
    # After dedup each key must appear exactly once, and any surviving duplicate goes to train.
    rows = [_row(f"https://lichess.org/{i}", QUIET_GAME) for i in range(200)]
    corpus = lichess.build_corpus([_shard(tmp_path, rows)], positions_per_game=2,
                                  val_frac=0.3, test_frac=0.3, processes=1)
    assert corpus["key"].is_unique
    train_keys = set(corpus.loc[corpus.split == "train", "key"])
    held = corpus.loc[corpus.split != "train", "key"]
    assert not (set(held) & train_keys)


def test_split_is_deterministic():
    assert lichess.split_of("https://lichess.org/abc", 0.05, 0.05) == lichess.split_of(
        "https://lichess.org/abc", 0.05, 0.05
    )

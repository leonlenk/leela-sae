"""Lichess position corpus: sample positions (with their real move history) from rated human games (🤖).

Why this exists: the original HF corpus (src/data/positions.py) is ~30k bare FENs of unknown
provenance and no move history. For an SAE we want (a) many more, independent, realistic positions
and (b) each position's REAL history, so the history-robustness check (src/eval/history_filter.py)
can compare "current board repeated" (the canonical encoding everything trains on) against what
Leela would see in an actual game.

Source: the official `Lichess/standard-chess-games` parquet dump on the HF Hub (one month, a few
shards). Design choices — each one guards a specific way the data could mislead the SAE / eval:
  * QUALITY FILTER — both players >= min_elo, base time >= min_base_seconds (no bullet), no BOT
    accounts. Keeps positions closer to strong play (Leela's own distribution) than beginner chaos.
  * FEW POSITIONS PER GAME — `positions_per_game` random plies per game, so one long game can't
    dominate and consecutive (near-identical) plies don't flood the training set.
  * NO TRIVIAL OPENINGS / TERMINALS — plies < min_ply are skipped (book positions repeat across
    thousands of games); positions with no legal move are dropped (Leela never evaluates them).
  * SPLIT BY GAME, THEN DEDUP BY POSITION — train/val/test are assigned per game (hash of the game
    URL), positions are de-duplicated on a counter-free key (board + side + castling + ep), and any
    val/test position whose key also occurs in train is removed. So held-out numbers are held out.
"""

from __future__ import annotations

import hashlib
import io
import random
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Sequence

import chess
import chess.pgn
import pandas as pd

HF_REPO = "Lichess/standard-chess-games"
_META_COLS = ["Site", "WhiteElo", "BlackElo", "WhiteTitle", "BlackTitle", "TimeControl", "movetext"]


def shard_paths(year: int, month: int, n_shards: int, offset: int = 0) -> list[str]:
    """Download (or hit the HF cache for) `n_shards` parquet shards of one month. Returns local paths."""
    from huggingface_hub import HfApi, hf_hub_download

    prefix = f"data/year={year}/month={month:02d}/"
    names = sorted(
        s.rfilename for s in HfApi().dataset_info(HF_REPO).siblings if s.rfilename.startswith(prefix)
    )
    if len(names) < offset + n_shards:
        raise ValueError(f"{prefix} has only {len(names)} shards")
    return [hf_hub_download(HF_REPO, n, repo_type="dataset") for n in names[offset : offset + n_shards]]


def position_key(board: chess.Board) -> str:
    """Counter-free identity of a position: placement, side to move, castling, LEGAL ep square.

    The FEN's halfmove/fullmove counters are dropped, so the same position reached at move 12 and
    move 30 counts as one position (string-FEN dedup misses this). python-chess's default
    `en_passant="legal"` only writes the ep square when an ep capture is actually legal.
    """
    return " ".join(board.fen().split(" ")[:4])


def split_of(game_id: str, val_frac: float, test_frac: float) -> str:
    """Deterministic per-GAME split from a hash of the game URL (stable across runs and machines)."""
    u = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"


def _filter_games(df: pd.DataFrame, min_elo: int, min_base_seconds: int, exclude_bots: bool) -> pd.DataFrame:
    base = pd.to_numeric(df["TimeControl"].str.split("+").str[0], errors="coerce")
    keep = (df["WhiteElo"] >= min_elo) & (df["BlackElo"] >= min_elo) & (base >= min_base_seconds)
    if exclude_bots:
        keep &= (df["WhiteTitle"] != "BOT") & (df["BlackTitle"] != "BOT")
    return df[keep]


def _sample_game(args) -> list[dict]:
    """Parse one game's movetext and return up to `per_game` sampled positions (worker function)."""
    site, movetext, white_elo, black_elo, per_game, min_ply, seed = args
    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None:
        return []
    moves = list(game.mainline_moves())
    candidates = list(range(min_ply, len(moves) + 1))      # ply p == position AFTER p half-moves
    if not candidates:
        return []
    rng = random.Random(f"{seed}:{site}")                   # per-game RNG: reproducible, order-free
    plies = sorted(rng.sample(candidates, min(per_game, len(candidates))))

    out = []
    board = chess.Board()
    done = 0
    for p in plies:
        while done < p:                                     # advance the board to ply p
            board.push(moves[done])
            done += 1
        if not any(board.generate_legal_moves()):           # mate/stalemate: no policy to read
            continue
        out.append(
            {
                "game_id": site,
                "ply": p,
                "fen": board.fen(),
                "key": position_key(board),
                # Full UCI history from the standard start position — lets history_filter rebuild
                # the board WITH its move stack (real history planes) on demand.
                "uci_history": " ".join(m.uci() for m in moves[:p]),
                "white_elo": int(white_elo),
                "black_elo": int(black_elo),
                "n_pieces": chess.popcount(board.occupied),
            }
        )
    return out


def build_corpus(
    paths: Sequence[str],
    min_elo: int = 2000,
    min_base_seconds: int = 180,
    exclude_bots: bool = True,
    positions_per_game: int = 2,
    min_ply: int = 6,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 0,
    max_games: int | None = None,
    processes: int = 8,
) -> pd.DataFrame:
    """Sample a de-duplicated, game-split position corpus from Lichess parquet shards.

    Returns a DataFrame with one row per position and a `split` column in {train, val, test}.
    """
    frames = []
    for p in paths:
        df = pd.read_parquet(p, columns=_META_COLS)
        frames.append(_filter_games(df, min_elo, min_base_seconds, exclude_bots))
    games = pd.concat(frames, ignore_index=True).drop_duplicates("Site")
    if max_games is not None:
        games = games.sample(n=min(max_games, len(games)), random_state=seed)

    jobs: Iterable = (
        (r.Site, r.movetext, r.WhiteElo, r.BlackElo, positions_per_game, min_ply, seed)
        for r in games.itertuples(index=False)
    )
    if processes <= 1:
        rows = [row for job in jobs for row in _sample_game(job)]
    else:
        # forkserver, not fork: torch/pyarrow threads may already be running in this process.
        with get_context("forkserver").Pool(processes) as pool:
            rows = [row for chunk in pool.imap_unordered(_sample_game, jobs, chunksize=256) for row in chunk]
    corpus = pd.DataFrame(rows)
    corpus["split"] = [split_of(g, val_frac, test_frac) for g in corpus["game_id"]]

    # Dedup on the counter-free key. Sort first so "which duplicate survives" is deterministic and
    # prefers train (alphabetically test < train < val, so sort by a train-first rank instead).
    rank = corpus["split"].map({"train": 0, "val": 1, "test": 2})
    corpus = corpus.assign(_rank=rank).sort_values(["_rank", "game_id", "ply"])
    corpus = corpus.drop_duplicates("key", keep="first")     # also removes held-out keys seen in train
    corpus = corpus.drop(columns="_rank")
    # Shuffle row order (reproducibly) so any prefix is a random sample, not one game-hash range.
    return corpus.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_corpus(path: str | Path, split: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load a corpus parquet written by build_corpus (optionally one split, first `limit` rows)."""
    df = pd.read_parquet(path)
    if split is not None:
        df = df[df["split"] == split]
    if limit is not None:
        df = df.iloc[:limit]
    return df.reset_index(drop=True)


def board_with_history(uci_history: str) -> chess.Board:
    """Rebuild the position by replaying its moves from the start, so the board carries a REAL move
    stack. Pass the result to an lczerolens encoder to get genuine history planes."""
    from lczerolens import LczeroBoard

    board = LczeroBoard()
    for uci in uci_history.split():
        board.push_uci(uci)
    return board

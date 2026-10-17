"""Build the Lichess position corpus (src/data/lichess.py): ~316k positions from June 2024.

    uv run python scripts/00_build_corpus.py

Downloads 10 parquet shards (~1.7 GB) of Lichess/standard-chess-games into the HF cache, keeps games
where both players are >= 2000 at >= 3+0 with no bots, samples 2 positions per game, splits by game
and de-duplicates positions. Writes outputs/positions/lichess_2024-06.parquet.
"""

from pathlib import Path

from src.data import lichess

OUT = Path("outputs/positions/lichess_2024-06.parquet")


def main():
    paths = lichess.shard_paths(2024, 6, n_shards=10)
    corpus = lichess.build_corpus(paths, processes=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(OUT)
    print(corpus["split"].value_counts())
    print(corpus[["n_pieces", "ply"]].describe())


if __name__ == "__main__":
    main()

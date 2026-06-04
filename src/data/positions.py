"""Position distribution (you pick) + Ray workers for parallel generation (agent scaffolds ⚙️).

The "distribution you pick" (DESIGN.md §3, step 3) is a fixed corpus of real chess positions:
the `bonna46/Chess-FEN-and-NL-Format-30K-Dataset` on the HuggingFace Hub. We use only its `FEN`
column — ~30k unique, already-valid FENs spanning real midgame/endgame positions. That breadth is
the whole point for SAE training: the residual-stream features we want to surface only light up
across a varied board distribution, not the handful of openings you'd get from self-play seeds.

Why a static corpus instead of Ray-parallel self-play: the expensive, parallelizable step in this
project is the *activation capture* (running every FEN through the Leela net), and that already
lives behind `capture_fn` in src/data/cache.py. Reading 30k strings out of one cached CSV is I/O
that finishes in well under a second, so a Ray fan-out here would be pure ceremony. If you later
swap to a generated distribution (Lichess puzzles, X% random, self-play), this is the file that
grows the Ray workers — the cacher downstream won't change.
"""

from __future__ import annotations

import pandas as pd
from huggingface_hub import hf_hub_download

# The corpus. hf_hub_download caches the file under ~/.cache/huggingface after the first pull, so
# repeated calls are offline + instant. Pinning these as module constants keeps the "which
# distribution did we train on?" answer in one obvious place (it feeds reproducibility later).
HF_REPO = "bonna46/Chess-FEN-and-NL-Format-30K-Dataset"
HF_FILE = "Chess_FEN+NL_format.csv"
FEN_COLUMN = "FEN"  # the CSV also has Next move / NL-description columns; we want only this one


def load_fens(
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    dedup: bool = True,
    validate: bool = False,
) -> list[str]:
    """Load chess FENs from the HF corpus, returning a plain list[str] for the cacher.

    The output is exactly the `Sequence[str]` that src/data/cache.cache_activations expects, so the
    data path is: load_fens() -> cache_activations(fens, capture_fn, ...) -> load_activations().

    Args:
        limit:    keep at most this many FENs (after optional shuffle). None = all ~30k.
        shuffle:  shuffle before applying `limit`. Off by default so runs are deterministic and a
                  `limit` always returns the same prefix; turn on (with `seed`) for a random subset.
        seed:     RNG seed for the shuffle, so a "random 5k" subset is reproducible.
        dedup:    drop duplicate FEN strings. The corpus is already unique today, but this guards
                  against training the SAE on the same residual vector twice if that ever changes.
        validate: parse every FEN with python-chess and raise on the first malformed one. Off by
                  default (it's an extra pass over all rows); flip on when ingesting a new corpus.

    Returns:
        list[str] of FENs in the order they'll be cached.
    """
    # Pull (or hit cache for) the single CSV, then read ONLY the FEN column — usecols means pandas
    # never materializes the Next move / NL-description text we don't use.
    csv_path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    col = pd.read_csv(csv_path, usecols=[FEN_COLUMN])[FEN_COLUMN]

    col = col.dropna()                       # never hand a NaN FEN to LczeroBoard downstream
    if dedup:
        col = col.drop_duplicates()
    if shuffle:
        # frac=1 = return all rows in random order; fixed seed keeps the subset reproducible.
        col = col.sample(frac=1, random_state=seed)
    if limit is not None:
        col = col.iloc[:limit]               # prefix AFTER shuffle, so it's a random sample when shuffled

    fens = col.tolist()

    if validate:
        # Fail loud and early: a malformed FEN here would otherwise surface as a confusing crash
        # deep inside the net's input-plane encoding. Import locally so the common path stays light.
        import chess

        for fen in fens:
            try:
                chess.Board(fen)
            except ValueError as e:
                raise ValueError(f"invalid FEN from {HF_REPO}: {fen!r}") from e

    return fens

"""The ORIGINAL training corpus: ~30k FENs from the HF dataset bonna46/Chess-FEN-and-NL-Format-30K-Dataset.

Superseded by src/data/lichess.py and kept only so scripts/05_data_comparison.py can reproduce the
comparison. The 2026-10 audit found it off-distribution for this net: no endgames (median 30 pieces),
41% of positions with a >= 3-pawn material imbalance, two-thirds of its "best moves" are captures,
Leela rates 68% of positions as decisive — it looks like random play, not real games — and it has no
move histories. See RESULTS.md.
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

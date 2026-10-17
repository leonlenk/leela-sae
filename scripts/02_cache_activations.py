"""Cache residual-stream activations for the Lichess TRAIN split, several layers from one forward pass.

    uv run python scripts/02_cache_activations.py --layers 3 7 11 --n-boards 200000

Writes outputs/acts/L{layer}_train.npy as (N, 64, 768) float16 (plus a .meta.json fingerprint).
Canonical history-free encoding (hook.encode_fens) — the same input every other script uses.

--real-history instead feeds each position with its REAL game history (hook.encode_boards_with_history)
and writes L{layer}_train_realhist.npy. This is ONLY for the history-filter positive control; the
main SAEs never see history (CLAUDE.md).
"""

import argparse
from pathlib import Path

import lczerolens
import numpy as np

from src import hook
from src.data import cache, lichess

CORPUS = Path("outputs/positions/lichess_2024-06.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[7])
    ap.add_argument("--n-boards", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-dir", default="outputs/acts")
    ap.add_argument("--real-history", action="store_true")
    args = ap.parse_args()

    model = hook.load_model()
    modules = hook.list_residual_modules(model)
    names = {L: modules[L] for L in args.layers}
    corpus = lichess.load_corpus(CORPUS, split="train", limit=args.n_boards)
    fens = corpus["fen"].tolist()
    history = dict(zip(corpus["fen"], corpus["uci_history"]))

    def capture(batch):
        if args.real_history:
            x = hook.encode_boards_with_history([lichess.board_with_history(history[f]) for f in batch])
        else:
            x = hook.encode_fens(batch)
        acts, _ = hook.run_with_cache(model, x, list(names.values()))
        return {L: acts[n] for L, n in names.items()}

    mode = "realhist" if args.real_history else "fen"
    suffix = "_realhist" if args.real_history else ""
    sig = lambda L: f"{hook.DEFAULT_CHECKPOINT}|{names[L]}|{hook.ENCODING.name}|{mode}|lczerolens=={lczerolens.__version__}"
    paths = cache.cache_activations_multi(
        fens, capture, {L: Path(args.out_dir, f"L{L}_train{suffix}.npy") for L in args.layers},
        d_model=768, batch_size=args.batch_size, dtype=np.float16,
        signatures={L: sig(L) for L in args.layers},
    )
    for L, p in paths.items():
        print(L, p)


if __name__ == "__main__":
    main()

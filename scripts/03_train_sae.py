"""Train a Top-K SAE on cached activations for one layer (scripts/02_cache_activations.py first).

    uv run python scripts/03_train_sae.py --layer 7
    uv run python scripts/03_train_sae.py --layer 7 --k 64 --tag k64

Writes outputs/sae/L{layer}{_tag}.pt (config + weights + training history) and a .json summary.
Validation FVU is tracked on the Lichess VAL split (held-out games), captured on the fly.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src import hook, train
from src.data import cache, lichess
from src.sae import TopKSAE

CORPUS = Path("outputs/positions/lichess_2024-06.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n-boards", type=int, default=None, help="train on the first N cached boards")
    ap.add_argument("--acts", default=None, help="override the activation cache path")
    ap.add_argument("--tag", default="")
    ap.add_argument("--val-boards", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None, help="default: Gao et al. scaling rule")
    ap.add_argument("--real-history", action="store_true", help="val acts with real game history (pair with a _realhist cache)")
    ap.add_argument("--random-weights", type=int, default=None, metavar="SEED",
                    help="val acts from the randomized net (pair with a _randnet cache)")
    ap.add_argument("--random-init", default="permute", choices=["permute", "fresh"])
    args = ap.parse_args()

    acts_path = Path(args.acts or f"outputs/acts/L{args.layer}_train.npy")
    grouped = cache.load_activations(acts_path, flatten=False, in_memory=True)       # (N, 64, D) fp16
    if args.n_boards:
        grouped = grouped[: args.n_boards]
    acts = grouped.reshape(-1, grouped.shape[-1])                                     # (N*64, D)
    print(f"training rows: {acts.shape[0]:,} from {grouped.shape[0]:,} boards ({acts_path})")

    model = hook.load_model()
    if args.random_weights is not None:
        hook.randomize_weights(model, seed=args.random_weights, mode=args.random_init)
    module = hook.list_residual_modules(model)[args.layer]
    val = lichess.load_corpus(CORPUS, split="val", limit=args.val_boards)
    if args.real_history:
        val_x = hook.encode_boards_with_history([lichess.board_with_history(h) for h in val["uci_history"]])
    else:
        val_x = hook.encode_fens(val["fen"].tolist())
    val_acts = torch.cat([hook.run_with_cache(model, val_x[s:s + 250], [module])[0][module].cpu()
                          for s in range(0, len(val_x), 250)]).reshape(-1, grouped.shape[-1])
    del model
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    d = acts.shape[-1]
    sae = TopKSAE(d, args.expansion * d, args.k)
    sae.init_pre_bias(acts[torch.randperm(acts.shape[0])[:200_000]])
    config = {"epochs": args.epochs, "seed": args.seed, "lr": args.lr}
    t0 = time.time()
    sae, history = train.train(sae, acts, config, val_acts=val_acts)
    minutes = (time.time() - t0) / 60

    name = f"L{args.layer}" + (f"_{args.tag}" if args.tag else "")
    out = Path("outputs/sae", f"{name}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"layer": args.layer, "module": module, "acts": str(acts_path), "train_boards": int(grouped.shape[0]),
            "real_history": args.real_history, "random_weights": args.random_weights,
            "random_init": args.random_init if args.random_weights is not None else None,
            "train_config": {**train.DEFAULTS, **config}, "minutes": minutes,
            "final": history[-1], "val_fvu": train.fvu(sae, val_acts)}
    sae.save(out, meta=meta, history=history)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

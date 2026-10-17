"""Does the original HF corpus (bonna46 30k FENs) help? Same-budget comparison against Lichess.

    uv run python scripts/05_data_comparison.py

Trains two layer-7 SAEs with IDENTICAL code, config and step count — one on 18k HF boards (the
original training set, outputs/data.npy), one on the first 18k Lichess train boards — and scores both
on held-out positions from EACH distribution (Lichess test games; the HF 10% hold-out). Writes
outputs/results/data_comparison.json.
"""

import json
from pathlib import Path

import torch

from src import hook, train
from src.data import cache, lichess, positions
from src.eval.fidelity import fidelity, mean_activation
from src.sae import TopKSAE

N_BOARDS = 18_000
LAYER = 7


def main():
    model = hook.load_model()
    module = hook.list_residual_modules(model)[LAYER]

    hf_fens = positions.load_fens(limit=20_000, shuffle=True, seed=67)       # the original split
    hf_train_acts = cache.load_activations("outputs/data.npy", flatten=False, in_memory=True)[:N_BOARDS]
    hf_eval = hf_fens[18_000:][:500]
    li_train_acts = cache.load_activations(f"outputs/acts/L{LAYER}_train.npy", flatten=False)[:N_BOARDS].float()
    li_eval = lichess.load_corpus("outputs/positions/lichess_2024-06.parquet", split="test", limit=500)["fen"].tolist()
    means = {"lichess": mean_activation(li_train_acts), "hf": mean_activation(hf_train_acts)}

    results = {}
    for name, acts in (("hf", hf_train_acts), ("lichess", li_train_acts)):
        torch.manual_seed(0)
        flat = acts.reshape(-1, acts.shape[-1]).half()
        sae = TopKSAE(768, 8 * 768, 32)
        sae.init_pre_bias(flat[torch.randperm(flat.shape[0])[:200_000]])
        # Same number of optimiser steps for both (the HF set is only 1.15M tokens): 4 epochs.
        sae, _ = train.train(sae, flat, {"epochs": 4, "dead_tokens": 1_000_000})
        results[f"trained_on_{name}"] = {
            f"eval_{ev}": fidelity(model, sae, module, fens, means[ev])
            for ev, fens in (("lichess", li_eval), ("hf", hf_eval))
        }
        print(name, json.dumps(results[f"trained_on_{name}"], indent=1))

    out = Path("outputs/results/data_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

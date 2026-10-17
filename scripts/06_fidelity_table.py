"""One fidelity table for every trained SAE, on the SAME 1000 held-out Lichess test boards.

    uv run python scripts/06_fidelity_table.py outputs/sae/L3.pt outputs/sae/L7.pt ...

The mean-ablation baseline for layer L is always the Lichess training mean at that layer, so SAEs
trained on other data (the original HF baseline) are judged against the same reference.
Writes outputs/results/fidelity_table.{json,md}.
"""

import json
import sys
from pathlib import Path

import torch

from src import hook
from src.data import cache, lichess
from src.eval.fidelity import fidelity, mean_activation
from src.sae import TopKSAE

N_TEST = 1000


def main():
    paths = sys.argv[1:]
    model = hook.load_model()
    fens = lichess.load_corpus("outputs/positions/lichess_2024-06.parquet", split="test", limit=N_TEST)["fen"].tolist()
    rows = {}
    for p in paths:
        meta = torch.load(p, map_location="cpu", weights_only=True)["meta"]
        layer = meta["layer"]
        mean_act = mean_activation(cache.load_activations(f"outputs/acts/L{layer}_train.npy", flatten=False))
        rows[Path(p).stem] = {"layer": layer, **fidelity(model, TopKSAE.load(p), meta["module"], fens, mean_act)}
        print(Path(p).stem, json.dumps(rows[Path(p).stem], indent=1), flush=True)

    out = Path("outputs/results/fidelity_table.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    cols = [("fvu", "FVU"), ("kl_recon_mean", "KL legal"), ("kl_recon_median", "KL median"),
            ("top1_agree_recon", "top-1 agree"), ("abs_dq_recon_mean", "|dQ|"), ("kl_noise_mean", "KL noise"),
            ("error_vs_noise_kl_ratio", "err/noise"), ("kl_mean_ablation_mean", "KL mean-abl"),
            ("loss_recovered_vs_mean", "recovered vs mean")]
    lines = ["| SAE | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[k]:.4f}" for k, _ in cols) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Is a property's information in the SAE, just spread over several features?

    uv run python scripts/11_feature_splitting.py --sae outputs/sae/L7.pt

For each derived per-square property: rank features by F1 on S, then fit a logistic probe on the
activations of only the top-n features (n = 1, 3, 10, 30) and score on T. Compare with the probe on the
full residual stream (outputs/results/probes_L{layer}.json). If a handful of features recover most of
the probe's F1, the concept is in the dictionary but split; if not, the SAE is missing it.
Writes outputs/results/<sae>/feature_splitting.json.
"""

import argparse
import json
from pathlib import Path

import torch

from src import hook
from src.data import lichess
from src.eval import board_props, correlational, probes
from src.eval.codes import encode_corpus
from src.sae import TopKSAE

NS = (1, 3, 10, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()
    meta = torch.load(args.sae, map_location="cpu", weights_only=True)["meta"]
    sae = TopKSAE.load(args.sae)
    model = hook.load_model()
    S = lichess.load_corpus("outputs/positions/lichess_2024-06.parquet", split="val", limit=args.n)["fen"].tolist()
    T = lichess.load_corpus("outputs/positions/lichess_2024-06.parquet", split="test", limit=args.n)["fen"].tolist()
    cS, cT = encode_corpus(model, sae, meta["module"], S), encode_corpus(model, sae, meta["module"], T)
    sqS, _ = board_props.label_arrays(S)
    sqT, _ = board_props.label_arrays(T)
    full = json.loads(Path(f"outputs/results/probes_L{meta['layer']}.json").read_text())

    out = {}
    for fam, origin in board_props.SQUARE_FAMILIES.items():
        if origin != "derived" or sqS[fam].sum() < 100:
            continue
        yS, yT = torch.from_numpy(sqS[fam]), torch.from_numpy(sqT[fam])
        f1_S = correlational.token_f1_all(cS, yS)[2]
        ranked = torch.argsort(f1_S, descending=True)[: max(NS)].tolist()
        XS = torch.stack([cS.feature(f).reshape(-1) for f in ranked], 1)       # (M, 30) dense
        XT = torch.stack([cT.feature(f).reshape(-1) for f in ranked], 1)
        row = {"probe_full_residual_f1": full[fam]["probe_layer"]["test_f1"]}
        for n in NS:
            row[f"top{n}_features_f1"] = probes.fit_probe(XS[:, :n], yS.reshape(-1), XT[:, :n], yT.reshape(-1))["test_f1"]
        out[fam] = row
        print(f"{fam:20s} " + "  ".join(f"{k.split('_')[0]} {v:.2f}" for k, v in row.items()), flush=True)
    Path("outputs/results", Path(args.sae).stem, "feature_splitting.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

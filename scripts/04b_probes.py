"""Linear-probe baselines per layer (independent of any SAE), on the same S / T splits as 04_evaluate.

    uv run python scripts/04b_probes.py --layers 3 7 11

For every property: a logistic probe on the layer's residual stream, and one on the raw input planes
(per token for square properties; mean-pooled residual / all 64x112 planes for board properties).
Fit + threshold on S (Lichess val), scored on T (Lichess test). Writes outputs/results/probes_L{L}.json.
"""

import argparse
import json
from pathlib import Path

import torch

from src import hook
from src.data import lichess
from src.eval import board_props, probes

CORPUS = "outputs/positions/lichess_2024-06.parquet"


def acts_and_inputs(model, module, fens, batch=250):
    acts, planes = [], []
    for s in range(0, len(fens), batch):
        x = hook.encode_fens(fens[s:s + batch])                              # (b, 112, 8, 8)
        a, _ = hook.run_with_cache(model, x, [module])
        acts.append(a[module].cpu())                                         # (b, 64, D)
        planes.append(x.permute(0, 2, 3, 1).reshape(-1, 64, 112))            # token = row*8 + col
    return torch.cat(acts), torch.cat(planes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 7, 11])
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()
    model = hook.load_model()
    S = lichess.load_corpus(CORPUS, split="val", limit=args.n)["fen"].tolist()
    T = lichess.load_corpus(CORPUS, split="test", limit=args.n)["fen"].tolist()
    sqS, bdS = board_props.label_arrays(S)
    sqT, bdT = board_props.label_arrays(T)
    for L in args.layers:
        module = hook.list_residual_modules(model)[L]
        aS, pS = acts_and_inputs(model, module, S)
        aT, pT = acts_and_inputs(model, module, T)
        out = {}
        for fam in board_props.SQUARE_FAMILIES:
            yS, yT = torch.from_numpy(sqS[fam]).reshape(-1), torch.from_numpy(sqT[fam]).reshape(-1)
            if yS.sum() < 20:
                continue
            out[fam] = {"probe_layer": probes.fit_probe(aS.reshape(-1, aS.shape[-1]), yS, aT.reshape(-1, aT.shape[-1]), yT),
                        "probe_input": probes.fit_probe(pS.reshape(-1, 112), yS, pT.reshape(-1, 112), yT)}
        for fam in board_props.BOARD_FAMILIES:
            yS, yT = torch.from_numpy(bdS[fam]), torch.from_numpy(bdT[fam])
            if yS.sum() < 20:
                continue
            out[fam] = {"probe_layer": probes.fit_probe(aS.mean(1), yS, aT.mean(1), yT),
                        "probe_input": probes.fit_probe(pS.reshape(len(S), -1), yS, pT.reshape(len(T), -1), yT)}
        for fam, r in out.items():
            print(f"L{L} {fam:28s} layer {r['probe_layer']['test_f1']:.3f}  input {r['probe_input']['test_f1']:.3f}", flush=True)
        Path(f"outputs/results/probes_L{L}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

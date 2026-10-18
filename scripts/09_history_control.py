"""Positive control for the history filter, using an SAE trained WITH real game history.

    uv run python scripts/02_cache_activations.py --layers 7 --real-history
    uv run python scripts/03_train_sae.py --layer 7 --acts outputs/acts/L7_train_realhist.npy --real-history --tag realhist
    uv run python scripts/09_history_control.py

The main SAEs never see history, so they cannot contain history features and the filter has nothing
to catch on them. Here:
  1. A linear probe for "destination square of the last move" on layer-7 activations with REAL vs
     canonical (REP) history: does the residual stream carry move history at all?
  2. The real-history SAE's best feature for that property, and whether the filter REJECTS it (it
     must, if the filter has power).
  3. Fraction of features surviving the filter, real-history SAE vs the canonical SAE.
Writes outputs/results/history_control.json.
"""

import json
from pathlib import Path

import torch

from src import hook
from src.data import lichess
from src.eval import correlational, history_filter, probes
from src.eval.codes import encode_corpus
from src.sae import TopKSAE

N = 3000


def main():
    model = hook.load_model()
    module = hook.list_residual_modules(model)[7]
    tst = lichess.load_corpus("outputs/positions/lichess_2024-06.parquet", split="test")
    tst = tst[tst["ply"] >= 8].iloc[:N].reset_index(drop=True)
    rep_x = hook.encode_fens(tst["fen"].tolist())
    real_x = hook.encode_boards_with_history([lichess.board_with_history(h) for h in tst["uci_history"]])
    last_to = torch.from_numpy(history_filter.last_move_to_labels(tst["uci_history"].tolist()))   # (N, 64)
    half = N // 2
    out = {"n_boards": N}

    # 1. Probe: is the last move linearly decodable at layer 7?
    acts = {}
    for tag, x in (("rep", rep_x), ("real", real_x)):
        acts[tag] = torch.cat([hook.run_with_cache(model, x[s:s + 250], [module])[0][module].cpu()
                               for s in range(0, N, 250)]).reshape(-1, 768)
    y = last_to.reshape(-1)
    M = half * 64
    out["probe_last_move_to"] = {tag: probes.fit_probe(a[:M], y[:M], a[M:], y[M:]) for tag, a in acts.items()}
    print("probe:", json.dumps(out["probe_last_move_to"], indent=1), flush=True)

    # 2-3. Filter on each SAE.
    for name in ("L7", "L7_realhist"):
        sae = TopKSAE.load(f"outputs/sae/{name}.pt")
        codes_rep = encode_corpus(model, sae, module, rep_x)
        codes_real = encode_corpus(model, sae, module, real_x)
        agree = history_filter.activation_agreement(codes_rep, codes_real)
        surv = history_filter.survivors(agree)
        # Select the last-move feature on the first half of boards under REAL history, test on the rest.
        lm = correlational.evaluate_property(codes_real.subset(range(half)), last_to[:half],
                                             codes_real.subset(range(half, N)), last_to[half:])
        f = lm["feature"]
        # The same feature's held-out F1 when the net is fed canonical history instead.
        fires_rep = codes_rep.subset(range(half, N)).feature(f) > 0
        tp = int((fires_rep & last_to[half:]).sum())
        _, _, f1_rep = correlational.f1_from_counts(torch.tensor([tp]), torch.tensor([int(fires_rep.sum())]), last_to[half:].sum())
        out[name] = {
            "measurable_features": int(surv["measurable"].sum()),
            "survive_frac": float(surv["survives"].sum() / surv["measurable"].sum()),
            "last_move_feature": lm,
            "last_move_feature_f1_with_rep_history": float(f1_rep),
            "last_move_feature_corr": float(agree["corr"][f]),
            "last_move_feature_survives": bool(surv["survives"][f]),
            "corr": agree["corr"][surv["measurable"]].tolist(),
        }
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "corr"}, indent=1), flush=True)

    Path("outputs/results/history_control.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()

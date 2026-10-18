"""Evaluate one trained SAE end to end: fidelity, correlational, causal, history robustness.

    uv run python scripts/04_evaluate.py --sae outputs/sae/L7.pt

Splits (all from held-out GAMES, never seen in SAE training):
  S = first --n-select boards of the Lichess VAL split   -> feature SELECTION, probe fitting
  T = first --n-test boards of the Lichess TEST split    -> every reported number
Writes outputs/results/<sae name>/results.json and summary.md.

DESIGN.md §4: a feature is only "real" if it (1) matches a board property on held-out data, (2) has
a causal effect on the heads beyond an activation-matched control, and (3) survives the history
filter. The summary table reports all three per property.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import hook
from src.data import cache, lichess
from src.eval import board_props, causal, correlational, history_filter, probes
from src.eval.codes import encode_corpus
from src.eval.feature_mining import density_from_codes
from src.eval.fidelity import fidelity, mean_activation
from src.sae import TopKSAE

CORPUS = Path("outputs/positions/lichess_2024-06.parquet")
MIN_POS_SQUARE = 100     # minimum positives in S to score a per-square property
MIN_POS_BOARD = 50


def acts_and_inputs(model, module, fens, batch=250):
    """(N*64, D) residual acts and (N*64, 112) raw input planes per token, in token order."""
    acts, planes = [], []
    for s in range(0, len(fens), batch):
        x = hook.encode_fens(fens[s:s + batch])                              # (b, 112, 8, 8)
        a, _ = hook.run_with_cache(model, x, [module])
        acts.append(a[module].reshape(-1, a[module].shape[-1]).cpu())
        planes.append(x.permute(0, 2, 3, 1).reshape(-1, 112))                # token = row*8 + col
    return torch.cat(acts), torch.cat(planes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--n-select", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--n-fidelity", type=int, default=1000)
    ap.add_argument("--n-probe", type=int, default=1500)
    ap.add_argument("--n-causal", type=int, default=64)
    ap.add_argument("--n-random-features", type=int, default=30)
    ap.add_argument("--n-history", type=int, default=1500)
    ap.add_argument("--skip-probes", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.sae, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    sae = TopKSAE.load(args.sae)
    name = Path(args.sae).stem
    out_dir = Path("outputs/results", name)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = hook.load_model()
    module = meta["module"]
    res = {"sae": args.sae, "module": module, "config": sae.config}

    sel = lichess.load_corpus(CORPUS, split="val", limit=args.n_select)
    tst = lichess.load_corpus(CORPUS, split="test", limit=args.n_test)
    fens_S, fens_T = sel["fen"].tolist(), tst["fen"].tolist()

    # ---------------------------------------------------------------- 1. fidelity
    grouped = cache.load_activations(meta["acts"], flatten=False)
    mean_act = mean_activation(grouped)                                       # (64, D)
    res["fidelity"] = fidelity(model, sae, module, fens_T[: args.n_fidelity], mean_act)
    print("fidelity:", json.dumps(res["fidelity"], indent=1))

    # ---------------------------------------------------------------- 2. codes + density
    codes_S = encode_corpus(model, sae, module, fens_S)
    codes_T = encode_corpus(model, sae, module, fens_T)
    dens = density_from_codes(codes_T)
    res["density"] = {
        "tokens": int(codes_T.idx.shape[0] * 64),
        "dead_frac": float((dens == 0).float().mean()),
        "dense_frac_gt_10pct": float((dens > 0.1).float().mean()),
        "median_live_density": float(dens[dens > 0].median()),
        "hist_log10": np.histogram(np.log10(dens[dens > 0].numpy()), bins=40)[0].tolist(),
        "hist_edges": np.histogram(np.log10(dens[dens > 0].numpy()), bins=40)[1].tolist(),
    }
    print("density:", {k: v for k, v in res["density"].items() if not k.startswith("hist")})

    # ---------------------------------------------------------------- 3. correlational (+ probes)
    sq_S, bd_S = board_props.label_arrays(fens_S)
    sq_T, bd_T = board_props.label_arrays(fens_T)
    rows = []
    for fam, origin in board_props.SQUARE_FAMILIES.items():
        if sq_S[fam].sum() < MIN_POS_SQUARE:
            continue
        r = correlational.evaluate_property(codes_S, torch.from_numpy(sq_S[fam]), codes_T, torch.from_numpy(sq_T[fam]))
        rows.append({"property": fam, "level": "square", "origin": origin, **r})
    for fam, origin in board_props.BOARD_FAMILIES.items():
        if bd_S[fam].sum() < MIN_POS_BOARD:
            continue
        r = correlational.evaluate_property(codes_S, torch.from_numpy(bd_S[fam]), codes_T, torch.from_numpy(bd_T[fam]), level="board")
        rows.append({"property": fam, "level": "board", "origin": origin, **r})

    if not args.skip_probes:
        n = args.n_probe
        aS, pS = acts_and_inputs(model, module, fens_S[:n])
        aT, pT = acts_and_inputs(model, module, fens_T[:n])
        for row in rows:
            fam = row["property"]
            if row["level"] == "square":
                yS, yT = torch.from_numpy(sq_S[fam][:n]).reshape(-1), torch.from_numpy(sq_T[fam][:n]).reshape(-1)
                row["probe_layer"] = probes.fit_probe(aS, yS, aT, yT)
                row["probe_input"] = probes.fit_probe(pS, yS, pT, yT)
            else:
                yS, yT = torch.from_numpy(bd_S[fam][:n]), torch.from_numpy(bd_T[fam][:n])
                row["probe_layer"] = probes.fit_probe(aS.reshape(n, 64, -1).mean(1), yS, aT.reshape(n, 64, -1).mean(1), yT)
                row["probe_input"] = probes.fit_probe(pS.reshape(n, -1), yS, pT.reshape(n, -1), yT)
            print(f"{fam:28s} sae F1 {row['test_f1']:.3f}  probe(layer) {row['probe_layer']['test_f1']:.3f}  "
                  f"probe(input) {row['probe_input']['test_f1']:.3f}")
        del aS, aT, pS, pT

    # ---------------------------------------------------------------- 4. causal
    for row in rows:
        c = causal.feature_vs_control(model, sae, module, codes_T, fens_T, row["feature"], n_boards=args.n_causal)
        row["causal"] = {k: v for k, v in c.items() if k != "raw"}
        print(f"causal {row['property']:28s} feat {row['feature']:5d}  KL {c.get('kl_median', 0):.4f} vs "
              f"{c.get('control_kl_median', 0):.4f}  win {c.get('kl_win_rate', float('nan')):.2f}  "
              f"target win {c.get('target_win_rate', float('nan')):.2f}")
    # Calibration: the same test on random live features (no property attached).
    rng = torch.Generator().manual_seed(0)
    live = torch.nonzero((dens > 1e-4) & (dens < 0.2)).flatten()
    rand_feats = live[torch.randperm(len(live), generator=rng)[: args.n_random_features]].tolist()
    res["causal_random_features"] = [
        {k: v for k, v in causal.feature_vs_control(model, sae, module, codes_T, fens_T, f, n_boards=args.n_causal).items() if k != "raw"}
        for f in rand_feats
    ]

    # ---------------------------------------------------------------- 5. history filter
    hist_rows = [i for i in range(len(tst)) if tst["ply"][i] >= 8][: args.n_history]
    boards = [lichess.board_with_history(tst["uci_history"][i]) for i in hist_rows]
    real_inputs = hook.encode_boards_with_history(boards)
    codes_rep = codes_T.subset(hist_rows)
    codes_real = encode_corpus(model, sae, module, real_inputs)
    agree = history_filter.activation_agreement(codes_rep, codes_real)
    surv = history_filter.survivors(agree)
    # Whole-net effect of real vs repeated history, and SAE reconstruction under both.
    rep_inputs = hook.encode_fens([fens_T[i] for i in hist_rows])
    kls, fvu_parts = [], {"rep": [0.0, 0.0], "real": [0.0, 0.0]}
    for s in range(0, len(hist_rows), 250):
        a_rep, o_rep = hook.run_with_cache(model, rep_inputs[s:s + 250], [module])
        a_real, o_real = hook.run_with_cache(model, real_inputs[s:s + 250], [module])
        for tag, a in (("rep", a_rep[module]), ("real", a_real[module])):
            flat = a.reshape(-1, a.shape[-1])
            recon = sae.to(flat.device)(flat)[0]
            fvu_parts[tag][0] += float((recon - flat).pow(2).sum())
            # Same definition as fidelity(): variance about the (global) training-set mean.
            fvu_parts[tag][1] += float((flat - mean_act.mean(0).to(flat.device)).pow(2).sum())
        for b in range(a_rep[module].shape[0]):
            fen = fens_T[hist_rows[s + b]]
            _, idx = hook.legal_moves_and_indices(lichess.board_with_history(tst["uci_history"][hist_rows[s + b]]))
            kls.append(float(hook.legal_kl(o_rep["policy"][b], o_real["policy"][b], idx)))
    last_to = torch.from_numpy(history_filter.last_move_to_labels([tst["uci_history"][i] for i in hist_rows]))
    half = len(hist_rows) // 2
    lm_real = correlational.evaluate_property(codes_real.subset(range(half)), last_to[:half],
                                              codes_real.subset(range(half, len(hist_rows))), last_to[half:])
    lm_rep = correlational.evaluate_property(codes_rep.subset(range(half)), last_to[:half],
                                             codes_rep.subset(range(half, len(hist_rows))), last_to[half:])
    corr = agree["corr"]
    res["history"] = {
        "n_boards": len(hist_rows),
        "policy_kl_rep_vs_real_median": float(torch.tensor(kls).median()),
        "policy_kl_rep_vs_real_mean": float(torch.tensor(kls).mean()),
        "fvu_rep": fvu_parts["rep"][0] / fvu_parts["rep"][1],
        "fvu_real": fvu_parts["real"][0] / fvu_parts["real"][1],
        "measurable_features": int(surv["measurable"].sum()),
        "surviving_features": int(surv["survives"].sum()),
        "survive_frac": float(surv["survives"].sum() / surv["measurable"].sum().clamp_min(1)),
        "corr_quantiles": {q: float(torch.nanquantile(corr[surv["measurable"]], q)) for q in (0.05, 0.25, 0.5, 0.75)},
        "positive_control_last_move_to": {
            "real_history": lm_real, "rep_history": lm_rep,
            "selected_feature_corr": float(corr[lm_real["feature"]]),
            "selected_feature_survives": bool(surv["survives"][lm_real["feature"]]),
        },
    }
    for row in rows:
        f = row["feature"]
        row["history"] = {"corr": float(corr[f]), "measurable": bool(surv["measurable"][f]), "survives": bool(surv["survives"][f])}
    print("history:", json.dumps({k: v for k, v in res["history"].items() if k != "positive_control_last_move_to"}, indent=1))
    print("positive control:", json.dumps(res["history"]["positive_control_last_move_to"], indent=1))

    # ---------------------------------------------------------------- 6. verdicts
    # Causal threshold is CALIBRATED on the random live features: a property feature passes only if
    # it beats its matched controls more often than 90% of random features beat theirs. (A fixed
    # 0.5/0.75 is wrong: the targeted test is biased upward even for random features, because the
    # control usually fires on other squares than the candidate.)
    rnd = [c for c in res["causal_random_features"] if c.get("n_boards", 0) >= 32]
    q90 = lambda key: float(torch.tensor([c[key] for c in rnd if c[key] == c[key]]).quantile(0.9))
    thr = {"kl_win_rate": q90("kl_win_rate"), "target_win_rate": q90("target_win_rate")}
    res["causal_thresholds_random_q90"] = thr
    for row in rows:
        c = row["causal"]
        row["passes"] = {
            "correlational": row["test_f1"] >= 0.5 and row["test_f1"] > row["null_f1_p95"] + 0.1,
            "causal": c.get("n_boards", 0) >= 32 and (c["kl_win_rate"] > thr["kl_win_rate"]
                                                     or (c["target_win_rate"] or 0) > thr["target_win_rate"]),
            "history": row["history"]["survives"],
        }
        row["passes"]["all"] = all(row["passes"].values())
    res["properties"] = rows
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))

    lines = ["| property | origin | feat | test F1 (P/R) | AUROC | null p95 | probe layer | probe input | "
             "KL win | target win | hist corr | passes |", "|" + "---|" * 12]
    for r in sorted(rows, key=lambda r: -r["test_f1"]):
        pl, pi, c = r.get("probe_layer", {}), r.get("probe_input", {}), r["causal"]
        lines.append(
            f"| {r['property']} | {r['origin']} | {r['feature']} | {r['test_f1']:.2f} ({r['test_precision']:.2f}/{r['test_recall']:.2f}) "
            f"| {r['test_auroc']:.2f} | {r['null_f1_p95']:.2f} | {pl.get('test_f1', float('nan')):.2f} | {pi.get('test_f1', float('nan')):.2f} "
            f"| {c.get('kl_win_rate', float('nan')):.2f} | {c.get('target_win_rate', float('nan')):.2f} | {r['history']['corr']:.2f} "
            f"| {'ALL' if r['passes']['all'] else ','.join(k[:4] for k, v in r['passes'].items() if v and k != 'all') or '-'} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

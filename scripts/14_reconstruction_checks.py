"""Robustness checks on the board-reconstruction results (scripts/13), raised by the second audit.

    uv run python scripts/14_reconstruction_checks.py      # after scripts/13; ~10 min

Same SAEs, boards and features as scripts/13. For each SAE:
  * the paper's own protocol: Eq. 5 (best threshold per test board) and pieces on their starting
    squares left unscored (t chosen on train);
  * reconstruction at every threshold with >= 5 training firings required;
  * how many distinct features are the best coverage match for the 768 (square, piece) BSPs.
Plus: the always-fires baseline for every strategy property, and, for the trained / permuted / fresh
nets at layer 7, how much the 64 tokens collapse (within-board share of variance, mean cosine between
tokens) and whether the board survives in the token MEAN (a ridge probe from the mean vector to all
64 squares, against guessing each square's most common class).
Writes outputs/results/board_reconstruction_checks.json.
"""

import json
from pathlib import Path

import torch

from src import hook
from src.data import lichess
from src.eval import board_reconstruction as br
from src.eval.codes import encode_corpus
from src.sae import TopKSAE

SAES = ["L3", "L7", "L11", "L7_baseline_v0", "L7_freshnet", "L7_randnet"]
CORPUS = "outputs/positions/lichess_2024-06.parquet"
N_TEST = 2000                       # as scripts/13
N_MEAN_TRAIN, N_MEAN_TEST = 8000, 1500


def net_for(meta, cache={}):
    key = (meta.get("random_weights"), meta.get("random_init") or "permute")
    if key not in cache:
        m = hook.load_model()
        cache[key] = hook.randomize_weights(m, seed=key[0], mode=key[1]) if key[0] is not None else m
    return cache[key]


def protocol_checks(c_tr, y_tr, c_te, y_te, coverage_t):
    fmax = br.feature_max(c_tr)
    start = br.start_square_mask()
    preds_te, masked, min5 = [], {}, {}
    for t in br.THRESHOLDS:
        a = br.assign_features(c_tr, y_tr, fmax, t)
        p_te = br.reconstruct(c_te, a, fmax, t)
        preds_te.append(p_te)
        masked[t] = (br.score_boards(br.reconstruct(c_tr, a, fmax, t), y_tr, ignore=start)["f1"],
                     br.score_boards(p_te, y_te, ignore=start))
        a5 = br.assign_features(c_tr, y_tr, fmax, t, min_fires=5)
        s5 = br.score_boards(br.reconstruct(c_te, a5, fmax, t), y_te)
        min5[str(t)] = {"f1": s5["f1"], "exact_board_rate": s5["exact_board_rate"]}
    t_m = max(br.THRESHOLDS, key=lambda t: masked[t][0])
    return {
        "eq5_per_board_best_t_f1": br.per_board_best_f1(preds_te, y_te),
        "start_squares_unscored": {"t": t_m, "f1": masked[t_m][1]["f1"],
                                   "exact_board_rate": masked[t_m][1]["exact_board_rate"]},
        "min_fires_5_per_threshold": min5,
        "best_feature_sharing": br.best_feature_sharing(c_te, y_te, fmax, coverage_t),
    }


@torch.no_grad()
def token_geometry(model, module, fens, batch=250):
    """Within-board share of variance, mean off-diagonal token cosine, and the per-board token mean."""
    n = s1 = s2 = within = cos = 0.0
    means = []
    for s in range(0, len(fens), batch):
        a = hook.get_residual_stream_batch(model, fens[s:s + batch], module).double()   # (b, 64, D)
        mu = a.mean(1)
        means.append(mu.float().cpu())
        within += float(((a - mu[:, None]) ** 2).sum())
        s1 = s1 + a.sum((0, 1))
        s2 += float((a ** 2).sum())
        n += a.shape[0] * 64
        an = torch.nn.functional.normalize(a, dim=-1)
        c = an @ an.transpose(1, 2)
        cos += float((c.sum((1, 2)) - 64).sum() / (64 * 63))
    total = s2 - float((s1 ** 2).sum()) / n
    return within / total, cos / len(fens), torch.cat(means)


def mean_vector_probe(X_tr, y_tr, X_te, y_te, lam=100.0):
    """Ridge regression from the token-mean vector to one-hot classes on all 64 squares."""
    mu, sd = X_tr.mean(0), X_tr.std(0).clamp_min(1e-6)
    A = torch.cat([(X_tr - mu) / sd, torch.ones(len(X_tr), 1)], 1).double()
    B = torch.cat([(X_te - mu) / sd, torch.ones(len(X_te), 1)], 1).double()
    Y = torch.nn.functional.one_hot(y_tr, br.N_CLASSES).reshape(len(A), -1).double()
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], dtype=torch.double), A.T @ Y)
    pred = (B @ W).reshape(len(B), 64, br.N_CLASSES).argmax(-1)
    s = br.score_probe_predictions(pred, y_te)
    return {"f1": s["f1"], "exact_board_rate": s["exact_board_rate"],
            "square_accuracy": float((pred == y_te).double().mean())}


def main():
    tr = lichess.load_corpus(CORPUS, split="val")["fen"].tolist()
    te = lichess.load_corpus(CORPUS, split="test", limit=N_TEST)["fen"].tolist()
    y_tr, y_te = br.piece_grid(tr), br.piece_grid(te)
    recon = json.loads(Path("outputs/results/board_reconstruction.json").read_text())
    out_path = Path("outputs/results/board_reconstruction_checks.json")
    out = {"strategy_always_fires_f1": br.always_fires_f1(*br.strategy_labels(te)), "saes": {}, "token_geometry": {}}

    for name in SAES:
        p = f"outputs/sae/{name}.pt"
        meta = torch.load(p, map_location="cpu", weights_only=True)["meta"]
        sae, model = TopKSAE.load(p), net_for(meta)
        c_tr, c_te = encode_corpus(model, sae, meta["module"], tr), encode_corpus(model, sae, meta["module"], te)
        r = out["saes"][name] = protocol_checks(c_tr, y_tr, c_te, y_te, recon[name]["coverage_t"])
        print(f"{name:16s} Eq.5 F1 {r['eq5_per_board_best_t_f1']:.3f} | start squares unscored: F1 "
              f"{r['start_squares_unscored']['f1']:.3f} exact {r['start_squares_unscored']['exact_board_rate']:.3f} | "
              f"{r['best_feature_sharing']}", flush=True)
        out_path.write_text(json.dumps(out, indent=1))

    maj = torch.mode(y_tr[:N_MEAN_TRAIN], 0).values[None].expand(N_MEAN_TEST, 64)
    s = br.score_probe_predictions(maj, y_te[:N_MEAN_TEST])
    out["token_geometry"]["majority_class_per_square"] = {
        "f1": s["f1"], "exact_board_rate": s["exact_board_rate"],
        "square_accuracy": float((maj == y_te[:N_MEAN_TEST]).double().mean())}
    for name in ("L7", "L7_randnet", "L7_freshnet"):
        meta = torch.load(f"outputs/sae/{name}.pt", map_location="cpu", weights_only=True)["meta"]
        model = net_for(meta)
        wf, cos, M_tr = token_geometry(model, meta["module"], tr[:N_MEAN_TRAIN])
        _, _, M_te = token_geometry(model, meta["module"], te[:N_MEAN_TEST])
        g = out["token_geometry"][name] = {
            "within_board_variance_share": wf, "mean_token_cosine": cos,
            "token_mean_probe": mean_vector_probe(M_tr, y_tr[:N_MEAN_TRAIN], M_te, y_te[:N_MEAN_TEST])}
        print(f"{name:16s} within-board variance {wf:.4f}, token cosine {cos:.3f}, "
              f"token-mean probe {g['token_mean_probe']}", flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    print("majority class per square:", out["token_geometry"]["majority_class_per_square"])


if __name__ == "__main__":
    main()

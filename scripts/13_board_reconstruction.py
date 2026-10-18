"""Coverage + board reconstruction (Karvonen et al. 2024) for each SAE, with the paper's baselines.

    uv run python scripts/13_board_reconstruction.py outputs/sae/L3.pt outputs/sae/L7.pt ...

For every SAE: feature assignment on all ~15.5k Lichess VAL boards, scoring on 2000 TEST boards
(src/eval/board_reconstruction.py). Reported twice: with the paper's assignment rule (any feature
with >= 95% precision) and requiring >= 5 firings at that square on the training boards. (With 2000
training boards the paper's rule assigns many lucky one-off features: test F1 0.95 vs 0.98 here.) Baselines, as in the paper:
  * SAEs trained on random-weights copies of the net (outputs/sae/L7_randnet.pt = permuted trained
    weights, L7_freshnet.pt = conventional fresh init); activations come from the same random net,
    rebuilt from the seed/mode in the SAE's metadata;
  * a linear probe: 13-way (empty / 12 pieces) softmax regression on each token's residual vector,
    fit on train boards, its argmax scored with the same board F1.
Writes outputs/results/board_reconstruction.json and prints a table.
"""

import json
import sys
from pathlib import Path

import torch

from src import hook
from src.data import lichess
from src.eval import board_reconstruction as br
from src.eval.codes import encode_corpus
from src.sae import TopKSAE

N_TRAIN, N_TEST, N_PROBE = None, 2000, 1500   # None = the whole val split


def net_for(meta, cache={}):
    key = (meta.get("random_weights"), meta.get("random_init") or "permute")
    if key not in cache:
        m = hook.load_model()
        cache[key] = hook.randomize_weights(m, seed=key[0], mode=key[1]) if key[0] is not None else m
    return cache[key]


@torch.no_grad()
def residuals(model, module, fens, batch=250):
    return torch.cat([hook.get_residual_stream_batch(model, fens[s:s + batch], module).cpu()
                      for s in range(0, len(fens), batch)])                        # (N, 64, D)


def probe_reconstruction(X_tr, y_tr, X_te, y_te, steps=400, lr=0.05):
    """Multinomial logistic regression per token -> board F1 of its argmax predictions."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Xte = X_tr.reshape(-1, X_tr.shape[-1]).to(dev).float(), X_te.reshape(-1, X_te.shape[-1]).to(dev).float()
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    W = torch.zeros(Xtr.shape[1], br.N_CLASSES, device=dev, requires_grad=True)
    b = torch.zeros(br.N_CLASSES, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    ytr = y_tr.reshape(-1).to(dev)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(Xtr @ W + b, ytr).backward()
        opt.step()
    with torch.no_grad():
        pred = (Xte @ W + b).argmax(-1).reshape(y_te.shape).cpu()
    return br.score_probe_predictions(pred, y_te)


def main():
    paths = sys.argv[1:]
    corpus = "outputs/positions/lichess_2024-06.parquet"
    tr = lichess.load_corpus(corpus, split="val", limit=N_TRAIN)["fen"].tolist()
    te = lichess.load_corpus(corpus, split="test", limit=N_TEST)["fen"].tolist()
    y_tr, y_te = br.piece_grid(tr), br.piece_grid(te)
    strat_sq, strat_bd = br.strategy_labels(te)
    out_path = Path("outputs/results/board_reconstruction.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    probes_done = {}
    for p in paths:
        name = Path(p).stem
        meta = torch.load(p, map_location="cpu", weights_only=True)["meta"]
        sae, model = TopKSAE.load(p), net_for(meta)
        c_tr, c_te = encode_corpus(model, sae, meta["module"], tr), encode_corpus(model, sae, meta["module"], te)
        res = br.evaluate(c_tr, y_tr, c_te, y_te)
        res["strategy_coverage"] = br.strategy_coverage(c_te, strat_sq, strat_bd, br.feature_max(c_tr))
        res["min_fires_5"] = {k: v for k, v in br.evaluate(c_tr, y_tr, c_te, y_te, min_fires=5).items()
                              if k in ("reconstruction", "reconstruction_t")}
        key = (meta["module"], meta.get("random_weights"), meta.get("random_init"))
        if key not in probes_done:                        # probe depends on the net + layer, not the SAE
            X_tr = residuals(model, meta["module"], tr[:N_PROBE])
            X_te = residuals(model, meta["module"], te[:N_PROBE])
            probes_done[key] = probe_reconstruction(X_tr, y_tr[:N_PROBE], X_te, y_te[:N_PROBE])
        res["linear_probe"] = probes_done[key]
        res["layer"], res["random_weights"], res["random_init"] = meta["layer"], meta.get("random_weights"), meta.get("random_init")
        results[name] = res
        r = res["reconstruction"]
        print(f"{name:16s} coverage {res['coverage']:.3f} (t={res['coverage_t']})  reconstruction F1 {r['f1']:.3f} "
              f"exact {r['exact_board_rate']:.3f} (t={res['reconstruction_t']})  | probe F1 {res['linear_probe']['f1']:.3f} "
              f"exact {res['linear_probe']['exact_board_rate']:.3f}  | min5 F1 {res['min_fires_5']['reconstruction']['f1']:.3f} "
              f"exact {res['min_fires_5']['reconstruction']['exact_board_rate']:.3f}  | strategy coverage "
              f"{res['strategy_coverage']['mean']:.3f}", flush=True)
        out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

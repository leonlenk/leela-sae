"""Auto-interp: LLM labels + held-out detection scores for SAE features (src/autointerp/labeler.py).

    uv run python scripts/07_autointerp.py --sae outputs/sae/L7.pt

Labels two groups so they can be compared:
  * the features the correlational eval selected for each property (results.json), and
  * --n-random random LIVE features (density 1e-4 .. 0.2), i.e. what a typical feature looks like.
Mines activations on the Lichess TEST split boards used by scripts/04_evaluate.py. LLM calls go
through the Claude Code CLI (`claude -p`, uses your Claude subscription). Writes
outputs/results/<sae>/autointerp.json.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from src import hook
from src.autointerp import labeler
from src.data import lichess
from src.eval.codes import encode_corpus
from src.eval.feature_mining import density_from_codes
from src.sae import TopKSAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--n-boards", type=int, default=4000)
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    name = Path(args.sae).stem
    out_dir = Path("outputs/results", name)
    results = json.loads((out_dir / "results.json").read_text())
    meta = torch.load(args.sae, map_location="cpu", weights_only=True)["meta"]
    sae = TopKSAE.load(args.sae)
    model = hook.load_model()
    fens = lichess.load_corpus("outputs/positions/lichess_2024-06.parquet", split="test", limit=args.n_boards)["fen"].tolist()
    codes = encode_corpus(model, sae, meta["module"], fens)
    del model
    torch.cuda.empty_cache()

    jobs = {}
    for row in results["properties"]:
        jobs.setdefault(row["feature"], {"group": "property", "properties": []})["properties"].append(row["property"])
    dens = density_from_codes(codes)
    live = torch.nonzero((dens > 1e-4) & (dens < 0.2)).flatten()
    gen = torch.Generator().manual_seed(1)
    for f in live[torch.randperm(len(live), generator=gen)].tolist():
        if len([j for j in jobs.values() if j["group"] == "random"]) >= args.n_random:
            break
        jobs.setdefault(f, {"group": "random", "properties": []})

    llm = labeler.claude_cli(model=args.model)

    def run(f):
        try:
            r = labeler.interpret_feature(codes, fens, f, llm)
        except Exception as e:                     # one failed call shouldn't sink the batch
            r = {"feature": f, "error": str(e)[:300]}
        r.update(jobs[f], density=float(dens[f]))
        print(json.dumps({k: r.get(k) for k in ("feature", "group", "properties", "label", "balanced_accuracy",
                                                 "square_hit_rate", "error", "skipped")}), flush=True)
        return r

    with ThreadPoolExecutor(args.workers) as pool:
        out = list(pool.map(run, list(jobs)))
    (out_dir / "autointerp.json").write_text(json.dumps(out, indent=1))

    for group in ("property", "random"):
        ok = [r for r in out if r["group"] == group and "balanced_accuracy" in r]
        if ok:
            ba = sum(r["balanced_accuracy"] for r in ok) / len(ok)
            hit = sum(r["square_hit_rate"] for r in ok) / len(ok)
            print(f"{group}: n={len(ok)}  mean balanced accuracy {ba:.3f}  square hit rate {hit:.3f}")


if __name__ == "__main__":
    main()

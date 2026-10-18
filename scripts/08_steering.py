"""Steering (sufficiency) tests for derived-concept features: does WRITING the feature where it is off
move Leela's policy the way its meaning predicts?

    uv run python scripts/08_steering.py --sae outputs/sae/L7.pt

Each experiment names a feature (the one the correlational eval selected for a property), a kind of
square to write it on (where the property is FALSE and the feature is off), and the moves whose
probability its meaning predicts should RISE:

  enemy_hanging      write on a defended enemy piece we can capture   -> captures on that square
  own_hanging        write on an attacked-but-defended own piece       -> moves of that piece
  own_passed_pawn    write on an own pawn that is not passed           -> moves of that pawn
  own_king_in_check  write on the own king's square, not in check      -> king moves

Controls, both written on the SAME token:
  * feature control — a random live feature whose typical (p90) activation is within 30% of the
    candidate's, at its own p90 activation;
  * direction control — a random unit direction in residual space at the candidate's activation
    (no assumptions about the dictionary; matters when few features share the candidate's scale).
Reported: median change in log P(target moves) for candidate vs controls, and how often the
candidate raises it more.
Writes outputs/results/<sae>/steering.json.
"""

import argparse
import json
import random
from pathlib import Path

import chess
import torch

from src import hook
from src.data import lichess
from src.eval import board_props, causal
from src.eval.codes import encode_corpus
from src.eval.feature_mining import density_from_codes
from src.sae import TopKSAE
from src.viz.features import token_to_square

EXPERIMENTS = {
    # property: (token condition on per-square labels, target-move rule)
    "enemy_hanging": (lambda L, t: L["capture_target"][t] and not L["enemy_hanging"][t], "to"),
    "own_hanging": (lambda L, t: L["attacked_by_enemy"][t] and not L["own_hanging"][t]
                    and not L["own_king"][t] and not L["empty"][t] and any(L[f"own_{p}"][t] for p in
                    ("pawn", "knight", "bishop", "rook", "queen")), "from"),
    "own_passed_pawn": (lambda L, t: L["own_pawn"][t] and not L["own_passed_pawn"][t], "from"),
    "own_king_in_check": (lambda L, t: L["own_king"][t] and not L["own_king_in_check"][t], "from"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--n-boards", type=int, default=4000)
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 3.0])
    args = ap.parse_args()

    name = Path(args.sae).stem
    results = json.loads(Path("outputs/results", name, "results.json").read_text())
    feat_of = {r["property"]: r["feature"] for r in results["properties"]}
    meta = torch.load(args.sae, map_location="cpu", weights_only=True)["meta"]
    sae = TopKSAE.load(args.sae)
    model = hook.load_model()
    module = meta["module"]
    fens = lichess.load_corpus("outputs/positions/lichess_2024-06.parquet", split="test", limit=args.n_boards)["fen"].tolist()
    codes = encode_corpus(model, sae, module, fens)
    sq, _ = board_props.label_arrays(fens)
    dens = density_from_codes(codes)
    rng = random.Random(0)

    live = torch.nonzero((dens > 1e-4) & (dens < 0.2)).flatten().tolist()
    pool = rng.sample(live, min(300, len(live)))
    p90 = {f: causal.active_quantile(codes, f) for f in pool}

    out = {}
    for prop, (cond, rule) in EXPERIMENTS.items():
        f = feat_of[prop]
        a_f = causal.active_quantile(codes, f)
        controls = [g for g in pool if g != f and abs(p90[g] - a_f) <= 0.3 * a_f]
        act_f = codes.feature(f)                                             # (N, 64)
        trials = []                                                          # (board, token, target move idxs)
        for i in rng.sample(range(len(fens)), len(fens)):
            L = {k: v[i] for k, v in sq.items()}
            toks = [t for t in range(64) if cond(L, t) and act_f[i, t] == 0]
            if not toks:
                continue
            t = rng.choice(toks)
            s = token_to_square(fens[i], t)
            moves = list(chess.Board(fens[i]).legal_moves)
            tgt = [j for j, m in enumerate(moves) if (m.to_square if rule == "to" else m.from_square) == s]
            if tgt:
                trials.append((i, t, tgt))
            if len(trials) >= args.n_trials:
                break
        boards = [fens[i] for i, _, _ in trials]
        tokens = torch.tensor([t for _, t, _ in trials])
        targets = [tg for _, _, tg in trials]
        ctrl = torch.tensor([rng.choice(controls) for _ in trials])
        res = {"feature": f, "p90_activation": a_f, "n_trials": len(trials), "n_control_pool": len(controls)}
        for scale in args.scales:
            e_f = causal.steering_effects(model, sae, module, boards, torch.full((len(trials),), f), tokens,
                                          torch.full((len(trials),), a_f * scale), targets)
            e_c = causal.steering_effects(model, sae, module, boards, ctrl, tokens,
                                          torch.tensor([p90[int(c)] * scale for c in ctrl]), targets)
            gen = torch.Generator().manual_seed(int(scale * 1000))
            rand_dirs = torch.nn.functional.normalize(torch.randn(len(trials), sae.config["d_model"], generator=gen), dim=-1)
            e_r = causal.steering_effects(model, sae, module, boards, ctrl, tokens, torch.full((len(trials),), a_f * scale),
                                          targets, directions=rand_dirs)
            res[f"x{scale:g}"] = {
                "target_dlogp_median": float(e_f["target_dlogp"].median()),
                "control_target_dlogp_median": float(e_c["target_dlogp"].median()),
                "raises_more_than_control": float((e_f["target_dlogp"] > e_c["target_dlogp"]).float().mean()),
                "random_direction_target_dlogp_median": float(e_r["target_dlogp"].median()),
                "raises_more_than_random_direction": float((e_f["target_dlogp"] > e_r["target_dlogp"]).float().mean()),
                "frac_raised": float((e_f["target_dlogp"] > 0).float().mean()),
                "kl_median": float(e_f["kl"].median()),
                "control_kl_median": float(e_c["kl"].median()),
            }
        out[prop] = res
        print(prop, json.dumps(res, indent=1), flush=True)
    Path("outputs/results", name, "steering.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

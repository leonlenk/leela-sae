"""Joint ablation: is a derived concept's weak single-feature effect redundancy or recomputation?

    uv run python scripts/12_joint_ablation.py --sae outputs/sae/L7.pt

For each derived property, rank features by F1 on validation boards and remove the top-n together
(n = 1, 3, 10, 30; error-preserving) on test boards where the property holds. Measure the policy KL
and the change in log P(moves touching the property's squares). Control: for each removed feature,
the non-candidate feature on the same board with the closest activation mass, removed jointly — so
both sets remove the same amount of activation. If the effect grows with n and pulls away from the
control, the concept was spread over redundant features; if it stays flat, later layers rebuild it.

Piece-identity features are EXCLUDED from the candidate sets (any feature whose F1 against an input
property — a piece type on the square, or an empty square — exceeds 0.3): removing "my king is
here" would trivially change king moves and confound "in check".
Writes outputs/results/<sae>/joint_ablation.json.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from src import hook
from src.data import lichess
from src.eval import board_props, causal, correlational
from src.eval.codes import encode_corpus
from src.sae import TopKSAE

PROPS = ["own_king_in_check", "enemy_hanging", "own_pinned", "enemy_passed_pawn", "own_passed_pawn"]
NS = (1, 3, 10, 30)


def matched_controls(codes, board, candidates):
    """One control per active candidate: closest-mass non-candidate feature active on the board."""
    idx, val = codes.idx[board].reshape(-1), codes.val[board].reshape(-1)
    mass = torch.zeros(codes.n_features).index_add_(0, idx, val)
    pool = [f for f in torch.nonzero(mass > 0).flatten().tolist() if f not in set(candidates)]
    out = []
    for f in candidates:
        if mass[f] <= 0 or not pool:
            continue
        g = min(pool, key=lambda p: abs(float(mass[p] - mass[f])))
        pool.remove(g)
        out.append(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--n-boards", type=int, default=4000)
    ap.add_argument("--n-trials", type=int, default=128)
    args = ap.parse_args()
    meta = torch.load(args.sae, map_location="cpu", weights_only=True)["meta"]
    sae = TopKSAE.load(args.sae)
    model = hook.load_model()
    module = meta["module"]
    S = lichess.load_corpus("outputs/positions/lichess_2024-06.parquet", split="val", limit=args.n_boards)["fen"].tolist()
    T = lichess.load_corpus("outputs/positions/lichess_2024-06.parquet", split="test", limit=args.n_boards)["fen"].tolist()
    cS, cT = encode_corpus(model, sae, module, S), encode_corpus(model, sae, module, T)
    sqS, _ = board_props.label_arrays(S)
    sqT, _ = board_props.label_arrays(T)
    rng = random.Random(0)

    identity_f1 = torch.stack([correlational.token_f1_all(cS, torch.from_numpy(sqS[fam]))[2]
                               for fam, origin in board_props.SQUARE_FAMILIES.items() if origin == "input"]).amax(0)
    identity = identity_f1 > 0.3                                          # (F,) piece-identity-like features
    out = {"n_identity_features_excluded": int(identity.sum())}
    for prop in PROPS:
        f1 = correlational.token_f1_all(cS, torch.from_numpy(sqS[prop]))[2]
        f1[identity] = -1                                                 # never remove piece identity
        ranked = torch.argsort(f1, descending=True)[: max(NS)].tolist()
        boards = [i for i in range(len(T)) if sqT[prop][i].any()]
        boards = rng.sample(boards, min(args.n_trials, len(boards)))
        fens = [T[i] for i in boards]
        targets = [torch.nonzero(torch.from_numpy(sqT[prop][i])).flatten().tolist() for i in boards]
        res = {"n_boards": len(boards), "top_features": ranked[:10]}
        for n in NS:
            cand = [ranked[:n]] * len(boards)
            ctrl = [matched_controls(cT, i, ranked[:n]) for i in boards]
            e_f = causal.joint_ablation_effects(model, sae, module, fens, cand, targets)
            e_c = causal.joint_ablation_effects(model, sae, module, fens, ctrl, targets)
            ok = ~torch.isnan(e_f["target_dlogp"]) & ~torch.isnan(e_c["target_dlogp"])
            res[f"top{n}"] = {
                "kl_median": float(e_f["kl"].median()),
                "control_kl_median": float(e_c["kl"].median()),
                "kl_win_rate": float((e_f["kl"] > e_c["kl"]).float().mean()),
                "target_dlogp_median": float(e_f["target_dlogp"][ok].median()) if ok.any() else float("nan"),
                "control_target_dlogp_median": float(e_c["target_dlogp"][ok].median()) if ok.any() else float("nan"),
                "n_target_boards": int(ok.sum()),
            }
        out[prop] = res
        print(prop, json.dumps(res, indent=1), flush=True)
    Path("outputs/results", Path(args.sae).stem, "joint_ablation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

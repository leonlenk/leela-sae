"""SAE fidelity: how well does the SAE reconstruct the residual stream, and does the NET still behave
the same when its activation is replaced by the reconstruction?

Reconstruction (activation space):  FVU (fraction of variance unexplained) and L0.
Behaviour (splice recon into the net at the SAE's layer, read the heads; LEGAL moves only):
  * kl_recon     — KL(clean policy || spliced policy), nats, mean over boards
  * top1_agree   — fraction of boards where the spliced policy's top legal move is unchanged
  * abs_dq       — mean |ΔQ|, Q = P(win) − P(loss)
  * loss recovered = 1 − KL_recon / KL_ablate, against two ablation baselines:
        zero  — replace the activation by 0 (weak: a LayerNorm output is never 0, so the net is
                wrecked and almost anything "recovers" a lot of it);
        mean  — replace each token by the training-set mean activation AT THAT SQUARE (stricter: keeps
                the typical activation, deletes everything position-specific).
    The original metric used zero-ablation with a softmax over all 1858 policy slots (94% of whose
    mass sits on untrained illegal moves), which inflated it.
"""

from __future__ import annotations

from typing import Sequence

import chess
import torch

from src import hook


@torch.no_grad()
def fidelity(model, sae, module: str, fens: Sequence[str], mean_act: torch.Tensor, batch_size: int = 128) -> dict:
    """`mean_act`: (64, D) per-token mean activation over training data (the mean-ablation baseline)."""
    device = hook.model_device(model)
    sae = sae.to(device).eval()
    mean_act = mean_act.to(device).float()
    sx = sxx = serr = 0.0
    n_rows, l0_sum = 0, 0.0
    per = {k: [] for k in ("kl_recon", "kl_zero", "kl_mean", "top1_recon", "top1_mean", "dq_recon")}
    for s in range(0, len(fens), batch_size):
        chunk = list(fens[s:s + batch_size])
        x = hook.encode_fens(chunk)
        acts, clean = hook.run_with_cache(model, x, [module])
        a = acts[module]                                              # (B, 64, D)
        B, T, D = a.shape
        recon, feats, _ = sae(a.reshape(-1, D))
        recon = recon.reshape(B, T, D)
        flat = a.reshape(-1, D).double()
        sx = sx + flat.sum(0)
        sxx += float(flat.pow(2).sum())
        serr += float((recon - a).pow(2).sum())
        n_rows += B * T
        l0_sum += float((feats > 0).sum())

        spliced = hook.run_patched(model, x, module, lambda _: recon)
        zeroed = hook.run_patched(model, x, module, torch.zeros_like)
        meaned = hook.run_patched(model, x, module, lambda t: mean_act.expand_as(t))
        for b, fen in enumerate(chunk):
            _, idx = hook.legal_moves_and_indices(chess.Board(fen))
            lp = hook.legal_log_probs(clean["policy"][b], idx)
            top = lp.argmax()
            for name, out in (("recon", spliced), ("zero", zeroed), ("mean", meaned)):
                lq = hook.legal_log_probs(out["policy"][b], idx)
                per[f"kl_{name}"].append(float((lp.exp() * (lp - lq)).sum()))
                if name in ("recon", "mean"):
                    per[f"top1_{name}"].append(float(lq.argmax() == top))
            q = lambda w: w[0] - w[2]
            per["dq_recon"].append(float(q(spliced["wdl"][b]) - q(clean["wdl"][b])))

    t = {k: torch.tensor(v) for k, v in per.items()}
    total_var = sxx - float((sx ** 2).sum()) / n_rows                # sum of squared deviations from mean
    kl_r, kl_z, kl_m = t["kl_recon"].mean(), t["kl_zero"].mean(), t["kl_mean"].mean()
    return {
        "n_boards": len(fens),
        "fvu": serr / total_var,
        "l0": l0_sum / n_rows,
        "kl_recon_mean": float(kl_r),
        "kl_recon_median": float(t["kl_recon"].median()),
        "kl_recon_p90": float(t["kl_recon"].quantile(0.9)),
        "kl_zero_mean": float(kl_z),
        "kl_mean_ablation_mean": float(kl_m),
        "loss_recovered_vs_zero": float(1 - kl_r / kl_z),
        "loss_recovered_vs_mean": float(1 - kl_r / kl_m),
        "top1_agree_recon": float(t["top1_recon"].mean()),
        "top1_agree_mean_ablation": float(t["top1_mean"].mean()),
        "abs_dq_recon_mean": float(t["dq_recon"].abs().mean()),
    }


def mean_activation(acts_grouped: torch.Tensor, n_boards: int = 20_000) -> torch.Tensor:
    """(64, D) per-token mean over the first n_boards of a (N, 64, D) activation cache."""
    return acts_grouped[:n_boards].float().mean(0)

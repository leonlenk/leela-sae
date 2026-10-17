"""Single-GPU Top-K SAE training loop + its loss pieces.

Objective: normalised reconstruction MSE (no L1 — Top-K already fixes sparsity) plus the AuxK loss
that revives dead latents. Data: an in-memory (N, D) activation tensor (fp16 is fine; batches are
upcast on the GPU), sampled as shuffled minibatches so the 64 correlated tokens of a board are spread
across many steps.

Network-level fidelity (does the net still play the same when the SAE is spliced in?) lives in
src/eval/fidelity.py; this file only needs activations.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

DEFAULTS = {
    "batch_size": 4096,
    "epochs": 2,
    "lr": None,              # None -> Gao et al. scaling rule 2e-4 / sqrt(F / 2^14)
    "warmup_steps": 200,
    "decay_frac": 0.2,       # linearly decay the LR to 0 over the last 20% of steps
    "aux_coef": 1 / 32,
    "aux_k": 512,
    "dead_tokens": 2_000_000,  # a latent is "dead" if it hasn't fired in this many training tokens
    "grad_clip": 1.0,
    "log_every": 100,
    "seed": 0,
}


def reconstruction_loss(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Plain MSE. (B, D) -> scalar."""
    return F.mse_loss(recon, x)


def normalized_mse(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE divided by the variance of x around its batch mean: the fraction of variance unexplained
    (FVU). 1.0 == no better than predicting the mean; scale-free, so losses are comparable across
    layers and the aux term's coefficient means the same thing everywhere."""
    return reconstruction_loss(recon, x) / (x - x.mean(0, keepdim=True)).pow(2).mean().clamp_min(1e-12)


def l0_norm(feats: torch.Tensor) -> torch.Tensor:
    """Mean number of nonzero features per token. (B, F) -> scalar. <= k for a Top-K SAE."""
    return torch.count_nonzero(feats, dim=-1).float().mean()


def update_tokens_since_fired(counter: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    """(F,) tokens elapsed since each latent last fired; reset to 0 for latents that fired now."""
    counter += feats.shape[0]
    counter[(feats > 0).any(0)] = 0
    return counter


def aux_k_loss(x: torch.Tensor, recon: torch.Tensor, pre: torch.Tensor, dead_mask: torch.Tensor,
               aux_k: int, sae) -> torch.Tensor:
    """AuxK loss (Gao et al. §A.2): let ONLY the dead latents try to reconstruct what the live ones
    missed, e = x - recon (detached). We take each token's top-aux_k (ReLU'd) dead pre-activations,
    decode them with W_dec alone (no b_pre: e already has the bias removed), and score with the
    normalised MSE. Gradient therefore flows only into dead latents, pulling them towards directions
    the dictionary is currently missing. Returns 0 when nothing is dead."""
    if not dead_mask.any():
        return pre.new_zeros(())
    dead_pre = F.relu(pre[:, dead_mask])                              # (B, F_dead), >= 0
    k = min(aux_k, dead_pre.shape[-1])
    top = torch.topk(dead_pre, k, dim=-1)
    codes = torch.zeros_like(pre)
    dead_idx = torch.nonzero(dead_mask).flatten()
    codes.scatter_(-1, dead_idx[top.indices], top.values)              # back into full-width (B, F)
    aux_recon = sae.decoder(codes)                                     # (B, D), no pre_bias
    target = (x - recon).detach()
    return normalized_mse(aux_recon, target)


@torch.no_grad()
def fvu(sae, acts: torch.Tensor, chunk: int = 8192) -> float:
    """Fraction of variance unexplained over a whole (N, D) set, chunked to bound GPU memory."""
    device = sae.decoder.weight.device
    mean = acts.float().mean(0).to(device)
    err = var = 0.0
    for s in range(0, acts.shape[0], chunk):
        x = acts[s:s + chunk].to(device).float()
        err += float((sae(x)[0] - x).pow(2).sum())
        var += float((x - mean).pow(2).sum())
    return err / var


def _lr_lambda(total_steps: int, warmup: int, decay_frac: float):
    decay_start = int(total_steps * (1 - decay_frac))

    def f(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if step >= decay_start:
            return max(0.0, (total_steps - step) / max(1, total_steps - decay_start))
        return 1.0
    return f


def train(sae, acts: torch.Tensor, config: dict | None = None, val_acts: torch.Tensor | None = None,
          device: str | None = None) -> tuple[torch.nn.Module, list[dict]]:
    """Train `sae` on activation rows `acts` (N, D). Returns (sae, history of logged metrics)."""
    cfg = {**DEFAULTS, **(config or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device).train()
    F_dim = sae.decoder.weight.shape[1]
    lr = cfg["lr"] or 2e-4 / math.sqrt(F_dim / 2 ** 14)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999), eps=6.25e-10)

    N, bs = acts.shape[0], cfg["batch_size"]
    steps_per_epoch = N // bs
    total = steps_per_epoch * cfg["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, cfg["warmup_steps"], cfg["decay_frac"]))
    gen = torch.Generator().manual_seed(cfg["seed"])
    since_fired = torch.zeros(F_dim, device=device)

    history, step = [], 0
    bar = tqdm(total=total, desc="train SAE")
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(N, generator=gen)
        for s in range(steps_per_epoch):
            x = acts[perm[s * bs:(s + 1) * bs]].to(device, non_blocking=True).float()   # (B, D)
            recon, feats, pre = sae(x)
            dead = since_fired > cfg["dead_tokens"]
            mse = normalized_mse(recon, x)
            aux = aux_k_loss(x, recon, pre, dead, cfg["aux_k"], sae)
            loss = mse + cfg["aux_coef"] * aux

            opt.zero_grad(set_to_none=True)
            loss.backward()
            sae.remove_parallel_decoder_grad()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            sae.normalize_decoder()
            with torch.no_grad():
                update_tokens_since_fired(since_fired, feats)

            if step % cfg["log_every"] == 0 or step == total - 1:
                rec = {"step": step, "tokens": (step + 1) * bs, "lr": sched.get_last_lr()[0],
                       "fvu": mse.item(), "aux": aux.item(), "l0": l0_norm(feats).item(),
                       "dead_frac": float(dead.float().mean())}
                if val_acts is not None:
                    rec["val_fvu"] = fvu(sae, val_acts)
                history.append(rec)
                bar.set_postfix({k: round(v, 4) for k, v in rec.items() if k in ("fvu", "val_fvu", "l0", "dead_frac")})
            step += 1
            bar.update(1)
    bar.close()
    return sae.eval(), history

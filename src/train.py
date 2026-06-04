"""SAE training: single-GPU loop + loss/metrics (🔒 — you write these, §3.5 scaffold).

The Accelerate multi-GPU wrapper is ⚙️ and comes LATER — only after this single-GPU loop works
and the tests pass (DESIGN.md §4 step 3b). Don't add it here yet.
"""

from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from nnsight import NNsight
from lczerolens import LczeroBoard
from lczerolens.board import InputEncoding

def reconstruction_loss(recon, x):
    """Plain MSE between reconstruction and input. (B, D) -> scalar.

    This is the ENTIRE sparsity-free objective. Top-K already pins L0 == k, so there is
    deliberately NO L1 term — adding one would re-introduce the feature-suppression bias
    we chose Top-K to avoid.
    """
    return F.mse_loss(recon, x)

def l0_norm(feats):
    """Mean number of nonzero features per token. (B, F) -> scalar.

    A sanity probe, not a loss: for a correct Top-K SAE this should equal k exactly. If it
    drifts from k, the top-k mask is wrong.
    """
    count = torch.count_nonzero(feats, dim=-1).float()
    count = count.mean(dim=0)
    return count


def update_dead_latents(fired_counts, feats):
    """Accumulate how often each latent fired, so we can flag "dead" (never-firing) latents.

    `fired_counts`: (F,) running count. `feats`: (B, F) this step. Returns the updated (F,) count.
    Dead latents are the SAE's failure mode — they occupy dictionary capacity but encode nothing.
    """
    counts = torch.count_nonzero(feats, dim=0)
    fired_counts += counts
    return fired_counts


def aux_k_loss(x, recon, feats_pre, dead_mask, aux_k, sae):
    """Auxiliary loss that revives dead latents (the "aux_k"/ghost-grads trick).

    Idea (read before coding): the residual the model FAILED to reconstruct is (x - recon).
    We let ONLY the dead latents (dead_mask over F) try to reconstruct that residual using their
    top-aux_k activations, and add the MSE of that attempt. This routes gradient into latents that
    never fire, giving them a chance to come back to life. If there are no dead latents -> 0.

    `feats_pre`: (B, F) pre-top-k activations. `dead_mask`: (F,) bool. -> scalar.
    """
    if not dead_mask.any():
        return 0
    masked_feats_pre = feats_pre[:, dead_mask]              # (B, F_dead)
    k = min(aux_k, masked_feats_pre.shape[-1])
    topk = torch.topk(masked_feats_pre, k, dim=-1)
    sub = torch.zeros_like(masked_feats_pre)               # (B, F_dead)
    sub.scatter_(dim=-1, index=topk.indices, src=topk.values)
    out = torch.zeros_like(feats_pre)                      # (B, F) full width, live cols stay 0
    out[:, dead_mask] = sub                                # place dead-subset at its real cols
    new_recon = sae.decoder(out)                         # (B, D)
    target = (x - recon).detach()                          # the error the live latents left behind
    return reconstruction_loss(new_recon, target)


def loss_recovered(base_model, fen, sae, module_name):
    """"Loss recovered": substitute the SAE recon back into the net and see how much the net's
    own behaviour is preserved. The real quality metric — pure MSE can look great while the net
    falls apart. 1.0 == recon is as good as the true activation, 0.0 == as bad as zero-ablation.

    Single position. `fen` is a FEN string. Behaviour drift is measured as KL(clean_policy ||
    patched_policy) (option A): loss_clean == KL(clean||clean) == 0, so the recovered fraction
    collapses to (loss_zero - loss_recon) / loss_zero.
    """
    board = LczeroBoard(fen)
    device = next(base_model.parameters()).device
    board_input = board.to_input_tensor(input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED).to(device)

    nn_model: Any = NNsight(base_model)
    resid_module = nn_model.get(module_name)
    policy_module = nn_model.get("module.output/policy")

    with torch.no_grad():
        # no intervention
        with nn_model.trace(board_input):
            act = resid_module.output.save()             # (1, 64, D)
            clean_logits = policy_module.output.save()   # (1, n_moves)

        # run sae
        recon = sae(act.squeeze(0))[0].unsqueeze(0)      # (1, 64, D)
        zeros = torch.zeros_like(act)                    # the zero-ablation baseline

        # splice the sae's reconstruction into the base model
        with nn_model.trace(board_input):
            resid_module.output = recon
            recon_logits = policy_module.output.save()

        # splice in zeros instead for the counterfactual
        with nn_model.trace(board_input):
            resid_module.output = zeros
            zero_logits = policy_module.output.save()

    # KL(clean || patched) over the policy distribution: p_clean * (logp_clean - logp_patched), summed.
    p_clean = clean_logits.softmax(-1)                   # (1, n_moves)
    logp_clean = clean_logits.log_softmax(-1)
    loss_recon = (p_clean * (logp_clean - recon_logits.log_softmax(-1))).sum()
    loss_zero = (p_clean * (logp_clean - zero_logits.log_softmax(-1))).sum()

    if loss_zero < 1e-8:
        return torch.tensor(1.0)
    return (loss_zero - loss_recon) / loss_zero          # loss_clean == 0 -> denominator is loss_zero


def mean_loss_recovered(base_model, fens, sae, module_name):
    """Average loss_recovered over a held-out FEN list (the thin eval loop). One board is noisy;
    the reported metric is the mean over a held-out set. Returns a python float."""
    vals = [float(loss_recovered(base_model, f, sae, module_name)) for f in fens]
    return sum(vals) / len(vals) if vals else float("nan")


def train(config, sae, base_model, loader, F_dim, eval_fens=None, module_name=None):
    """Single-GPU training loop (DESIGN.md §4 step 2). No Ray, no Accelerate yet.

    Watch ONE feature until you believe it — this loop exists to build intuition, not throughput.

    `eval_fens` + `module_name` are optional: pass a held-out FEN list to log loss_recovered during
    training. Left None (e.g. in fast tests) the loop trains on cached acts alone, no live net needed.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = sae.to(device)
    sae.train()
    optimizer = torch.optim.NAdam(sae.parameters(), config["lr"])

    warmup_steps = config.get("warmup_steps", 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / warmup_steps) if warmup_steps > 0 else 1.0,
    )

    fired_counts = torch.zeros(F_dim, device=device)                  # (F,) running fire count for the CURRENT window
    dead_mask = torch.zeros(F_dim, dtype=torch.bool, device=device)   # (F,) nothing counts as dead until window 1 ends

    step = 0
    for epoch in range(config["epochs"]):            
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for (batch,) in pbar:                          # TensorDataset yields 1-tuples; batch: (B, D)
            batch = batch.to(device)
            recon, feats, pre_feats = sae(batch)     # feats: post-top-k (B,F); pre_feats: pre-top-k
            fired_counts = update_dead_latents(fired_counts, feats)   # tally this batch's fires (F,)

            loss = (reconstruction_loss(recon, batch)
                    + config["aux_coef"] * aux_k_loss(batch, recon, pre_feats,
                                                      dead_mask, config["aux_k"], sae))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            with torch.no_grad():                      # re-unit-norm decoder cols after the step
                sae.decoder.weight.data = F.normalize(sae.decoder.weight, dim=-2)

            # keeps track of when features are dead for the aux loss
            if (step + 1) % config["dead_window"] == 0:
                dead_mask = fired_counts == 0
                fired_counts = torch.zeros(F_dim, device=device)

            if step % config["train_eval_freq"] == 0:
                postfix = dict(
                    loss=loss.item(),
                    l0_norm=float(l0_norm(feats)),              # sanity: should sit at k
                    dead=float(dead_mask.float().mean()),  # fraction of the dictionary currently dead
                )

                if eval_fens is not None and module_name is not None:
                    sae.eval()
                    sample = eval_fens[: config.get("eval_sample", 16)]
                    postfix["recov"] = mean_loss_recovered(base_model, sample, sae, module_name)
                    sae.train()
                pbar.set_postfix(postfix)

            step += 1    

    return sae

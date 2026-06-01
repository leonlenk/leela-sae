"""SAE training: single-GPU loop + loss/metrics (🔒 — you write these, §3.5 scaffold).

The Accelerate multi-GPU wrapper is ⚙️ and comes LATER — only after this single-GPU loop works
and the tests pass (DESIGN.md §4 step 3b). Don't add it here yet.
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(recon, x):
    """Plain MSE between reconstruction and input. (B, D) -> scalar.

    This is the ENTIRE sparsity-free objective. Top-K already pins L0 == k, so there is
    deliberately NO L1 term — adding one would re-introduce the feature-suppression bias
    we chose Top-K to avoid.
    """
    # TODO (1): return F.mse_loss(recon, x).
    raise NotImplementedError


def l0_norm(feats):
    """Mean number of nonzero features per token. (B, F) -> scalar.

    A sanity probe, not a loss: for a correct Top-K SAE this should equal k exactly. If it
    drifts from k, the top-k mask is wrong.
    """
    # TODO (1): count nonzeros per row (feats != 0), then take the mean over the batch.
    raise NotImplementedError


def update_dead_latents(fired_counts, feats):
    """Accumulate how often each latent fired, so we can flag "dead" (never-firing) latents.

    `fired_counts`: (F,) running count. `feats`: (B, F) this step. Returns the updated (F,) count.
    Dead latents are the SAE's failure mode — they occupy dictionary capacity but encode nothing.
    """
    # TODO (1): add the per-latent nonzero count of this batch to fired_counts and return it.
    raise NotImplementedError


def aux_k_loss(x, recon, feats_pre, dead_mask, aux_k):
    """Auxiliary loss that revives dead latents (the "aux_k"/ghost-grads trick).

    Idea (read before coding): the residual the model FAILED to reconstruct is (x - recon).
    We let ONLY the dead latents (dead_mask over F) try to reconstruct that residual using their
    top-aux_k activations, and add the MSE of that attempt. This routes gradient into latents that
    never fire, giving them a chance to come back to life. If there are no dead latents -> 0.

    `feats_pre`: (B, F) pre-top-k activations. `dead_mask`: (F,) bool. -> scalar.
    """
    # TODO (1): if dead_mask has no True entries, return a zero scalar (nothing to revive).
    # TODO (2): take feats_pre but keep only columns where dead_mask is True.   # (B, F)
    # TODO (3): among those, keep the top-aux_k per row (same top-k idea, on the dead subset).
    # TODO (4): decode that sparse code back to (B, D) and MSE it against (x - recon).detach().
    raise NotImplementedError


def loss_recovered(model, board, sae, module_name):
    """"Loss recovered": substitute the SAE recon back into the net and see how much the net's
    own behaviour is preserved. The real quality metric — pure MSE can look great while the net
    falls apart. 1.0 == recon is as good as the true activation, 0.0 == as bad as zero-ablation.
    """
    # TODO (1): get clean output["policy"] (and/or wdl) from model(board).
    # TODO (2): get the activation at module_name, run it through the SAE, and patch the recon
    #           back in at that site (nnsight trace) -> output_recon.
    # TODO (3): also measure output_zero with the activation zero-ablated (the bad baseline).
    # TODO (4): return (loss_zero - loss_recon) / (loss_zero - loss_clean) using cross-entropy
    #           of the policy vs the clean policy (or KL). Higher is better.
    raise NotImplementedError


def train(config):
    """Single-GPU training loop (DESIGN.md §4 step 2). No Ray, no Accelerate yet.

    Watch ONE feature until you believe it — this loop exists to build intuition, not throughput.
    """
    # TODO (1): build the TopKSAE from config (d_model, dict_size, k) and an Adam optimiser.
    # TODO (2): init sae.pre_bias to the data mean of a warmup batch (mirrors src/sae.py TODO 1).
    # TODO (3): for each batch of cached activations x:  (B, D)
    #             recon, feats = sae(x)
    #             loss = reconstruction_loss(recon, x) + aux_coef * aux_k_loss(...)   # TODO (A)/(B)
    #             backward, step, then RE-NORMALISE decoder columns to unit norm.
    # TODO (4): every N steps log L0 (== k), fraction-dead, and loss_recovered.       # TODO (C)
    # TODO (5): checkpoint the SAE so the eval + canary tests can load it.
    raise NotImplementedError

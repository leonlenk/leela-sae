"""The Top-K sparse autoencoder (🔒 — you write the bodies; this file is a scaffold per DESIGN.md §3.5)."""

import torch, torch.nn as nn


class TopKSAE(nn.Module):
    """One-layer Top-K sparse autoencoder.

    Shapes:  D = d_model (residual-stream width, e.g. 768)
             F = dict_size = expansion_factor * D  (the over-complete basis)
    A forward pass maps activations (B, D) -> sparse features (B, F) -> recon (B, D).
    Top-K means: keep only the k largest feature activations per token, zero the
    rest. This gives DIRECT control of sparsity (L0 == k) with no L1 penalty, so
    we avoid the L1 "feature suppression" that biases reconstruction.
    """
    def __init__(self, d_model: int, dict_size: int, k: int):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_model, dict_size)            # W_enc: (D -> F)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)# W_dec: (F -> D)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))      # subtracted on input
        # TODO (1): initialise pre_bias to the DATA MEAN (or geometric median) of a
        #           batch of activations, so reconstruction starts from a sane point.
        # TODO (2): normalise decoder columns to unit L2 norm (and re-normalise after
        #           each optimiser step). Without this the basis directions aren't
        #           comparable and feature magnitudes are arbitrary.

    def encode(self, x):                       # x: (B, D) -> feats: (B, F)
        # TODO (3): subtract pre_bias from x.                # (B, D)
        # TODO (4): apply self.encoder.                      # (B, F) pre-activations
        # TODO (5): keep only the top-k values per row, zero everything else.
        #           Hint: torch.topk over dim=-1, scatter back into a zeros tensor.
        raise NotImplementedError

    def decode(self, feats):                   # feats: (B, F) -> recon: (B, D)
        # TODO (6): self.decoder(feats) + pre_bias.          # (B, D)
        raise NotImplementedError

    def forward(self, x):                      # -> (recon (B,D), feats (B,F))
        # TODO (7): feats = encode(x); recon = decode(feats); return both.
        raise NotImplementedError


# Training loop lives in train.py. The breadcrumbs you'll implement there:
#   # TODO (A): loss = mse(recon, x)   — NO L1 term; Top-K already enforces sparsity
#   # TODO (B): add aux_k loss: let dead latents reconstruct the residual (x - recon)
#   #           so they get gradient and can revive. Track which latents are "dead".
#   # TODO (C): log every step: L0 (== k here, sanity check), fraction-dead,
#   #           and "loss recovered" (substitute recon into the net, compare CE).

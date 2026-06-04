"""The Top-K sparse autoencoder (🔒 — you write the bodies; this file is a scaffold per DESIGN.md §3.5)."""

import torch, torch.nn as nn
import torch.nn.functional as F

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
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight, dim=-2)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))      # subtracted on input

    def init_pre_bias(self, batch):
        self.pre_bias.data = batch.mean(0)

    def encode(self, x):                       # x: (B, D) -> feats: (B, F)
        pre = x - self.pre_bias
        pre = self.encoder(pre)
        pre_relued = F.relu(pre)
        topk = torch.topk(pre_relued, self.k, dim=-1)
        out = torch.zeros_like(pre_relued)
        out.scatter_(dim=-1, index = topk.indices, src=topk.values)
        return out, pre

    def decode(self, feats):                   # feats: (B, F) -> recon: (B, D)
        return self.decoder(feats) + self.pre_bias

    def forward(self, x):                      # -> (recon (B,D), feats (B,F))
        feats, pre = self.encode(x)
        recon = self.decode(feats)
        return recon, feats, pre


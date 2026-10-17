"""Top-K sparse autoencoder (Gao et al. 2024, "Scaling and evaluating sparse autoencoders").

Shapes: D = d_model (768 here), F = dict_size (the over-complete basis, e.g. 8 x 768 = 6144).

    encode:  z = TopK( ReLU( W_enc (x - b_pre) + b_enc ) )      (B, D) -> (B, F), <= k nonzeros/row
    decode:  x_hat = W_dec z + b_pre                             (B, F) -> (B, D)

Top-K gives DIRECT control of sparsity (L0 <= k, == k whenever k pre-activations are positive) with
no L1 penalty, so there's no L1 "shrinkage" bias on the kept activations.

Choices that matter (each one fixed a failure mode of the first version):
  * Decoder columns are unit-norm, so a feature's activation IS its contribution's magnitude and
    activations are comparable across features. Gradient components parallel to the columns are
    projected out before each step (they'd only change the norm, which we reset anyway).
  * W_enc is initialised to W_dec^T. With an independent random encoder many latents start pointing
    nowhere useful and never enter anyone's top-k — they're dead from step 0.
  * b_pre starts at the data mean, so reconstruction starts from a sane point.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, dict_size: int, k: int):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_model, dict_size)                  # W_enc (F, D), b_enc (F,)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)       # W_dec (D, F)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))            # b_pre (D,)
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(torch.randn(d_model, dict_size), dim=0))
            self.encoder.weight.copy_(self.decoder.weight.T)          # tied init
            self.encoder.bias.zero_()

    @property
    def config(self) -> dict:
        return {"d_model": self.decoder.weight.shape[0], "dict_size": self.decoder.weight.shape[1], "k": self.k}

    def init_pre_bias(self, batch: torch.Tensor) -> None:
        """Set b_pre to the per-dimension mean of a (N, D) batch of activations (in place)."""
        with torch.no_grad():
            self.pre_bias.copy_(batch.float().mean(0).to(self.pre_bias.device))

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        """W_enc (x - b_pre) + b_enc, before ReLU / top-k. (B, D) -> (B, F)."""
        return self.encoder(x - self.pre_bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (feats (B, F) with <= k nonzeros per row, pre-activations (B, F))."""
        pre = self.pre_activations(x)
        top = torch.topk(F.relu(pre), self.k, dim=-1)                  # the k largest, all >= 0
        feats = torch.zeros_like(pre).scatter_(-1, top.indices, top.values)
        return feats, pre

    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        return self.decoder(feats) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (recon (B, D), feats (B, F), pre-activations (B, F))."""
        feats, pre = self.encode(x)
        return self.decode(feats), feats, pre

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    @torch.no_grad()
    def remove_parallel_decoder_grad(self) -> None:
        """Project out the gradient component along each (unit) decoder column."""
        W, G = self.decoder.weight, self.decoder.weight.grad
        if G is not None:
            G -= W * (W * G).sum(0, keepdim=True)

    # ----------------------------------------------------------------------------- persistence
    def save(self, path, **extra) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "TopKSAE":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        sae = cls(**ckpt["config"])
        sae.load_state_dict(ckpt["state_dict"])
        return sae.eval()

"""Run the net + SAE over a corpus once and keep the SAE's sparse codes (the TopK support per token).

A TopK SAE has exactly k nonzero features per token, so the full (N, 64, F) activation grid is
mostly zeros. We store only (idx, val) of shape (N, 64, k): ~4 bytes x k per token instead of F.
Every eval (correlational, causal candidate selection, feature mining, history filter) reads these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from src import hook


@dataclass
class SparseCodes:
    idx: torch.Tensor   # (N, 64, k) long  — which features fired on each token
    val: torch.Tensor   # (N, 64, k) float — their activations (>= 0; 0 means "in top-k but ReLU'd to 0")
    n_features: int

    @property
    def n_boards(self) -> int:
        return self.idx.shape[0]

    def tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat per-token view: (N*64, k) idx and val. Row r <-> board r // 64, token r % 64."""
        k = self.idx.shape[-1]
        return self.idx.reshape(-1, k), self.val.reshape(-1, k)

    def feature(self, f: int) -> torch.Tensor:
        """Dense (N, 64) activation of ONE feature (0 where it is not in a token's top-k)."""
        return (self.val * (self.idx == f)).sum(-1)

    def board_max(self) -> torch.Tensor:
        """Dense (N, F): each feature's max activation over the 64 tokens of each board."""
        out = torch.zeros(self.n_boards, self.n_features)
        k = self.idx.shape[-1]
        flat_idx = self.idx.reshape(self.n_boards, -1)
        flat_val = self.val.reshape(self.n_boards, -1)
        return out.scatter_reduce_(1, flat_idx, flat_val, reduce="amax", include_self=True)

    def subset(self, rows) -> "SparseCodes":
        return SparseCodes(self.idx[rows], self.val[rows], self.n_features)


@torch.no_grad()
def encode_corpus(model, sae, module: str, inputs: torch.Tensor | Sequence[str], batch_size: int = 256) -> SparseCodes:
    """SAE codes for every board. `inputs` is a list of FENs (canonical encoding) or a precomputed
    (N, 112, 8, 8) input tensor (e.g. real-history inputs from the history filter)."""
    device = hook.model_device(model)
    sae = sae.to(device).eval()
    idxs, vals = [], []
    n = len(inputs)
    for s in range(0, n, batch_size):
        chunk = inputs[s:s + batch_size]
        x = hook.encode_fens(chunk) if not torch.is_tensor(chunk) else chunk
        acts, _ = hook.run_with_cache(model, x, [module])
        a = acts[module]                                        # (b, 64, D)
        feats = sae.encode(a.reshape(-1, a.shape[-1]))[0]      # (b*64, F) post-top-k
        top = torch.topk(feats, sae.k, dim=-1)                  # the support (exactly k per token)
        idxs.append(top.indices.reshape(-1, 64, sae.k).cpu())
        vals.append(top.values.reshape(-1, 64, sae.k).float().cpu())
    return SparseCodes(torch.cat(idxs), torch.cat(vals), sae.decoder.weight.shape[1])

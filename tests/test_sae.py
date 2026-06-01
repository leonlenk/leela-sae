"""Tests for the Top-K SAE, written FIRST (DESIGN.md §3.5a). Run red, implement src/sae.py TODOs, run green."""

import torch

from src.sae import TopKSAE


def test_topk_enforces_l0_equals_k():
    sae = TopKSAE(d_model=16, dict_size=64, k=4)
    feats = sae.encode(torch.randn(8, 16))          # (B=8, F=64)
    # The whole point of Top-K: exactly k features are nonzero per row.
    assert (feats != 0).sum(dim=-1).tolist() == [4] * 8


def test_topk_keeps_the_LARGEST_activations():
    # Conceptual check: top-k must keep the k LARGEST pre-activations, not any k. To make the
    # survivors predictable we pin the encoder to the identity (pre-activations == input), so we
    # know exactly which features SHOULD survive and can assert the kept set is the true top-k.
    sae = TopKSAE(d_model=8, dict_size=8, k=3)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(8))      # pre-activation_j == x_j
        sae.encoder.bias.zero_()
        sae.pre_bias.zero_()
    x = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.05, 0.7, 0.0, 0.3]])   # top-3 are indices {1, 3, 5}
    feats = sae.encode(x)                            # (1, 8)
    kept = set((feats[0] != 0).nonzero().flatten().tolist())
    assert kept == {1, 3, 5}


def test_decoder_columns_unit_norm():
    sae = TopKSAE(d_model=16, dict_size=64, k=4)
    norms = sae.decoder.weight.norm(dim=0)          # one norm per dict column
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_forward_returns_recon_and_feats_with_right_shapes():
    sae = TopKSAE(d_model=16, dict_size=64, k=4)
    recon, feats = sae(torch.randn(8, 16))
    assert recon.shape == (8, 16) and feats.shape == (8, 64)

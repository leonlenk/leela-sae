"""Tests for the Top-K SAE, written FIRST (DESIGN.md §3.5a). Run red, implement src/sae.py TODOs, run green."""

import torch

from src.sae import TopKSAE


def test_topk_enforces_l0_equals_k():
    sae = TopKSAE(d_model=16, dict_size=64, k=4)
    feats, _ = sae.encode(torch.randn(8, 16))       # encode -> (feats, pre_acts); (B=8, F=64)
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
    feats, _ = sae.encode(x)                         # (1, 8)
    kept = set((feats[0] != 0).nonzero().flatten().tolist())
    assert kept == {1, 3, 5}


def test_decoder_columns_unit_norm():
    sae = TopKSAE(d_model=16, dict_size=64, k=4)
    norms = sae.decoder.weight.norm(dim=0)          # one norm per dict column
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_forward_returns_recon_and_feats_with_right_shapes():
    sae = TopKSAE(d_model=16, dict_size=64, k=4)
    # forward contract: (recon (B,D), feats post-top-k (B,F), pre pre-top-k (B,F)).
    recon, feats, pre = sae(torch.randn(8, 16))
    assert recon.shape == (8, 16) and feats.shape == (8, 64) and pre.shape == (8, 64)


def test_init_pre_bias_sets_data_mean_and_stays_a_parameter():
    # Contract: init_pre_bias takes FLATTENED activations (N, D) and sets pre_bias to the
    # per-dimension mean over the N tokens -> shape (D,). It must overwrite the VALUES of
    # the existing Parameter (so it still trains / moves with .to(device)), not replace the
    # Parameter object with a plain tensor.
    D = 4
    sae = TopKSAE(d_model=D, dict_size=16, k=2)
    # Build acts whose column means are exactly [10, 20, 30, 40]: each column is that value
    # plus mean-zero noise, so acts.mean(dim=0) recovers the target.
    torch.manual_seed(0)
    target = torch.tensor([10.0, 20.0, 30.0, 40.0])
    noise = torch.randn(5000, D)
    noise -= noise.mean(dim=0)                       # force exact zero-mean noise per column
    acts = target + noise                            # (N=5000, D=4)

    sae.init_pre_bias(acts)

    assert isinstance(sae.pre_bias, torch.nn.Parameter)   # didn't get swapped for a tensor
    assert sae.pre_bias.shape == (D,)                     # one value per d_model dim
    assert torch.allclose(sae.pre_bias, target, atol=1e-4)


def test_encode_is_differentiable_through_kept_entries():
    # The kept (top-k) activations must carry gradient back to the encoder, else training is
    # silently dead. A scatter that detaches its src would pass the shape/L0 tests but fail here.
    sae = TopKSAE(d_model=8, dict_size=32, k=5)
    feats, _ = sae.encode(torch.randn(16, 8))
    feats.sum().backward()
    assert sae.encoder.weight.grad is not None
    assert sae.encoder.weight.grad.abs().sum() > 0

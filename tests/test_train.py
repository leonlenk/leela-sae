"""Training-piece tests, written FIRST (§3.5a). All synthetic — no net needed. Run red, implement
the src/train.py TODOs, run green. These pin the two ideas that are easy to get subtly wrong:
"no L1 penalty" and "aux loss revives dead latents".
"""

import torch

from src import train


def test_reconstruction_loss_is_plain_mse():
    x = torch.randn(8, 16)
    recon = x.clone()
    # Perfect reconstruction -> exactly zero loss (it's MSE, nothing else).
    assert torch.isclose(train.reconstruction_loss(recon, x), torch.tensor(0.0))


def test_reconstruction_loss_is_quadratic_in_error():
    # Confirms it's genuinely MSE (squared), the basis for "no L1": doubling the per-element error
    # quadruples the loss. A linear (L1-style) loss would only double it. err 0.2 vs 0.1 -> 4x.
    x = torch.zeros(4, 16)
    loss_small = train.reconstruction_loss(torch.full((4, 16), 0.1), x)
    loss_big = train.reconstruction_loss(torch.full((4, 16), 0.2), x)
    assert torch.isclose(loss_big, 4.0 * loss_small)


def test_l0_equals_k_for_topk_feats():
    # Build feats with exactly 3 nonzeros per row; L0 must report 3.0.
    feats = torch.zeros(5, 20)
    feats[:, :3] = 1.0
    assert torch.isclose(train.l0_norm(feats), torch.tensor(3.0))


def test_dead_latents_are_the_never_fired_ones():
    counts = torch.zeros(6)
    feats = torch.zeros(4, 6)
    feats[:, 0] = 1.0          # latent 0 fires every row
    feats[2, 3] = 1.0          # latent 3 fires once
    counts = train.update_dead_latents(counts, feats)
    # Hand-checkable: latents 1,2,4,5 never fired -> count 0; latent 0 fired 4x; latent 3 fired 1x.
    assert counts.tolist() == [4.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_aux_loss_is_zero_when_no_dead_latents():
    x = torch.randn(4, 8)
    recon = torch.randn(4, 8)
    feats_pre = torch.randn(4, 12)
    dead_mask = torch.zeros(12, dtype=torch.bool)   # nothing dead
    # No dead latents -> nothing to revive -> the aux term contributes exactly 0.
    assert torch.isclose(train.aux_k_loss(x, recon, feats_pre, dead_mask, aux_k=4), torch.tensor(0.0))

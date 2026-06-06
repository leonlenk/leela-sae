"""Training-piece tests, written FIRST (§3.5a). All synthetic — no net needed. Run red, implement
the src/train.py TODOs, run green. These pin the two ideas that are easy to get subtly wrong:
"no L1 penalty" and "aux loss revives dead latents".
"""

import os

import pytest
import torch

from src import train
from src.sae import TopKSAE


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
    # Now that aux_k_loss decodes through the SAE, it takes the model. The empty-dead path must
    # short-circuit to 0 BEFORE touching the decoder (so an all-False mask -> exactly 0).
    sae = TopKSAE(d_model=8, dict_size=12, k=4)
    x = torch.randn(4, 8)
    recon = torch.randn(4, 8)
    feats_pre = torch.randn(4, 12)
    dead_mask = torch.zeros(12, dtype=torch.bool)   # nothing dead
    # No dead latents -> nothing to revive -> the aux term contributes exactly 0.
    out = train.aux_k_loss(x, recon, feats_pre, dead_mask, aux_k=4, sae=sae)
    assert torch.isclose(torch.as_tensor(out, dtype=torch.float), torch.tensor(0.0))


def test_aux_loss_reconstructs_residual_from_dead_latents_only():
    # The core of the ghost-grads trick, made fully predictable. We pin the decoder to the identity
    # so feature-space code passes straight through to residual space, and we set a NONZERO pre_bias
    # to prove the aux reconstruction must NOT add it (the residual x-recon already has bias removed).
    sae = TopKSAE(d_model=4, dict_size=4, k=2)
    with torch.no_grad():
        sae.decoder.weight.copy_(torch.eye(4))                  # decode(code) == code
        sae.pre_bias.copy_(torch.tensor([10., 20., 30., 40.]))  # must be ignored by aux
    # (B=1, F=4) pre-activations. Latent 0 is alive; {1,2,3} are dead with values 5, 3, 1.
    feats_pre = torch.tensor([[100., 5., 3., 1.]])
    dead_mask = torch.tensor([False, True, True, True])
    # Among the DEAD columns, top-aux_k=2 are 5 (col 1) and 3 (col 2). Scattered back into a full
    # (1,4) buffer -> [0,5,3,0]; identity decoder, no pre_bias -> reconstruction == [0,5,3,0].
    recon = torch.zeros(1, 4)
    x = torch.tensor([[0., 4., 3., 0.]])                        # residual target (x-recon) = [0,4,3,0]
    # reconstruction [0,5,3,0] vs target [0,4,3,0] -> diff [0,1,0,0] -> MSE over 4 elems = 0.25.
    out = train.aux_k_loss(x, recon, feats_pre, dead_mask, aux_k=2, sae=sae)
    assert torch.isclose(out, torch.tensor(0.25))


def test_aux_loss_routes_gradient_into_dead_latents_only():
    # The WHOLE point: aux grad must reach the dead latents (so they can revive) and must leave the
    # live latents untouched (we don't want to perturb what already works).
    torch.manual_seed(0)
    sae = TopKSAE(d_model=4, dict_size=4, k=2)
    feats_pre = torch.randn(3, 4, requires_grad=True)
    dead_mask = torch.tensor([False, False, True, True])        # cols 2,3 dead
    x, recon = torch.randn(3, 4), torch.randn(3, 4)
    loss = train.aux_k_loss(x, recon, feats_pre, dead_mask, aux_k=1, sae=sae)
    loss.backward()
    g = feats_pre.grad
    assert g[:, ~dead_mask].abs().sum() == 0     # live latents: exactly zero gradient
    assert g[:, dead_mask].abs().sum() > 0       # dead latents: gradient actually flows


def test_train_runs_configured_number_of_epochs(monkeypatch):
    # The ONE thing easy to get wrong about epochs: it means E *full passes* over the loader, so the
    # per-batch step body must execute exactly E * (batches_per_epoch) times — not E total steps, not
    # 1 pass. We count optimizer steps indirectly by counting reconstruction_loss calls (it runs once
    # per step, line 145). dead_window is set huge so dead_mask stays all-False and the aux path
    # short-circuits to 0 WITHOUT calling reconstruction_loss again — keeping the count exactly 1/step.
    torch.manual_seed(0)
    D, F_dim, k = 8, 16, 3
    sae = TopKSAE(D, F_dim, k)
    acts = torch.randn(20, D)                       # 20 rows, batch 5, drop_last -> 4 batches/epoch
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(acts), batch_size=5, drop_last=True
    )
    batches_per_epoch = len(loader)                 # 4

    calls = {"n": 0}
    real_recon_loss = train.reconstruction_loss
    def counting(recon, x):                         # spy that delegates to the real MSE
        calls["n"] += 1
        return real_recon_loss(recon, x)
    monkeypatch.setattr(train, "reconstruction_loss", counting)

    epochs = 3
    cfg = dict(
        lr=1e-3, sae_hidden=F_dim, aux_coef=0.0, aux_k=4,
        dead_window=10**9,        # never recompute dead_mask -> aux path never calls recon_loss
        train_eval_freq=10**9,    # don't log/eval (and base_model=None below would crash if we did)
        epochs=epochs,
    )
    # No net needed: eval_fens=None / base_model=None means the loss_recovered branch is skipped.
    # train() now takes F_dim explicitly (it sizes the fired_counts / dead_mask vectors); pass the
    # same dict width the SAE was built with so the dead-latent bookkeeping matches the dictionary.
    train.train(cfg, sae, base_model=None, loader=loader, F_dim=F_dim)

    # 3 epochs x 4 batches == 12 step bodies. Current (epoch-less) loop runs only 4 -> this fails RED
    # until you wrap the inner loop in `for epoch in range(config["epochs"])`.
    assert calls["n"] == epochs * batches_per_epoch


# --- loss_recovered: needs the LIVE Leela net, so skip-gate on env vars like the eval tests. ---
_NET = os.environ.get("LEELA_SAE_CHECKPOINT")
_MODULE = os.environ.get("LEELA_SAE_RESID_MODULE")

requires_net = pytest.mark.skipif(
    not (_NET and _MODULE),
    reason="set LEELA_SAE_CHECKPOINT and LEELA_SAE_RESID_MODULE (live net) to run loss_recovered",
)


class _FakeSAE:
    """Stand-in for a trained SAE so we can pin loss_recovered's anchors without a real dictionary.
    forward(x) -> (recon, None, None), matching the (recon, feats, pre) contract loss_recovered uses."""
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x), None, None

    def eval(self):        # train()/script call .eval(); no-op here
        return self


@pytest.fixture(scope="session")
def base_model():
    from src import hook
    return hook.load_model(_NET)


@requires_net
def test_loss_recovered_anchor_invariants(base_model):
    # INTERP-SAFETY (§3.5 carve-out — verify these two by hand before trusting any recovered number):
    # if the patch harness is wrong, every loss_recovered it reports is noise.
    import chess
    fen = chess.STARTING_FEN

    # Identity SAE: recon == the true activation -> patching rewrites the site to itself -> the heads
    # don't move -> loss_recon == loss_clean == 0 -> recovered == 1.0.
    identity = _FakeSAE(lambda a: a)
    rec_id = train.loss_recovered(base_model, fen, identity, _MODULE)
    assert torch.isclose(torch.as_tensor(rec_id, dtype=torch.float), torch.tensor(1.0), atol=1e-3)

    # Zeroing SAE: recon == zeros == the zero-ablation baseline -> loss_recon == loss_zero ->
    # recovered == 0.0.
    zeroing = _FakeSAE(lambda a: torch.zeros_like(a))
    rec_zero = train.loss_recovered(base_model, fen, zeroing, _MODULE)
    assert abs(float(rec_zero)) < 1e-3

"""Training-piece tests. Synthetic, no net, except the fidelity anchors at the bottom.

They pin the ideas that are easy to get subtly wrong: no L1 term, the AuxK loss touches ONLY dead
latents (and uses their positive activations), dead-latent bookkeeping, and the LR schedule.
"""

import torch

from conftest import requires_net
from src import train
from src.sae import TopKSAE


def test_reconstruction_loss_is_plain_mse():
    x = torch.randn(8, 16)
    assert torch.isclose(train.reconstruction_loss(x.clone(), x), torch.tensor(0.0))
    # Quadratic, not linear: doubling the error quadruples the loss.
    zero = torch.zeros(4, 16)
    assert torch.isclose(train.reconstruction_loss(torch.full((4, 16), 0.2), zero),
                         4 * train.reconstruction_loss(torch.full((4, 16), 0.1), zero))


def test_normalized_mse_is_one_for_predicting_the_mean():
    torch.manual_seed(0)
    x = torch.randn(1000, 8) * 3 + 5
    assert torch.isclose(train.normalized_mse(x.mean(0, keepdim=True).expand_as(x), x), torch.tensor(1.0))
    assert train.normalized_mse(x, x) == 0


def test_l0_counts_nonzeros_per_row():
    feats = torch.zeros(5, 20)
    feats[:, :3] = 1.0
    assert torch.isclose(train.l0_norm(feats), torch.tensor(3.0))


def test_tokens_since_fired_resets_only_fired_latents():
    counter = torch.tensor([10.0, 10.0, 10.0])
    feats = torch.zeros(4, 3)
    feats[2, 1] = 0.5                                       # latent 1 fired once in this batch of 4
    out = train.update_tokens_since_fired(counter, feats)
    assert out.tolist() == [14.0, 0.0, 14.0]


def test_aux_loss_is_zero_with_no_dead_latents():
    sae = TopKSAE(8, 12, 4)
    out = train.aux_k_loss(torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 12),
                           torch.zeros(12, dtype=torch.bool), aux_k=4, sae=sae)
    assert float(out) == 0.0


def test_aux_loss_hand_computed_with_identity_decoder():
    # Identity decoder: code passes straight to residual space. pre_bias is NONZERO to prove aux
    # reconstruction does not add it (the residual x - recon already has the bias removed).
    sae = TopKSAE(d_model=4, dict_size=4, k=2)
    with torch.no_grad():
        sae.decoder.weight.copy_(torch.eye(4))
        sae.pre_bias.copy_(torch.tensor([10., 20., 30., 40.]))
    # Latent 0 alive; 1,2,3 dead with pre-acts 5, 3, -1. top-2 dead (after ReLU) -> cols 1,2 -> [0,5,3,0].
    pre = torch.tensor([[100., 5., 3., -1.], [100., 1., 2., 3.]])
    dead = torch.tensor([False, True, True, True])
    x = torch.tensor([[0., 4., 3., 0.], [0., 1., 2., 0.]])
    recon = torch.zeros(2, 4)
    # Row 0 recon [0,5,3,0] vs target [0,4,3,0]; row 1 top-2 dead = cols 3,2 -> [0,0,2,3] vs [0,1,2,0].
    # Squared errors: row 0 -> 1; row 1 -> 1 + 9 = 10. MSE = 11 / 8.
    # Target variance about its column means: mean = [0, 2.5, 2.5, 0]; squared devs = 2.25*2 + 0.25*2 = 5 -> /8.
    out = train.aux_k_loss(x, recon, pre, dead, aux_k=2, sae=sae)
    assert torch.isclose(out, torch.tensor((11 / 8) / (5 / 8)))


def test_aux_loss_ignores_negative_dead_preactivations():
    # A dead latent with a NEGATIVE pre-activation must contribute nothing (ReLU before top-k).
    sae = TopKSAE(d_model=2, dict_size=2, k=1)
    with torch.no_grad():
        sae.decoder.weight.copy_(torch.eye(2))
    pre = torch.tensor([[1.0, -5.0], [1.0, -5.0]])
    dead = torch.tensor([False, True])
    x = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    out = train.aux_k_loss(x, torch.zeros(2, 2), pre, dead, aux_k=1, sae=sae)
    # aux_recon is all zeros -> normalised MSE of predicting 0 for a zero-mean target == 1.
    assert torch.isclose(out, torch.tensor(1.0))


def test_aux_loss_routes_gradient_into_dead_latents_only():
    torch.manual_seed(0)
    sae = TopKSAE(d_model=4, dict_size=4, k=2)
    pre = torch.rand(3, 4).add(0.1).requires_grad_(True)      # all positive so dead ones get picked
    dead = torch.tensor([False, False, True, True])
    loss = train.aux_k_loss(torch.randn(3, 4), torch.randn(3, 4), pre, dead, aux_k=1, sae=sae)
    loss.backward()
    assert pre.grad[:, ~dead].abs().sum() == 0 and pre.grad[:, dead].abs().sum() > 0


def test_lr_schedule_warms_up_holds_and_decays_to_zero():
    f = train._lr_lambda(total_steps=100, warmup=10, decay_frac=0.2)
    assert f(0) == 0.1 and f(9) == 1.0 and f(50) == 1.0
    assert f(80) == 1.0 and abs(f(90) - 0.5) < 1e-9 and f(100) == 0.0


def test_train_runs_epochs_keeps_unit_decoder_and_reduces_fvu():
    torch.manual_seed(0)
    # Data that a k=2 dictionary CAN explain: each row is a sparse combination of 8 true directions.
    true = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    codes = torch.zeros(4096, 8)
    for r in range(4096):
        codes[r, torch.randperm(8)[:2]] = torch.rand(2) + 0.5
    acts = codes @ true + 0.01 * torch.randn(4096, 16)
    sae = TopKSAE(16, 32, 2)
    sae.init_pre_bias(acts)
    before = train.fvu(sae, acts)
    sae, hist = train.train(sae, acts, {"batch_size": 256, "epochs": 20, "lr": 1e-2, "warmup_steps": 5,
                                        "log_every": 1, "dead_tokens": 10**9}, device="cpu")
    assert len(hist) == 20 * (4096 // 256)                    # every step logged: epochs x batches
    assert before > 0.4 and train.fvu(sae, acts) < 0.05       # learns the planted sparse structure
    norms = sae.decoder.weight.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------- network fidelity (real net)

class _FakeSAE(torch.nn.Module):
    """forward(x) -> (fn(x), feats, pre) so fidelity() can be pinned at its two anchors."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x), torch.zeros(x.shape[0], 1), None


@requires_net
def test_fidelity_anchors(model, resid_module):
    # INTERP-SAFETY anchors (verify by hand): an identity "SAE" must give KL 0, top-1 agreement 1 and
    # loss recovered 1; replacing the activation by the per-square mean must give recovered 0 vs mean.
    import chess
    from src import hook
    from src.eval.fidelity import fidelity
    fens = [chess.STARTING_FEN, "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5"]
    mean_act = hook.get_residual_stream_batch(model, fens, resid_module).mean(0)    # (64, D)
    ident = fidelity(model, _FakeSAE(lambda x: x), resid_module, fens, mean_act)
    assert ident["kl_recon_mean"] < 1e-6 and ident["top1_agree_recon"] == 1.0 and ident["fvu"] < 1e-9
    assert abs(ident["loss_recovered_vs_zero"] - 1) < 1e-4 and abs(ident["loss_recovered_vs_mean"] - 1) < 1e-4
    as_mean = fidelity(model, _FakeSAE(lambda x: mean_act.reshape(-1, 64, x.shape[-1]).expand(
        x.shape[0] // 64, 64, x.shape[-1]).reshape(x.shape)), resid_module, fens, mean_act)
    assert abs(as_mean["loss_recovered_vs_mean"]) < 1e-4

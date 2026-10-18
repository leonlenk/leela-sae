"""Causal-harness tests. The harness invariants hold for ANY SAE weights, so they use a fresh random
TopKSAE. INTERP-SAFETY: a harness that reports an effect when nothing was removed (or none when
something was) fakes every causal result downstream — these are the checks to verify by hand.
"""

import chess
import torch

from conftest import requires_net
from src.eval import causal
from src.eval.codes import SparseCodes
from src.sae import TopKSAE


def test_ablation_subtracts_exactly_the_feature_contribution():
    # Error-preserving ablation: x -> x - f_i(x) * d_i. Everything else (other features AND the SAE
    # reconstruction error) must be untouched, so (x - ablated) is exactly f_i(x) * d_i.
    torch.manual_seed(0)
    sae = TopKSAE(d_model=8, dict_size=32, k=4)
    act = torch.randn(2, 64, 8)
    feats = sae.encode(act.reshape(-1, 8))[0].reshape(2, 64, 32)
    f = torch.tensor([int(feats[0].sum(0).argmax()), int(feats[1].sum(0).argmax())])   # active ones
    out = causal._ablation_fn(sae, f)(act)
    for b in range(2):
        expected = feats[b, :, f[b], None] * sae.decoder.weight[:, f[b]][None, :]
        assert torch.allclose(act[b] - out[b], expected, atol=1e-6)
        assert (act[b] - out[b]).abs().sum() > 0          # an active feature really was removed


def test_ablating_an_inactive_feature_is_an_exact_noop():
    torch.manual_seed(0)
    sae = TopKSAE(d_model=8, dict_size=32, k=4)
    act = torch.randn(1, 64, 8)
    feats = sae.encode(act.reshape(-1, 8))[0]
    dead = int(torch.nonzero(feats.sum(0) == 0).flatten()[0])
    assert torch.equal(causal._ablation_fn(sae, torch.tensor([dead]))(act), act)


def test_matched_control_picks_closest_activation_mass_and_never_the_feature():
    # One board, k=2. Feature 0 mass 5; feature 1 mass 4.5; feature 2 mass 1; feature 3 inactive.
    idx = torch.zeros(1, 64, 2, dtype=torch.long)
    val = torch.zeros(1, 64, 2)
    idx[0, 0] = torch.tensor([0, 1]); val[0, 0] = torch.tensor([5.0, 4.5])
    idx[0, 1] = torch.tensor([2, 3]); val[0, 1] = torch.tensor([1.0, 0.0])
    codes = SparseCodes(idx, val, 4)
    assert causal.matched_control(codes, 0, 0, torch.Generator().manual_seed(0)) == 1
    assert causal.matched_control(codes, 0, 2, torch.Generator().manual_seed(0)) == 1   # |4.5-1| < |5-1|


@requires_net
def test_inactive_feature_has_zero_effect_and_active_one_has_some(model, resid_module):
    from src import hook
    fens = [chess.STARTING_FEN, "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5"]
    resid = hook.get_residual_stream_batch(model, fens, resid_module)          # (2, 64, 768)
    sae = TopKSAE(768, 4 * 768, 32).to(resid.device).eval()
    feats = sae.encode(resid.reshape(-1, 768))[0].reshape(2, 64, -1)
    totals = feats.sum(1)                                                      # (2, F)
    inactive = torch.tensor([int(torch.nonzero(totals[b] == 0).flatten()[0]) for b in range(2)])
    strongest = totals.argmax(-1)
    off = causal.ablation_effects(model, sae, resid_module, fens, inactive)
    assert off["kl"].abs().max() == 0 and off["dq"].abs().max() == 0          # nothing removed
    on = causal.ablation_effects(model, sae, resid_module, fens, strongest)
    assert on["kl"].min() > 0                                                  # power: effect reaches heads


def test_steering_adds_exactly_alpha_times_direction_at_one_token():
    torch.manual_seed(0)
    sae = TopKSAE(d_model=8, dict_size=16, k=3)
    act = torch.randn(2, 64, 8)
    out = causal._steer_fn(sae, torch.tensor([3, 5]), torch.tensor([10, 0]), torch.tensor([2.0, -1.0]))(act)
    delta = out - act
    assert torch.allclose(delta[0, 10], 2.0 * sae.decoder.weight[:, 3]) and torch.allclose(delta[1, 0], -sae.decoder.weight[:, 5])
    delta[0, 10] = 0
    delta[1, 0] = 0
    assert delta.abs().max() == 0                         # nothing else touched

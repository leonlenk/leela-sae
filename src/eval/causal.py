"""Causal eval: remove one SAE feature from the residual stream and measure what Leela's heads do.

Concept — ERROR-PRESERVING ABLATION. The SAE writes the activation as x = Σ_i f_i(x)·d_i + ε
(d_i = unit decoder column, ε = what the SAE fails to reconstruct). To remove feature i we patch
    x  ->  x − f_i(x)·d_i
into the TRUE activation, keeping ε and every other feature. The clean run is the unpatched net,
so the measured effect is purely "what this one direction contributed downstream". (The original
harness instead compared two SAE round-trips, which drops ε from both runs; splicing the SAE in
already moves the policy by more than most single features do, so that baseline was noisy.)

Effects, all on LEGAL moves only (illegal policy slots are untrained junk — see src/hook.py):
  * kl         — KL(clean policy || ablated policy), nats
  * dq         — change in the value Q = P(win) − P(loss), side-to-move view
  * top1_flip  — did the policy's top legal move change?
  * target_dlogp — change in log P(moves whose from- OR to-square is a square where the candidate
                   feature fires). A feature that "means" something about a square should, when
                   removed, specifically shift probability mass on moves through that square.

Experimental design (the audit found the old notebook's "random control" was inactive on the board,
so its zero effect was a tautology): for each candidate feature, sample boards where it is ACTIVE;
on each board pick a CONTROL feature that is also active there with the closest total activation;
ablate each separately and compare paired effects. Report medians and the fraction of boards on
which the candidate beats its control.
"""

from __future__ import annotations

import chess
import torch

from src import hook
from src.eval.codes import SparseCodes
from src.viz.features import token_to_square


def _ablation_fn(sae, features: torch.Tensor):
    """Patch fn removing feature features[b] from board b. act: (B, 64, D) -> (B, 64, D)."""
    W = sae.decoder.weight                                       # (D, F), unit-norm columns

    def fn(act):
        B, T, D = act.shape
        feats = sae.encode(act.reshape(-1, D))[0].reshape(B, T, -1)          # (B, 64, F) post-top-k
        f = features.to(act.device)
        f_vals = feats.gather(-1, f[:, None, None].expand(B, T, 1))          # (B, 64, 1)
        dirs = W[:, f].T                                                     # (B, D)
        return act - f_vals * dirs[:, None, :]                               # error-preserving
    return fn


def _q(wdl: torch.Tensor) -> torch.Tensor:
    return wdl[..., 0] - wdl[..., 2]                                         # P(win) − P(loss)


def _target_moves(fen: str, tokens: list[int]) -> list[int]:
    """Indices (into the board's legal-move list) of moves touching any of the given tokens' squares."""
    board = chess.Board(fen)
    squares = {token_to_square(fen, t) for t in tokens}
    return [i for i, m in enumerate(board.legal_moves) if m.from_square in squares or m.to_square in squares]


@torch.no_grad()
def ablation_effects(model, sae, module: str, fens: list[str], features: torch.Tensor,
                     target_tokens: list[list[int]] | None = None, batch_size: int = 128) -> dict[str, torch.Tensor]:
    """Ablate features[b] on board fens[b]. Returns per-board effects (B,): kl, dq, top1_flip and,
    if target_tokens is given, target_dlogp (nan when no legal move touches the target squares)."""
    sae = sae.to(hook.model_device(model)).eval()
    out = {k: [] for k in ("kl", "dq", "top1_flip", "target_dlogp")}
    for s in range(0, len(fens), batch_size):
        chunk, feats = fens[s:s + batch_size], features[s:s + batch_size]
        x = hook.encode_fens(chunk)
        clean = model(x.to(hook.model_device(model)))
        patched = hook.run_patched(model, x, module, _ablation_fn(sae, feats))
        for b, fen in enumerate(chunk):
            _, idx = hook.legal_moves_and_indices(chess.Board(fen))
            lp = hook.legal_log_probs(clean["policy"][b], idx)
            lq = hook.legal_log_probs(patched["policy"][b], idx)
            out["kl"].append(float((lp.exp() * (lp - lq)).sum()))
            out["dq"].append(float(_q(patched["wdl"][b]) - _q(clean["wdl"][b])))
            out["top1_flip"].append(float(lp.argmax() != lq.argmax()))
            if target_tokens is not None:
                tm = _target_moves(fen, target_tokens[s + b])
                if tm:
                    t = torch.tensor(tm, device=lp.device)
                    out["target_dlogp"].append(float(torch.logsumexp(lq[t], 0) - torch.logsumexp(lp[t], 0)))
                else:
                    out["target_dlogp"].append(float("nan"))
    return {k: torch.tensor(v) for k, v in out.items() if v}


def matched_control(codes: SparseCodes, board: int, feature: int, rng: torch.Generator) -> int | None:
    """A different feature active on `board` whose total activation (sum over tokens) is closest to
    `feature`'s. Ties broken at random. None if no other feature is active."""
    idx, val = codes.idx[board].reshape(-1), codes.val[board].reshape(-1)
    mass = torch.zeros(codes.n_features).index_add_(0, idx, val)            # (F,) total activation
    target = mass[feature]
    cand = torch.nonzero((mass > 0) & (torch.arange(codes.n_features) != feature)).flatten()
    if len(cand) == 0:
        return None
    gap = (mass[cand] - target).abs()
    best = cand[gap == gap.min()]
    return int(best[torch.randint(len(best), (1,), generator=rng)])


def feature_vs_control(model, sae, module: str, codes: SparseCodes, fens: list[str], feature: int,
                       n_boards: int = 64, seed: int = 0) -> dict:
    """Paired causal test of one feature against activation-matched controls on boards where it fires."""
    rng = torch.Generator().manual_seed(seed)
    act = codes.feature(feature)                                             # (N, 64)
    active = torch.nonzero(act.amax(1) > 0).flatten()
    if len(active) == 0:
        return {"feature": feature, "n_boards": 0}
    boards = active[torch.randperm(len(active), generator=rng)[:n_boards]].tolist()
    controls = [matched_control(codes, b, feature, rng) for b in boards]
    keep = [(b, c) for b, c in zip(boards, controls) if c is not None]
    boards, controls = [b for b, _ in keep], [c for _, c in keep]
    sub = [fens[b] for b in boards]
    tokens = [torch.nonzero(act[b] > 0).flatten().tolist() for b in boards]  # where the CANDIDATE fires

    eff_f = ablation_effects(model, sae, module, sub, torch.tensor([feature] * len(sub)), tokens)
    eff_c = ablation_effects(model, sae, module, sub, torch.tensor(controls), tokens)
    ok = ~torch.isnan(eff_f["target_dlogp"]) & ~torch.isnan(eff_c["target_dlogp"])
    return {
        "feature": feature,
        "n_boards": len(sub),
        "kl_median": float(eff_f["kl"].median()),
        "control_kl_median": float(eff_c["kl"].median()),
        "kl_win_rate": float((eff_f["kl"] > eff_c["kl"]).float().mean()),
        "abs_dq_median": float(eff_f["dq"].abs().median()),
        "control_abs_dq_median": float(eff_c["dq"].abs().median()),
        "top1_flip_rate": float(eff_f["top1_flip"].mean()),
        "control_top1_flip_rate": float(eff_c["top1_flip"].mean()),
        # Targeted: signed medians (negative = ablating suppresses moves through the feature's
        # squares), and how often the candidate shifts that move mass MORE (in magnitude) than its
        # matched control does. Magnitude, because a feature may encourage OR discourage those moves.
        "target_dlogp_median": float(eff_f["target_dlogp"][ok].median()) if ok.any() else float("nan"),
        "control_target_dlogp_median": float(eff_c["target_dlogp"][ok].median()) if ok.any() else float("nan"),
        "target_win_rate": float((eff_f["target_dlogp"][ok].abs() > eff_c["target_dlogp"][ok].abs()).float().mean()) if ok.any() else float("nan"),
        "raw": {"feature": eff_f, "control": eff_c, "boards": boards, "controls": controls},
    }


# --------------------------------------------------------------------------------------------------
# Steering (sufficiency): ADD a feature where it is off and check the policy moves as its meaning
# predicts. Ablation asks "does the net need this direction?"; at a middle layer the answer can be
# "no" simply because later layers recompute the fact from the intact board. Steering asks "does
# writing this direction change what the net does, in the predicted way?"
# --------------------------------------------------------------------------------------------------

def _steer_fn(sae, features: torch.Tensor, tokens: torch.Tensor, alphas: torch.Tensor,
              directions: torch.Tensor | None = None):
    """Add alphas[b] * d_b at token tokens[b] of board b, where d_b is feature features[b]'s unit
    decoder column — or directions[b] (B, D, unit-norm) if given. act: (B, 64, D)."""
    W = sae.decoder.weight

    def fn(act):
        out = act.clone()
        b = torch.arange(act.shape[0], device=act.device)
        d = directions.to(act.device, act.dtype) if directions is not None else W[:, features.to(act.device)].T
        out[b, tokens.to(act.device)] += alphas.to(act.device, act.dtype)[:, None] * d
        return out
    return fn


@torch.no_grad()
def steering_effects(model, sae, module: str, fens: list[str], features: torch.Tensor, tokens: torch.Tensor,
                     alphas: torch.Tensor, target_moves: list[list[int]], batch_size: int = 128,
                     directions: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Per board: change in log P(target moves) (indices into the board's legal-move list) and KL,
    after adding the feature direction at one token."""
    sae = sae.to(hook.model_device(model)).eval()
    dlogp, kl = [], []
    for s in range(0, len(fens), batch_size):
        sl = slice(s, s + batch_size)
        chunk = fens[sl]
        x = hook.encode_fens(chunk)
        clean = model(x.to(hook.model_device(model)))
        dirs = directions[sl] if directions is not None else None
        steered = hook.run_patched(model, x, module, _steer_fn(sae, features[sl], tokens[sl], alphas[sl], dirs))
        for b, fen in enumerate(chunk):
            _, idx = hook.legal_moves_and_indices(chess.Board(fen))
            lp = hook.legal_log_probs(clean["policy"][b], idx)
            lq = hook.legal_log_probs(steered["policy"][b], idx)
            t = torch.tensor(target_moves[s + b], device=lp.device)
            dlogp.append(float(torch.logsumexp(lq[t], 0) - torch.logsumexp(lp[t], 0)))
            kl.append(float((lp.exp() * (lp - lq)).sum()))
    return {"target_dlogp": torch.tensor(dlogp), "kl": torch.tensor(kl)}


def active_quantile(codes: SparseCodes, feature: int, q: float = 0.9) -> float:
    """q-quantile of a feature's activation over the tokens where it is active."""
    v = codes.val[codes.idx == feature]
    v = v[v > 0]
    return float(v.quantile(q)) if v.numel() else 0.0

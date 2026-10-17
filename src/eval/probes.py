"""Supervised linear-probe baselines for the correlational eval.

A single SAE feature matching a property only means something relative to two references:
  * a probe on the SAME residual layer — the property is linearly decodable there at all (a soft
    upper bound for any single direction, including an SAE feature);
  * a probe on the token's own RAW INPUT planes (112 numbers per square) — the property is already
    locally present in the input, so the net didn't have to compute it.
Probes are logistic regressions fit on the selection split, with the decision threshold chosen on
that split to maximise F1, then scored on the disjoint test split (same protocol as the SAE).
"""

from __future__ import annotations

import torch

from src.eval.correlational import auroc, f1_from_counts


def fit_probe(X_S: torch.Tensor, y_S: torch.Tensor, X_T: torch.Tensor, y_T: torch.Tensor,
              steps: int = 300, lr: float = 0.05, weight_decay: float = 1e-4, device: str | None = None) -> dict:
    """Logistic-regression probe. X: (M, d) features, y: (M,) bool. Returns test F1 / AUROC."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_S, X_T = X_S.to(device, torch.float32), X_T.to(device, torch.float32)
    y_S, y_T = y_S.reshape(-1).to(device).float(), y_T.reshape(-1).to(device).bool()
    mu, sd = X_S.mean(0), X_S.std(0).clamp_min(1e-6)            # standardise with S statistics only
    X_S, X_T = (X_S - mu) / sd, (X_T - mu) / sd

    w = torch.zeros(X_S.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    pos = y_S.mean().clamp(1e-6, 1 - 1e-6)
    pos_weight = (1 - pos) / pos                                 # balance rare properties
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    for _ in range(steps):                                       # full-batch: M is a few 100k rows
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X_S @ w + b, y_S, pos_weight=pos_weight)
        loss.backward()
        opt.step()

    with torch.no_grad():
        s_S, s_T = X_S @ w + b, X_T @ w + b
        # Threshold that maximises F1 on S (searched over S-score quantiles), applied unchanged to T.
        qs = torch.quantile(s_S[torch.randperm(len(s_S), device=device)[:200_000]],
                            torch.linspace(0.0, 1.0, 201, device=device))
        best_t, best_f1 = 0.0, -1.0
        yS = y_S.bool()
        for t in qs:
            on = s_S > t
            f1 = float(f1_from_counts((on & yS).sum().cpu()[None], on.sum().cpu()[None], yS.sum().cpu())[2])
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        on_T = s_T > best_t
        p, r, f1 = (float(x) for x in f1_from_counts((on_T & y_T).sum().cpu()[None], on_T.sum().cpu()[None], y_T.sum().cpu()))
    return {"test_f1": f1, "test_precision": p, "test_recall": r, "test_auroc": auroc(s_T.cpu(), y_T.cpu())}

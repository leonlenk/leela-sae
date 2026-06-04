"""Step 2 — train the hand-written Top-K SAE on a small cached activation sample (🔒 — scaffold per §3.5).

Single GPU, NO Ray, NO Accelerate (that's step 3). The goal is intuition, not scale: cache a modest
pile of activations, train, and watch ONE feature until you believe it (§4). Logic lives in
src/sae.py + src/train.py and is covered by tests/test_sae.py + tests/test_train.py — implement those
green first, then this script orchestrates a real run.
"""

from src import hook, train
from src.sae import TopKSAE
from src.data import cache, positions
from src.eval import feature_mining
from src.viz import features

import yaml
from pathlib import Path
import torch

def main():
    with open(Path("configs/default.yaml")) as f:
            cfg = yaml.safe_load(f) 

    base_model = hook.load_model().eval()
    module = hook.list_residual_modules(base_model)[cfg["residual_modual"]]

    capture = lambda fens: hook.get_residual_stream_batch(base_model, fens, module)
    fens = positions.load_fens(limit=20_000, shuffle=True, seed=cfg["seed"])
    split = int(0.9 * len(fens))
    train_fens, eval_fens = fens[:split], fens[split:]

    with torch.no_grad():
        amura_v_calderin = "8/1p6/p2R1b2/P1B2k1p/1P3p1P/2r2b1K/5N2/8 w - - 0 1"
        resid = hook.get_residual_stream(base_model, amura_v_calderin, module)
        d_model = resid.shape[1]
        cache_path = cache.cache_activations(
             train_fens, capture,                          # cache ONLY the training FENs
             Path(cfg["output_path"], "data.npy"),
             d_model,
             batch_size=cfg["base_model_batch"],
             signature=f"leela@{cfg['residual_modual']}"
            )

    acts = cache.load_activations(cache_path)                       # (N*64, D) tensor
    dataset = torch.utils.data.TensorDataset(acts)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["sae_batch"], shuffle=True, drop_last=True,
    )
    F_dim = cfg["sae_hidden_factor"] * d_model
    sae = TopKSAE(d_model, F_dim, cfg["top_k"])
    sae.init_pre_bias(acts)                                # centre on the data mean over all rows (N*64, D)
    sae = train.train(cfg, sae, base_model, loader, F_dim, eval_fens=eval_fens, module_name=module)
    torch.save(sae, Path(cfg["output_path"], "sae.pt"))

    # Final held-out score on the FULL eval set (the in-loop logging only sampled a slice).
    sae.eval()
    recovered = train.mean_loss_recovered(base_model, eval_fens, sae, module)
    print(f"held-out loss_recovered: {recovered:.4f}")


if __name__ == "__main__":
    main()

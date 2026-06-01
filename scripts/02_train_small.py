"""Step 2 — train the hand-written Top-K SAE on a small cached activation sample (🔒 — scaffold per §3.5).

Single GPU, NO Ray, NO Accelerate (that's step 3). The goal is intuition, not scale: cache a modest
pile of activations, train, and watch ONE feature until you believe it (§4). Logic lives in
src/sae.py + src/train.py and is covered by tests/test_sae.py + tests/test_train.py — implement those
green first, then this script orchestrates a real run.
"""

from src import hook, train
from src.sae import TopKSAE


def main():
    # TODO (1): load the net + chosen residual module confirmed in script 01 (read configs/default.yaml).
    # TODO (2): cache activations: run hook.get_residual_stream over a few thousand positions and stack
    #           them into one (N, 64, d_model) tensor on disk. Flatten to (N*64, d_model) for the SAE.
    # TODO (3): build TopKSAE(d_model, dict_size = expansion * d_model, k) and init pre_bias to the
    #           data mean of a warmup batch (src/sae.py TODO 1).
    # TODO (4): call train.train(config) on the cached activations (single GPU).
    # TODO (5): after training, pick ONE feature, find its top-activating positions, and eyeball them —
    #           does it correspond to something chess-meaningful? This is the "believe it" check.
    # TODO (6): save the trained SAE (the canary + eval tests load it via LEELA_SAE_SAE_CKPT).
    raise NotImplementedError


if __name__ == "__main__":
    main()

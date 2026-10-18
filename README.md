# leela-sae

Sparse autoencoders on the residual stream of a Leela Chess Zero transformer
([`BT3-768x15x24h-swa-2790000`](https://huggingface.co/lczerolens/BT3-768x15x24h-swa-2790000)),
with every learned feature checked against chess ground truth three ways: does it match a
board property on held-out positions, does it causally drive Leela's policy/value heads, and does it
survive being fed real move history.

**Results and conclusions: [RESULTS.md](RESULTS.md).** Original plan and design rationale:
[DESIGN.md](DESIGN.md).

## Setup

```bash
uv sync                      # Python 3.12, torch with CUDA; ~6 GB of wheels
uv run pytest                # offline tests (~5 s); net-dependent tests skip
LEELA_SAE_CHECKPOINT=lczerolens/BT3-768x15x24h-swa-2790000 uv run pytest   # + real-net tests
```

The net is downloaded from the Hugging Face Hub as ONNX and converted to torch by `lczerolens`.
Everything below ran on one 8 GB laptop GPU (RTX 5050); times are for that machine.

## Pipeline

| step | script | what it does | time |
|---|---|---|---|
| 0 | `scripts/00_build_corpus.py` | Sample 316k positions (with move history) from Lichess games, split by game | 4 min |
| 1 | `scripts/01_reproduce_move.py` | Gate: policy head vs 1000 Lichess puzzles, castling, promotions | 1 min |
| 2 | `scripts/02_cache_activations.py --layers 3 7 11` | Cache residual streams for 200k train boards (fp16, ~20 GB/layer) | 17 min |
| 3 | `scripts/03_train_sae.py --layer 7` | Train a Top-K SAE (k=32, 6144 features) | 6 min |
| 4 | `scripts/04_evaluate.py --sae outputs/sae/L7.pt` | Fidelity, held-out correlational eval, matched-control ablations, history filter | 25 min |
| 4b | `scripts/04b_probes.py` | Linear-probe baselines per layer | 5 min |
| 5 | `scripts/05_data_comparison.py` | Original HF corpus vs Lichess at the same budget | 10 min |
| 6 | `scripts/06_fidelity_table.py outputs/sae/*.pt` | One fidelity table for all SAEs | 10 min |
| 7 | `scripts/07_autointerp.py --sae outputs/sae/L7.pt` | LLM feature labels + held-out detection scores (uses the `claude` CLI) | 15 min |
| 8 | `scripts/08_steering.py --sae outputs/sae/L7.pt` | Write features where they're off; does the policy move as predicted? | 5 min |
| 9 | `scripts/09_history_control.py` | History-filter positive control; first cache + train a real-history SAE (`02 --real-history`, `03 --real-history --tag realhist`, see its docstring) | 25 min |
| 10 | `scripts/10_figures.py [--galleries]` | Figures in `docs/figures/` and per-property verdict tables | 1 min |
| 11 | `scripts/11_feature_splitting.py --sae outputs/sae/L7.pt` | How much of each concept the top-n features recover vs a probe | 3 min |
| 12 | `scripts/12_joint_ablation.py --sae outputs/sae/L7.pt` | Remove a concept's top-n features together; does the policy change? | 10 min |

Artifacts (positions, activation caches, SAEs, JSON results) go to `outputs/` (git-ignored).
`notebooks/explore_features.ipynb` is for eyeballing individual features.

## Layout

```
src/
  hook.py              load the net, encode positions, legal-move policy, capture/patch residuals
  sae.py               Top-K SAE
  train.py             training loop, AuxK loss, dead-latent tracking
  data/                lichess.py (corpus), cache.py (activation cache), positions.py (old HF corpus)
  eval/                board_props, codes, correlational, probes, causal (ablation + steering),
                       history_filter, fidelity, feature_mining
  autointerp/          LLM labeler + detection scoring
  viz/                 board rendering with the firing square highlighted
scripts/               the numbered pipeline above
tests/                 pytest; net-dependent tests gated on LEELA_SAE_CHECKPOINT
```

## Conventions worth knowing

- **Tokens.** The net works on 64 square-tokens in the side-to-move's frame: for White token `t` is
  square `t` (a1=0 … h8=63); for Black the board is flipped vertically, so token `t` is
  `chess.square_mirror(t)`. Labels in `eval/board_props.py` are produced in token order.
- **Own / enemy.** All properties are relative to the side to move, like the net's input planes.
- **History.** Everything trains and evaluates on positions encoded from a bare FEN, with the
  current board copied into all 8 history slots (lc0's own convention). Real game history is used
  only, explicitly, by the history filter. This encoding cannot represent en passant.
- **Policy.** `output["policy"]` is raw logits over 1858 move slots including illegal ones, which are
  untrained. Always go through `hook.legal_moves_and_indices` / `hook.legal_log_probs`; castling is
  king-takes-rook in lc0's encoding.

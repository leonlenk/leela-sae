# leela-sae

Sparse autoencoders on the residual stream of a Leela Chess Zero transformer
([`BT3-768x15x24h-swa-2790000`](https://huggingface.co/lczerolens/BT3-768x15x24h-swa-2790000)),
with every learned feature checked against chess ground truth three ways: does it match a
board property on held-out positions, does it causally drive Leela's policy/value heads, and does it
survive being fed real move history.

**Results and conclusions: [RESULTS.md](RESULTS.md).** Original plan and design rationale:
[DESIGN.md](DESIGN.md).

## Why this project

Leon wanted to learn how sparse autoencoders work, and to see whether a published result
generalizes. [Karvonen et al. (2024)](https://adamkarvonen.github.io/machine_learning/2024/06/12/sae-board-game-eval.html)
trained SAEs on a small language model that learned chess by predicting the next character of
Lichess game transcripts, and scored them against the known board. Would the same approach work on
a real chess engine? Leela Chess Zero is among the strongest engines in the world, was trained from
scratch by self-play rather than on text, and is a transformer — a different model in a different
training regime.

The guiding principle was learning over speed. SAEs and their evaluations fail quietly: a wrong
activation site or a leaky metric produces a plausible result, not a crash. So Leon wrote the parts
that carry the ideas — the model hook, the SAE, the training loop, the evaluations — by hand, with
Claude Code limited to boilerplate and to writing each function's tests before he implemented it
(the contract below). After a few dozen hours the SAE trained, and the first results looked
promising: a feature for "own queen on this square" with F1 0.94; ablating it moved the policy
(KL 0.003) and the win-draw-loss estimate (−0.01); and, surprisingly, the enemy-pawn feature mattered
most (KL 0.01, WDL 0.06).

His own write-up at that point listed what he wasn't sure about: zero-ablation might push the
model off-distribution; at ~1200 Elo he couldn't judge whether the Hugging Face dataset was a good
sample of chess; and he wanted to check pins and forks, follow Karvonen et al. in testing whether
the whole board can be reconstructed, and compare earlier and later layers. Those questions set the
agenda for finishing the project. The answers are in [RESULTS.md](RESULTS.md): the ablations were
redone to keep the SAE's error and compared against matched controls (§5); the dataset was indeed
off-distribution and was replaced (§8); pins are in §4–5 (forks were not attempted); board
reconstruction is §10; layers 3, 7 and 11 are compared in §7. The instinct that piece identity —
the enemy-pawn feature especially — matters to the policy held up: in the retrained SAE, removing
the enemy-pawn feature changes Leela's top move in 28% of positions where it fires (6% for an
activation-matched control feature).

## Who did what

This started as a learning project with an explicit split between human-written and AI-written
code: files marked 🔒 were written by Leon, files marked 🤖 / ⚙️ ("agent-fillable") by Claude, and
for every 🔒 function Claude wrote the failing test first and Leon wrote the body (see the original
contract in DESIGN.md §3.5). Months later Leon asked Claude to finish it.

**Leon (June 2026)**
- Designed the project ([DESIGN.md](DESIGN.md)): Top-K SAEs on a Leela transformer's per-square
  residual stream, validated correlationally against python-chess properties and causally through
  the policy/value heads, with the "a feature is only real if it passes three checks" criterion.
- Wrote the hook (`src/hook.py`, `scripts/01`): loading BT3 through lczerolens, reproducing Leela's
  moves, finding the per-block residual sites (`encoderN/ln2`), batched capture. Measured that the
  net depends on its move-history planes and pivoted the design from "history-free net" to "history
  as a controlled confound" with the repeated-board encoding.
- Wrote the SAE and training loop (`src/sae.py`, `src/train.py`, `scripts/02`): Top-K encoding,
  unit-norm decoder, AuxK dead-latent revival, dead-latent tracking, warmup, and a KL-based
  "loss recovered" metric spliced through the net.
- Wrote the evaluation (`src/eval/board_props.py`, `correlational.py`, `causal.py`,
  `feature_mining.py`, and two notebooks): side-to-move-relative per-square labels, best-feature F1,
  feature patching measured as policy KL, top-activation mining and a feature-density scan.
- Chose the model, layer 7, k = 32, the 8× dictionary and the original training data.

**Claude, June 2026 (🤖 / ⚙️ files)** — the repo scaffold and environment, the tests-first
scaffolds for every 🔒 function, the activation cache (`src/data/cache.py`), the HF corpus loader
(`src/data/positions.py`) and the board renderer (`src/viz/features.py`).

**Claude, October 2026 (finishing it)** — at Leon's request, Claude (Opus 5.5 in Claude Code) took
the project over: had independent agent auditors review the code and claims, fixed what they found
(the policy head read mostly through illegal-move slots, castling and promotion bugs, selection-
biased F1s, an inactive causal control, an uninformative loss-recovered metric, off-distribution
training data, dead latents), built the Lichess corpus and retrained, and added the held-out
evaluation with probe baselines, error-preserving and joint ablations, steering, the history
filter with its positive control, the layer sweep, the data comparison, auto-interp, the board-
reconstruction metrics of Karvonen et al. (2024), this README and [RESULTS.md](RESULTS.md); two
more audit rounds checked the write-up. Most of the original code was rewritten along the way, but the
design, the activation site and token mapping he established, the SAE math and the core idea of
every evaluation are Leon's and held up under audit. `git log` shows the June history; everything after `cb8b706`
is the October work.

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
| 13 | `scripts/13_board_reconstruction.py outputs/sae/*.pt` | Karvonen et al. (2024) coverage + board reconstruction, with probe baseline; untrained-net controls come from `02`/`03` with `--random-weights 0 --random-init {permute,fresh}` | 5 min / SAE |
| 14 | `scripts/14_reconstruction_checks.py` | Checks on 13: the paper's exact protocol, trivial baselines for the strategy set, token collapse in the random nets | 10 min |

Artifacts (positions, activation caches, SAEs, JSON results) go to `outputs/`. Only
`outputs/results/` (the JSON and markdown behind every number in RESULTS.md) is committed. The
rest is regenerable; the caches alone are ~110 GB.
`notebooks/explore_features.ipynb` is for eyeballing individual features.

## Layout

```
src/
  hook.py              load the net, encode positions, legal-move policy, capture/patch residuals
  sae.py               Top-K SAE
  train.py             training loop, AuxK loss, dead-latent tracking
  data/                lichess.py (corpus), cache.py (activation cache), positions.py (old HF corpus)
  eval/                board_props, codes, correlational, probes, causal (ablation + steering),
                       history_filter, fidelity, feature_mining, board_reconstruction
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

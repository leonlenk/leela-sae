# Results: sparse autoencoders on Leela Chess Zero

What we trained, what we found, and — because a large part of this project turned out to be
finding our own mistakes — what the first version got wrong. Every number below is on held-out
Lichess games (never seen in SAE training or feature selection) unless stated otherwise.

## TL;DR

- **The pipeline was mostly right where it mattered least and wrong where it mattered most.** The
  activation site, token ordering and SAE math checked out. But the policy head was read with ~94%
  of its probability mass on untrained illegal-move slots, castling and promotions were mis-decoded,
  and the headline correlational and causal results from the first pass were artifacts (a "perfect"
  en passant feature for a property the net literally cannot see; a causal control that never
  fired). All fixed; details in [§1](#1-audit-what-the-first-version-got-wrong).
- **The SAEs are decent reconstructions.** At layer 7 (of 15) a Top-K SAE (k=32, 6144 features)
  explains 85% of the residual variance; spliced into the net, Leela still picks the same move in
  90% of positions (vs 7% if the layer is replaced by its mean). Its errors are ~2.5× more harmful
  than random noise of the same size — they fall in directions the net uses.
- **Piece identity is everywhere and causally used.** Every layer has near-perfect
  (F1 ≈ 0.99) "own knight here" / "enemy pawn here"-style features. That is expected — it is the
  input, carried forward — but removing one does change play: e.g. ablating the own-knight feature
  cuts the probability of knight moves by ~20% and changes the top move in 28% of positions.
- **Computed concepts exist, but rarely as one clean feature.** Things the net must work out —
  check, hanging pieces, pins, passed pawns — are linearly decodable from the residual stream
  (probe F1 0.75–0.99) and absent from the raw input (probe F1 ≤ 0.13). The SAE has features for
  some of them: a single feature tracks "my king is in check" with F1 0.95 at layer 3; "enemy piece
  hanging" reaches 0.73 at layer 3. More often a concept is split across several features or only
  partly captured: for hanging pieces, pins and passed pawns even the best 30 features together
  reach F1 0.55–0.72, against 0.80–0.90 for a probe.
- **Those concept features are causal, weakly.** Removing a check, hanging-piece or passed-pawn
  feature moves the policy more than removing an activation-matched control on 77–94% of
  positions, but by tiny amounts (KL ~10⁻⁴ nats, 100–1000× less than a piece feature) — later
  layers can presumably recompute the fact from the intact board. Writing
  them in works better: adding the "in check" feature to a king that isn't in check raises the
  probability of king moves by 47%; adding the "hanging" feature to a defended enemy piece raises
  capture probability by 6%. Two of four steering hypotheses held; two did not.
- **History is not a big confound here.** Feeding real game history instead of the repeated
  current board barely changes the policy (median KL 0.005 nats) and 78–87% of features are
  stable (activation correlation ≥ 0.9), including every feature named above. {HC_TLDR}
- **The original training data was the wrong data.** The 30k-FEN Hugging Face corpus looks like
  random play (no endgames, 41% of positions down ≥ 3 pawns). At equal budget, an SAE trained on it
  reconstructs real games far worse (FVU 0.29 vs 0.20) than one trained on Lichess positions.

## Setup

| | |
|---|---|
| Net | `lczerolens/BT3-768x15x24h-swa-2790000`: 15 transformer blocks, d_model 768, 64 square-tokens |
| Site | `encoderN/ln2`, the residual stream after block N (post-LayerNorm architecture, so this is exactly block N+1's input). Layers 3, 7, 11 |
| Input | history-free: a bare FEN with the current board copied into all 8 history slots (lc0's own FEN convention). Real history only in §6 |
| Data | 316k positions from June 2024 Lichess games, both players ≥ 2000, ≥ 3+0, no bots; 2 random plies per game; split by game (90/5/5); de-duplicated by position, no held-out position also in train |
| SAE | Top-K, k = 32, 6144 features (8×), tied init, unit-norm decoder, AuxK dead-latent revival (Gao et al. 2024) |
| Training | 200k boards × 64 tokens = 12.8M tokens, 2 epochs, Adam (lr 3.3e-4, warmup, linear decay), ~6 min on a laptop RTX 5050 |
| Eval splits | feature selection / probe fitting on 4000 validation boards; every reported number on 4000 test boards (1000 for fidelity) |

## 1. Audit: what the first version got wrong

Three independent agent auditors (hook/data, SAE/training, evaluation) reviewed the code and
re-ran its claims on the GPU; the fixes were then re-audited. In rough order of how much they
mattered:

| Problem | Effect on the original results | Fix |
|---|---|---|
| Policy logits include 1858 move slots, most illegal; lc0 never trains illegal logits | 94% of a full softmax's mass sat on illegal moves, so the `loss_recovered` and causal KLs mostly measured junk (rank correlation with the legal-only KL: 0.63–0.87) | Softmax over legal moves only, everywhere |
| Castling scored at the wrong slot (lczerolens uses e1g1; lc0 encodes king-takes-rook, e1h1) | `predict_move` effectively never castled; castling mass missing from every KL | Remap castling in `hook.policy_index` |
| `decode_move` drops the promotion piece | Promotions returned as illegal moves; 2 of 9 "missed" mate-in-ones were correct | Score legal moves, return the move itself |
| En passant is invisible under the history-free encoding (no ep input plane; lc0 infers ep from history) | Reported en passant feature with F1 = **1.000** was impossible — 1 positive in 300 boards | Property marked hidden; only meaningful with real history |
| Board-level F1 = argmax over 6144 features, scored on the same 300 boards | Selection artifacts: shuffled labels reach the reported values; on held-out boards ep 1.00 → 0.00, "castled" 0.40 → 0.00, check 0.36 → 0.06 | Select on one split, report on another, with permutation null and always-true baseline |
| Causal "random control" was not active on the board (87.6% of the time), and each test used one board | The "control KL = 0" was true by construction | Error-preserving ablation, 64 boards where the feature is active, activation-matched control features on the same board, calibrated against random features |
| `loss_recovered` vs zero-ablation | Zeroing (or mean-ablating) this layer destroys ~2 nats of policy, so nearly any SAE scores ≈ 0.99 | Headline raw legal KL, top-1 agreement, ΔQ, and an equal-norm noise reference |
| Training data (30k HF FENs) | Off-distribution, small (1.15M tokens), no move history — see §8 | Lichess corpus, 11× more tokens |
| Random encoder init; dead-latent window of 100k tokens; AuxK picked negative pre-activations | 2.6–5% dead latents, 20% nearly dead; the window flagged 18.5% of latents as dead, 86% of which were alive | Tied init (0.05% dead), 2M-token window, ReLU'd AuxK |
| Tests that could not fail | e.g. the causal test passed for 4 of 4 deliberately broken harnesses; the F1 test could not tell F1 from precision | Tests with hand-computed asymmetric cases and "power" checks that must fail when nothing is patched |
| Found while redoing the eval: the linear-probe threshold was searched on a coarse quantile grid | Probes looked bad on rare properties (in-check: AUROC 1.0 but F1 0.30), which would have made one SAE feature look better than a probe | Exact threshold search |

What was right from the start: the activation site, the token ↔ square mapping (including the
vertical flip for Black to move), batched capture, the WDL head (probabilities, side-to-move view),
the SAE's encode/decode math, and the per-square piece-identity results.

## 2. Does the hook read the net correctly?

There is no lc0 binary here to give a reference move, so the gate uses positions with known
answers. The policy head alone (no search) on 1000 random Lichess puzzles:

| puzzle rating | ≤ 1000 | 1000–1500 | 1500–2000 | 2000–2500 | > 2500 | all |
|---|---|---|---|---|---|---|
| top-1 = solution | 99.1% | 99.1% | 90.3% | 85.9% | 75.6% | **93.8%** |

It finds mate in all 134 mate-in-one puzzles, castles and promotes correctly. With the history
planes zeroed instead (off-distribution), the top move changes in 94% of positions.

## 3. Reconstruction fidelity

All on the same 1000 test boards. KL is KL(clean ‖ SAE-spliced) over legal moves; "noise" splices
in isotropic Gaussian noise with the same per-token norm as the SAE's error.

| SAE | FVU | KL (nats) | top-1 move unchanged | mean \|ΔQ\| | KL with equal-norm noise | error vs noise |
|---|---|---|---|---|---|---|
| layer 3 | 0.112 | 0.0085 | 92.7% | 0.020 | 0.0036 | 2.4× |
| layer 7 | 0.151 | 0.020 | 89.6% | 0.032 | 0.0083 | 2.5× |
| layer 11 | 0.173 | 0.027 | 89.4% | 0.029 | 0.0108 | 2.5× |
| layer 7, original SAE (HF data, old code) | 0.309 | 0.065 | 83.7% | 0.056 | 0.0156 | 4.2× |

For reference, replacing layer 7 by its per-square mean gives KL 2.2 nats and 7% top-1 agreement,
so "loss recovered" is 0.99 for every row — not a useful number at this site. Deeper layers are
harder to reconstruct. A higher learning rate (1e-3) is marginally better at layer 7 (FVU 0.147).

## 4. Do features match chess properties?

For each property we take the single feature with the best F1 on validation boards and report its
F1 on test boards, next to two linear probes fit on the same validation boards: one on the layer's
full residual stream (is the information linearly there?) and one on each square's 112 raw input
planes (was it already in the input?).

![correlational](docs/figures/correlational_L7.png)

- **Input properties** (piece identity, castling rights) have near-perfect single features.
  Expected, and a good sanity check — but not a discovery.
- **Derived properties** are computed: input probes fail (hanging 0.06, pinned 0.03, in-check 0.13)
  while residual probes succeed (0.75–0.99). Single SAE features recover a fraction of that. At
  layer 7: in-check 0.75, enemy passed pawn 0.61, enemy hanging 0.51, pins ≤ 0.30.
- **Dense relational maps** (squares attacked, legal destinations, king zones) are almost perfectly
  linearly decodable (0.95–0.99) but no single feature tracks them (≤ 0.42).

**Split or missing?** Probing on only the n best SAE features for a property separates the two
(layer 7, test F1):

| property | 1 feature | 3 | 10 | 30 | full residual probe |
|---|---|---|---|---|---|
| own king in check | 0.74 | 0.95 | 0.96 | 0.96 | 0.99 |
| enemy en prise | 0.52 | 0.71 | 0.75 | 0.78 | 0.89 |
| enemy hanging | 0.56 | 0.69 | 0.70 | 0.72 | 0.90 |
| own pinned | 0.23 | 0.44 | 0.54 | 0.63 | 0.81 |
| enemy passed pawn | 0.65 | 0.62 | 0.60 | 0.64 | 0.85 |
| squares attacked by enemy | 0.68 | 0.68 | 0.75 | 0.82 | 0.99 |
| legal destinations | 0.51 | 0.51 | 0.69 | 0.80 | 0.95 |

"In check" is in the dictionary, split over ~3 features at layer 7 (at layers 3 and 11 a single
feature does it). Hanging pieces, pins and passed pawns plateau well below the probe: part of each
concept is not in any feature we can find — most likely it lives in the SAE's reconstruction error.

**What the matched features look like** (top activations on test boards, from the side to move's
view; highlighted square = where the feature fires):

![in check](docs/figures/gallery_own_king_in_check.svg)
![enemy hanging](docs/figures/gallery_enemy_hanging.svg)
![enemy passed pawn](docs/figures/gallery_enemy_passed_pawn.svg)
![own knight](docs/figures/gallery_own_knight.svg)

## 5. Do they matter to the policy?

**Ablation (necessity).** Remove one feature's contribution from the true activation (keeping the
SAE's error and every other feature), on 64 test boards where it is active, and compare with a
control feature active on the same board with matched activation mass.

![causal](docs/figures/causal_L7.png)

- Piece-identity features matter: median KL 10⁻⁴–10⁻¹ nats (kings and queens lowest, knights and
  pawns highest). Ablating the own-knight feature drops the probability of moves touching its
  squares by 20% (Δlog p = −0.22; control +0.003) and changes the top move on 28% of boards
  (control 3%).
- Features for check, hanging / en-prise pieces and passed pawns beat their matched controls on
  77–94% of boards (the 90th percentile for random features is 84%); weakly matched features
  (pins, attack maps, material) do not. But the absolute effect is ~10⁻⁵–10⁻⁴ nats: above a
  typical random feature (~2·10⁻⁶) yet 100–1000× below the piece features. Removing "in check"
  from a king in check changes almost nothing. A plausible reading: blocks 8–14 can recompute these
  facts from the board, which ablating one feature leaves intact, so ablation at a middle layer
  mostly tests redundancy.

**Steering (sufficiency).** Write a feature, at its typical (90th-percentile) activation or 3× that,
onto a square where its property is false, and test a specific prediction. 300 test boards each;
controls are a different feature of the same activation scale, and a random direction of the same
norm, written on the same square.

| feature (layer 7) | written on | prediction | Δ P(predicted moves), 1× / 3× | boards where it rose (3×) | beats feature control / random direction (3×) |
|---|---|---|---|---|---|
| in check (277) | own king, not in check | more king moves | **+3.8% / +47%** | 95% | 94% / 94% |
| enemy hanging (5123) | defended enemy piece we can capture | more captures there | **+1.4% / +6.3%** | 78% | 76% / 79% |
| own hanging (5424) | attacked-but-defended own piece | move it away | +0.1% / +0.5% | 55% | 61% / 57% |
| own passed pawn (1581) | own pawn that isn't passed | push it | −0.3% / −1.2% | 32% | 45% / 35% |

Two clear successes, one null and one failure. The failures line up with the correlational
evidence: feature 5424 only matches "own hanging" with F1 0.32 (auto-interp calls it a *defended*
pawn), and "passed pawn" may encode "advanced pawn near promotion" — which a blocked pawn is not.

**Scorecard against DESIGN.md's three checks** (correlational: held-out F1 ≥ 0.5 and ≥ 0.1 above the
permutation null; causal: ablation beats matched controls more often than 90% of random features
do; history: §6). Derived-concept features that pass all three:

| layer | features passing all three checks |
|---|---|
| 3 | in check (F1 0.95), enemy hanging / en prise (0.73), material ahead (0.71), own passed pawn (0.57) |
| 7 | enemy passed pawn (0.61), enemy hanging (0.51) |
| 11 | in check (0.84), own en prise (0.54), own piece has a legal move (0.54) |

11 of 12 piece-identity features pass at layer 7 (own pawn misses the calibrated causal bar despite
an 8× larger effect than its control: its matched controls are other strong features). Full
per-property tables are generated by `scripts/10_figures.py`.

## 6. Does history change the picture?

The SAEs never see move history. To check that their features track the board rather than an
artifact of the repeated-board encoding, we re-encoded 1500 test positions with their real Lichess
move history and compared every feature's activation on every square.

| | layer 3 | layer 7 | layer 11 |
|---|---|---|---|
| policy KL, real vs repeated history (median) | 0.005 | 0.005 | 0.005 |
| SAE FVU: repeated → real history | 0.112 → 0.131 | 0.152 → 0.160 | 0.175 → 0.182 |
| features with activation correlation ≥ 0.9 | 81% | 87% | 78% |

Every feature named in this document survives (correlation ≥ 0.94). The original SAE was far less
stable: only 33% of its features survive.

**Does the filter have power?** {HC_SECTION}

## 7. Across depth

![layers](docs/figures/layers.png)

Single-feature matches for derived concepts are generally best at layer 3 (in-check 0.95, enemy
hanging 0.73, own pinned 0.50, enemy king zone 0.56) and weakest at layer 7, where in-check is
split across several features; layer 11 recovers some (in-check 0.84). Reconstruction gets harder
with depth (FVU 0.11 → 0.15 → 0.17) and the layer-3 dictionary has more very rare features (16% never
fire on 256k test tokens, vs 3% at layer 7 and 1% at layer 11).

## 8. Did the original data help?

Same code, same config, same number of steps, 18k boards each, layer 7; FVU / top-1 move agreement
on 500 held-out boards from each source:

| trained on ↓ / evaluated on → | Lichess test games | HF hold-out FENs |
|---|---|---|
| HF FENs (original) | 0.294 / 84.0% | 0.167 / 93.8% |
| Lichess games | **0.199 / 87.8%** | 0.219 / 89.2% |

The HF-trained SAE barely transfers to real games; the Lichess-trained one transfers to the HF
positions reasonably. Scaling the Lichess data 11× (to the main run) improves layer-7 FVU further,
0.199 → 0.151. The HF corpus was dropped.

## 9. Auto-interp

Claude (Sonnet, through the CLI) labeled 61 layer-7 features from their top activations, then — in
a fresh call with only the label — predicted which of 10 held-out boards (half where the feature
fires) it fires on, and where.

| features | balanced accuracy | firing square guessed |
|---|---|---|
| matched to a property (31) | 0.70 | 56% |
| random live features (30) | 0.60 | 23% |

Labels are fluent and usually plausible ("own fianchettoed bishop on g7 shielding the castled
king", "empty squares an enemy knight attacks"), but for typical features only modestly
predictive. For the property-matched features they mostly agree with the property and add detail —
e.g. the own-rook feature fires most on passive back-rank rooks — and where they disagree the
correlational match was weak too. With 10 boards per feature, individual scores are noisy; only
the averages should be read.

## 10. DESIGN.md checklist

- [x] Hook reproduces Leela's move — proxy: 93.8% of 1000 puzzles, all 134 mate-in-ones (no lc0
  reference binary was available)
- [x] Top-K SAE trains; L0 = k; dead latents 0.05% during training
- [x] Loss recovered measured — and replaced as a headline by legal KL / top-1 / noise reference
- [x] ≥ 1 feature with high F1 against a board property and a causal head effect — piece identity
  (strongly), and derived concepts such as "in check" (F1 0.95 at layer 3; steering +47% king moves)
- [ ] Ray pipeline + Accelerate scaling — dropped: unnecessary on one GPU
- [x] Auto-interp labels for top features

## Limitations

- One net, three layers, one SAE size and sparsity (k = 32, 8× expansion); no sweep over either.
- The properties are hand-picked; features that track concepts we didn't think to label are
  invisible to the correlational eval (auto-interp suggests many fire on empty squares in
  structured ways — escape squares, knight destinations).
- Ablation and steering are single-feature, single-layer interventions; effects on the value head
  and on search were not studied.
- Positions come from human games ≥ 2000 Lichess rating, not Leela's own self-play distribution.
- The move-reproduction gate is a proxy; a check against the lc0 engine itself is still missing.

## Reproduce

See [README.md](README.md) — `scripts/00` … `scripts/11` regenerate every number and figure here
(≈ 3 hours on an 8 GB laptop GPU, plus ~2 GB of downloads and ~80 GB of activation caches).

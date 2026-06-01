# Leela SAE — Design Doc

**Goal:** Train a sparse autoencoder on the residual stream of a Leela Chess Zero
transformer net, and validate the learned features against chess ground truth
(both correlational, via python-chess board properties, and causal, via the
policy/value heads). The *point of the project is understanding*, so the work
split below optimizes for that, not for raw speed.

**Non-goal:** A polished library. This is a learning + research artifact. Scale
and glue can be delegated; the conceptual spine cannot.

---

## 1. Technical decisions (locked before any code)

| Decision | Choice | Why |
|---|---|---|
| Net family | Transformer (BT-series), not conv | Clean residual stream per square; mirrors LLM mech-interp |
| Net variant | **Current-board-only** finetuned net (Jenner et al. lineage) | Removes the 8-position history stack, so a feature can't secretly key off "pawn just moved." Simplifies activation patching. |
| Activation site | Per-square residual stream after a chosen block (transformer net) | per-square = 64 "tokens"/position. NB: on a conv net it's `[B, C, 8, 8]` instead — confirm which arch your checkpoint is |
| Hooking | `lczerolens` + **nnsight** | `LczeroModel.from_hf(...)` + `LczeroBoard`; nnsight does the capture/patch. Exact module name comes from `print(model)`, not from this doc |
| SAE variant | **TopK** to start (direct L0 control), JumpReLU as v2 | Sidesteps L1 feature-suppression; `k` is a clean knob |
| Eval — correlational | python-chess board-state properties (≈Karvonen's BSP method) | Ground truth; F1 of feature vs property |
| Eval — causal | Patch/ablate feature → watch policy & WDL heads move | Separates real features from sparse codes that merely reconstruct |

### Confirmed from lczerolens docs (v0.4.0)

Load + eval is two lines; heads come for free:

```python
from lczerolens import LczeroBoard, LczeroModel
model = LczeroModel.from_hf("lczerolens/<checkpoint>")   # loads ONNX under the hood
board = LczeroBoard("<FEN>")
output = model(board)        # TensorDict:
#   output["policy"] -> (1, 1858) move logits/probs (incl. illegal moves)
#   output["wdl"]    -> (1, 3)    win / draw / loss
#   output["board"]  -> (1, 112, 8, 8) input planes
move = board.decode_move(output["policy"].argmax())      # index -> move
legal = board.get_legal_indices()                        # mask to legal moves
```

→ The `policy` + `wdl` heads are your **causal-eval surface, no extra wiring**:
patch a feature, re-run `model(board)`, diff these tensors.

### Caveats that bit / will bite

- **History is in the input.** The `112`-plane input encodes Leela's 8-position
  history. It is NOT removable at the activation site. The Jenner "current-board"
  net is a *finetune* that ignores history. **Unverified:** whether the HF repo
  `lczerolens/evidence-of-learned-lookahead` is that finetune — its model card is
  empty. Confirm by loading it, not by reading a card.
- **Pick the ONNX file, not the .pt.** On HF these repos ship both `model.onnx`
  (scanner: Safe) and `model.pt` (scanner: Unsafe, because *all* pickles are
  flagged — the listed imports here are benign onnx2torch/torch.nn objects).
  `from_hf` uses ONNX anyway. General rule: prefer `.onnx`/`.safetensors` over
  `.pt`/`.bin`; if forced to load a pickle, `torch.load(..., weights_only=True)`.
- **Module names depend on architecture — read them off the model.** The docs'
  example `256x19-4508` is a *conv SE-ResNet* (modules like `block14/conv2/relu`,
  activations `[B, 256, 8, 8]`), NOT a transformer. The **BT3** net you want has
  attention/MLP blocks with different names. There is no way to know the exact
  hook target in advance — you get it by loading the net and printing it (this is
  step 1). lczerolens is framework-agnostic; pair it with **nnsight** for the
  actual capture/patch (the interpretability tutorial uses nnsight).

### Still to verify in step 1 (do these by hand, they ARE the first task)

1. Load `evidence-of-learned-lookahead` (or a BT3 checkpoint), `print(model)`,
   record the real module names of the per-square residual stream after each block.
2. Confirm history behavior: does changing only the history planes change the
   output? If not, it's the current-board net.
3. Confirm move reproduction on a handful of known positions.

---

## 2. The division principle

Delegate by **boredom, not difficulty**. The question for each task is *"does
doing this by hand build the mental model, or is the agent just typing faster?"*

But interpretability adds a second, overriding axis — **the trust asymmetry:**

> Interp bugs don't crash. They produce *plausible false results*. A wrong
> activation site, a label leaking into a probe, an eval measuring something
> subtly off — these make you "discover" a feature that isn't real, and it looks
> like a finding. So delegated code is not uniformly trustworthy. Trust it freely
> in the logging/plotting path. Treat it as guilty-until-verified in the hook and
> eval path.

---

## 3. Work split

### Human-owned (the spine — type every line)

- **The SAE module + training loop.** Conceptual heart, ~a screen of code. Owning
  it is the whole point: dead features, the sparsity/reconstruction tradeoff, L0
  collapse — you need the feel.
- **The first activation hook, end-to-end.** Load net → run a position →
  *reproduce Leela's actual move* → pull residual stream → confirm shape. Verify
  you did not include the history stack.
- **The eval logic.** Both halves. The feature→board-property mapping is where
  your *result* lives and it's chess-specific. The causal patching harness too.

### Agent-owned (limbs — delegate freely)

- Ray orchestration for parallel position generation + shuffle-buffer plumbing.
  *You* make the conceptual calls (distribution, buffer size, off-distribution %);
  the agent writes the worker boilerplate.
- Accelerate multi-GPU wrapper — *after* the single-GPU loop works by hand.
- Scaffolding: wandb logging, config management, plotting, repo skeleton.
- The ~1000 board-state-property enumeration — *after* you've hand-written the
  first dozen and understand the pattern.
- The auto-interp / DSPy / vLLM surface (side deliverable, not load-bearing).

---

## 3.5 The hand-holding contract (applies to EVERY 🔒 file)

I am learning this, not just shipping it. So the 🔒 files I write myself must
arrive from the agent as a *guided scaffold*, never as a blank function and never
as a finished implementation. Concretely, every 🔒 file the agent hands me has:

**(a) Tests first — written before I implement anything.** For each 🔒 function,
the agent writes the test(s) *with real, runnable assertions* up front, so I code
against a failing test (red → green). I should be able to run `pytest`, watch it
fail, implement, and watch it pass. This is the opposite of "leave assertions as
TODO" — I want the target defined before I start.

**(b) Heavy commenting.** Every non-obvious line gets an inline comment explaining
the *why*, not just the *what*. Tensor operations get shape annotations
(`# (B, 64, d_model)`). Anything that encodes a concept I'm trying to learn (the
top-k mask, the dead-feature aux loss, the F1 computation) gets a 2–4 line comment
explaining the idea, not just the mechanics.

**(c) Granular, ordered TODOs.** The body is broken into small numbered steps as
`# TODO (1): ...`, `# TODO (2): ...`, each one a single conceptual move I can do in
a few lines. Not one big `# TODO: implement this`. The TODOs should read like a
worked recipe so I always know the next small step.

**The one carve-out for interp safety:** even though the agent writes my tests, in
the **hook and eval paths** I must still *read and confirm each assertion against
something I can check by hand* before trusting it. A wrong test silently passes a
false result — in interpretability that is exactly as dangerous as wrong code. So:
agent writes the assertion; I verify the assertion is checking the right thing.

> Net effect: a 🔒 file is "comment + ordered TODO breadcrumbs + a failing test",
> and I fill in the bodies. The agent never writes the bodies of 🔒 files.

---

## 4. Sequencing (front-load understanding)

Do **not** touch distributed anything until step 3 — scaling something you don't
understand just multiplies the confusion.

1. **Hook + reproduce a move by hand.** Smallest possible script. Success =
   Leela's move matches and you've inspected one residual-stream tensor.
2. **Hand-write the SAE, train on a small cached activation sample.** Single GPU,
   no Ray. Watch *one* feature until you believe it.
3. **Now let agents scale the data pipeline** (Ray) and wrap training (Accelerate).
4. **Build the eval yourself** — correlational first, then causal.
5. **Bolt on auto-interp last** (DSPy pipeline, optional vLLM-served labeler).

A feature is only "real" once it passes: high activation correlation with a board
property **and** a causal effect on the heads when patched. Reconstruction quality
alone proves nothing.

---

## 5. Repo skeleton

Ownership marked: 🔒 = you write it, 🤖 = agent-fillable, ⚙️ = agent scaffolds, you decide design.

```
leela-sae/
├── CLAUDE.md                  🔒  ownership rules for Claude Code (see §6)
├── DESIGN.md                  🔒  this file
├── pyproject.toml             🤖
├── configs/
│   └── default.yaml           ⚙️  net checkpoint, layer idx, k, lr, tokens
├── src/
│   ├── hook.py                🔒  load net, run position, pull residual stream
│   ├── sae.py                 🔒  the SAE module + training loop
│   ├── data/
│   │   ├── positions.py       ⚙️  position distribution (you pick), Ray workers
│   │   └── shuffle_buffer.py  🤖
│   ├── train.py               🔒→⚙️  single-GPU first (you), then Accelerate (agent)
│   ├── eval/
│   │   ├── board_props.py     🔒  feature → python-chess property mapping
│   │   ├── correlational.py   🔒  F1 / coverage
│   │   └── causal.py          🔒  patch feature → policy/WDL delta
│   ├── autointerp/            🤖  DSPy pipeline, optional vLLM labeler
│   └── viz/                   🤖  plots, wandb
├── scripts/
│   ├── 01_reproduce_move.py   🔒  step 1 — the verification gate
│   └── 02_train_small.py      🔒  step 2
└── tests/
    ├── test_hook.py           🔒  reconstruct a known board by hand
    └── test_features.py       🔒  a "white king on g1" feature fires iff true
```

### Stub: `src/sae.py` — example of the §3.5 style the agent should produce

```python
import torch, torch.nn as nn

class TopKSAE(nn.Module):
    """One-layer Top-K sparse autoencoder.

    Shapes:  D = d_model (residual-stream width, e.g. 768)
             F = dict_size = expansion_factor * D  (the over-complete basis)
    A forward pass maps activations (B, D) -> sparse features (B, F) -> recon (B, D).
    Top-K means: keep only the k largest feature activations per token, zero the
    rest. This gives DIRECT control of sparsity (L0 == k) with no L1 penalty, so
    we avoid the L1 "feature suppression" that biases reconstruction.
    """
    def __init__(self, d_model: int, dict_size: int, k: int):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_model, dict_size)            # W_enc: (D -> F)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)# W_dec: (F -> D)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))      # subtracted on input
        # TODO (1): initialise pre_bias to the DATA MEAN (or geometric median) of a
        #           batch of activations, so reconstruction starts from a sane point.
        # TODO (2): normalise decoder columns to unit L2 norm (and re-normalise after
        #           each optimiser step). Without this the basis directions aren't
        #           comparable and feature magnitudes are arbitrary.

    def encode(self, x):                       # x: (B, D) -> feats: (B, F)
        # TODO (3): subtract pre_bias from x.                # (B, D)
        # TODO (4): apply self.encoder.                      # (B, F) pre-activations
        # TODO (5): keep only the top-k values per row, zero everything else.
        #           Hint: torch.topk over dim=-1, scatter back into a zeros tensor.
        raise NotImplementedError

    def decode(self, feats):                   # feats: (B, F) -> recon: (B, D)
        # TODO (6): self.decoder(feats) + pre_bias.          # (B, D)
        raise NotImplementedError

    def forward(self, x):                      # -> (recon (B,D), feats (B,F))
        # TODO (7): feats = encode(x); recon = decode(feats); return both.
        raise NotImplementedError

# Training loop lives in train.py. The breadcrumbs you'll implement there:
#   # TODO (A): loss = mse(recon, x)   — NO L1 term; Top-K already enforces sparsity
#   # TODO (B): add aux_k loss: let dead latents reconstruct the residual (x - recon)
#   #           so they get gradient and can revive. Track which latents are "dead".
#   # TODO (C): log every step: L0 (== k here, sanity check), fraction-dead,
#   #           and "loss recovered" (substitute recon into the net, compare CE).
```

### Matching test the agent writes FIRST (`tests/test_sae.py`)

```python
import torch
from src.sae import TopKSAE

def test_topk_enforces_l0_equals_k():
    sae = TopKSAE(d_model=16, dict_size=64, k=4)
    feats = sae.encode(torch.randn(8, 16))          # (B=8, F=64)
    # The whole point of Top-K: exactly k features are nonzero per row.
    assert (feats != 0).sum(dim=-1).tolist() == [4] * 8

def test_decoder_columns_unit_norm():
    sae = TopKSAE(d_model=16, dict_size=64, k=4)
    norms = sae.decoder.weight.norm(dim=0)          # one norm per dict column
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

def test_forward_returns_recon_and_feats_with_right_shapes():
    sae = TopKSAE(d_model=16, dict_size=64, k=4)
    recon, feats = sae(torch.randn(8, 16))
    assert recon.shape == (8, 16) and feats.shape == (8, 64)
```
(These fail until you implement the TODOs — that's the point. Run `pytest` red,
implement TODO (1)…(7), watch them go green.)

### Stub: `src/eval/causal.py` (same style — body is yours)

```python
def patch_feature(model, board, feature_idx, layer, scale=0.0):
    """Ablate/scale one SAE feature at `layer` and measure the effect on the heads.

    Returns (policy_delta, wdl_delta). A *real* feature moves the policy/WDL heads
    when you remove it; a feature that only helps reconstruction won't.
    """
    # TODO (1): run model(board) once, cache output["policy"], output["wdl"].
    # TODO (2): with an nnsight trace, at module `layer`: encode the activation
    #           with the SAE, set feature `feature_idx` to `scale`, decode back,
    #           and substitute it into the stream.
    # TODO (3): re-read policy/wdl from the patched run.
    # TODO (4): return the deltas vs the cached clean run.
    raise NotImplementedError
```

---

## 6. How to drive Claude Code

The mistake to avoid: telling Claude Code "build me a Leela SAE project." It will
helpfully implement the *whole thing*, including the spine you needed to write
yourself — and you'll have lost the project's entire purpose.

Instead, make the ownership boundary explicit and persistent.

### 6a. Drop a `CLAUDE.md` at the repo root

Claude Code reads this automatically every session. Put the boundary in it:

```markdown
# Working agreement

This is a LEARNING project. I write the core code myself; you scaffold it so I
can learn. Respect these ownership rules:

## 🔒 files — I write the bodies, you scaffold (see DESIGN.md §3.5)
Files: src/sae.py, src/hook.py, src/eval/*.py, src/train.py (single-GPU part),
scripts/01_*, scripts/02_*.
For these, NEVER write the function body. Instead hand me a guided scaffold:
  1. TESTS FIRST: write the test(s) for the function in tests/, with REAL runnable
     assertions, BEFORE I implement. I code against a failing test (red→green).
  2. HEAVY COMMENTS: inline comment the *why* on every non-obvious line; shape
     annotations on all tensor ops (e.g. `# (B, 64, d_model)`); a 2–4 line concept
     comment on anything conceptual (top-k mask, aux loss, F1).
  3. GRANULAR TODOS: break the body into small ordered steps `# TODO (1): ...`,
     `# TODO (2): ...` — each a single few-line move. Never one big TODO.
  4. End the function with `raise NotImplementedError` until I fill it in.
When I ask for a 🔒 file, give me the commented stub + the failing test + a short
plain-English explanation of the approach. NEVER the body.

## 🤖 / ⚙️ files — you may fully implement
src/data/*, src/viz/*, src/autointerp/*, configs/*, pyproject.toml, and the
Accelerate wrapper in train.py (ONLY after a working single-GPU loop exists).

## Always
- Never silently change the activation site or include the history stack.
- For hook/eval tests: I will personally verify each assertion checks the right
  thing before trusting it. Flag any assertion you're unsure about.
```

### 6b. Scaffolding prompts (use these in order)

**Step 0 — skeleton only:**
> "Create the directory structure and files from DESIGN.md §5. For every file
> marked 🔒, follow the §3.5 hand-holding contract: heavy comments, granular
> ordered `# TODO (n):` breadcrumbs, end each function with NotImplementedError,
> AND write the matching test with real assertions in tests/ so I can run it red.
> For 🤖/⚙️ files, leave a one-line module docstring. Do not write any 🔒 body."

**Per-file, when you reach a 🔒 file (e.g. the SAE):**
> "Scaffold src/sae.py per the §3.5 contract: commented stub with ordered TODOs
> and shape annotations, plus tests/test_sae.py with runnable assertions for the
> encoder top-k, decoder norm, and a dead-feature count. Don't write the bodies —
> I'll implement against your failing tests. Then explain the approach in 5 lines."

**Step 3 — delegate the data pipeline (after you've done steps 1–2 by hand):**
> "Implement src/data/positions.py and shuffle_buffer.py. Positions come from
> [your choice: Lichess puzzle DB / self-play / X% random]. Use Ray to parallelize
> generation across N workers. The shuffle buffer should hold M activations and
> emit decorrelated batches. Don't touch src/sae.py or the eval."

**Step 3b — the Accelerate wrapper:**
> "My single-GPU loop in train.py works (tests pass). Add an Accelerate wrapper
> for multi-GPU *without changing the loss, the SAE, or the eval logic*. Show me
> the diff."

**Eval tests (test-first, but I verify each assertion):**
> "Scaffold tests/test_hook.py and the eval tests per §3.5: fixtures to load the
> net + a known FEN, and write the assertions yourself — but for each assertion add
> a comment stating the hand-checkable fact it encodes (e.g. 'white king on g1 in
> this FEN'), so I can confirm the test is checking the right thing before I trust
> it. Leave the implementation under test as a commented TODO stub."

### 6c. Guardrails that catch the lying-bug class

- After any agent change to the data or hook path, re-run `scripts/01_reproduce_move.py`.
  If Leela's move stops matching, the pipeline drifted.
- Keep one hand-labeled position where you *know* what a feature should do
  (e.g. white king on g1). `tests/test_features.py` is your canary.
- Never accept agent-written eval code without checking it against a property you
  can verify by hand on a single board.

---

## 7. Definition of done (v1)

- [ ] Hook reproduces Leela's move on 100 positions (step 1 gate)
- [ ] Hand-written TopK SAE trains; L0 in target range, <X% dead features
- [ ] Loss-recovered measured (recon substituted back into the net)
- [ ] ≥1 feature with high F1 against a board property AND a causal head effect
- [ ] Ray pipeline + Accelerate scaling (agent-built, human-verified)
- [ ] (Stretch) auto-interp labels for top-N features via DSPy
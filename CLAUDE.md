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

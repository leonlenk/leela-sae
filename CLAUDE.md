# Working agreement

Status: the project is finished — results in RESULTS.md, pipeline and authorship in README.md.
It began as a learning project in which Leon wrote the 🔒 files and Claude only scaffolded them
(DESIGN.md §3.5); in October 2026 Leon asked Claude to finish it, so that contract no longer
applies. Ask before large new experiments.

## Always
- Never silently change the activation site (`encoderN/ln2`) or the input encoding. Main SAEs use
  the history-free encoding (`hook.encode_fens`); real move history enters only explicitly, via
  `hook.encode_boards_with_history`.
- Read the policy head only through `hook.legal_moves_and_indices` / `hook.legal_log_probs`
  (illegal slots are untrained; castling is king-takes-rook).
- Evaluation: select features / fit probes on the validation split, report on the test split, and
  show a baseline (probe, permutation null, matched control, untrained net) next to every number.
  Every number in RESULTS.md must trace to a file in `outputs/results/`.
- For hook/eval tests: each assertion states the hand-checkable fact it encodes, and include a
  power check that fails when nothing is patched. Flag any assertion you're unsure about.
- `scripts/07_autointerp.py` calls the `claude` CLI and spends Leon's subscription — ask first.

"""Step 1 — the verification gate (🔒 — scaffold per §3.5). Run this BY HAND; it IS the first task.

Success = (a) Leela's move matches on a few known positions, (b) you've printed the module names and
chosen a per-square residual site, (c) you've confirmed the net ignores history. Re-run after ANY
agent change to the data/hook path: if the move stops matching, the pipeline drifted (§6c).

The logic lives in src/hook.py and is covered by tests/test_hook.py — implement those first, then
this script just wires them together and prints what you need to eyeball.
"""

from src import hook


def main():
    # TODO (1): pick the checkpoint id. Do NOT guess — confirm it loads (model card may be empty).
    #           checkpoint = "lczerolens/<the BT3 / evidence-of-learned-lookahead repo you verified>"
    # TODO (2): model = hook.load_model(checkpoint)
    # TODO (3): print(model)  AND  hook.list_residual_modules(model) — record the real per-square
    #           residual module name(s). This is the only way to learn the hook target (§1).
    # TODO (4): for a handful of known FENs, assert hook.predict_move(model, fen) == the move you
    #           know Leela plays. Start with the mate-in-one from tests/test_hook.py.
    # TODO (5): resid = hook.get_residual_stream(model, fen, <chosen module>); print resid.shape and
    #           confirm it is (64, d_model) — per-square, transformer layout, not [C, 8, 8].
    # TODO (6): assert hook.output_depends_on_history(model, fen) is False — confirm current-board net.
    # TODO (7): write the confirmed checkpoint + module name into configs/default.yaml so the rest of
    #           the project (and the env-var-gated tests) use the same site.
    raise NotImplementedError


if __name__ == "__main__":
    main()

"""Step 1 — the verification gate (🔒 — scaffold per §3.5). Run this BY HAND; it IS the first task.

Success = (a) Leela's move matches on a few known positions, (b) you've printed the module names and
chosen a per-square residual site, (c) you've confirmed the net ignores history. Re-run after ANY
agent change to the data/hook path: if the move stops matching, the pipeline drifted (§6c).

The logic lives in src/hook.py and is covered by tests/test_hook.py — implement those first, then
this script just wires them together and prints what you need to eyeball.
"""

from src import hook
import chess

def main():
    model = hook.load_model("lczerolens/BT3-768x15x24h-swa-2790000", filename ="model.onnx")
    # print(model)
    modules = hook.list_residual_modules(model)
    print(modules)
    mate_in_one = "k7/8/1QK5/8/8/8/8/8 w - - 0 1"
    assert hook.predict_move(model, mate_in_one) == chess.Move.from_uci("b6b7")
    # puzzels from https://wtharvey.com/amur.html
    amura_v_calderin = "8/1p6/p2R1b2/P1B2k1p/1P3p1P/2r2b1K/5N2/8 w - - 0 1"
    best_move = hook.predict_move(model, amura_v_calderin)
    assert best_move == chess.Move.from_uci("d6f6"), f"Tried move: {best_move} instead"

    resid = hook.get_residual_stream(model, amura_v_calderin, modules[8])
    print("Should be (64, d_model) is: ", resid.shape)
    opening = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    # Diagnostic, not a gate (DESIGN §1/§5): characterise — don't reject — history dependence.
    deltas = hook.output_depends_on_history(model, opening)
    print(f"History dependence: zeroing Δ={deltas['zero_delta']:.3f}, "
          f"randomizing Δ={deltas['randomize_delta']:.3f}  (expect nonzero; we control for it)")


if __name__ == "__main__":
    main()

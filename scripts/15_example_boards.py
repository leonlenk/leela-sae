"""Example boards for the README: what the headline results look like on actual positions.

    uv run python scripts/15_example_boards.py       # needs outputs/sae/L7.pt and the corpus; ~3 min

All boards are Lichess TEST positions, picked by a fixed rule (stated in each figure's title), not by
hand. Two SVGs in docs/figures/, and the numbers behind them in outputs/results/example_boards.json:

  * example_ablation_enemy_pawn.svg — remove the enemy-pawn feature (error-preserving, as in §5)
    from every square where it fires; Leela's top move before vs after, and the value Q = W − L.
    Boards: the first test boards where the top move changes.
  * example_reconstruction.svg — the board read back from layer-7 SAE features with the §10 rule
    (features assigned on the validation boards, t from scripts/13), next to the true board, wrong
    squares marked. Boards: the first test boards.
"""

import json
from pathlib import Path

import chess
import chess.svg
import torch

from src import hook
from src.data import lichess
from src.eval import board_reconstruction as br
from src.eval import causal
from src.eval.codes import encode_corpus
from src.sae import TopKSAE
from src.viz.features import token_to_square

FIG, RES = Path("docs/figures"), Path("outputs/results")
CORPUS = "outputs/positions/lichess_2024-06.parquet"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BEFORE, AFTER = "#2a78d6cc", "#eb6834cc"            # validated categorical slots 1 and 2 (blue, orange)
FIRES, WRONG = "#f6c343cc", "#e5484dbb"
SIZE, PAD, CAP_LINE = 210, 14, 16


def compose(title: str, rows: list[list[tuple[str, list[str]]]], out: Path):
    """Rows of board SVGs with a title above and caption lines under each board."""
    heights = [SIZE + 18 + max(len(lines) for _, lines in row) * CAP_LINE for row in rows]
    w = max(len(row) for row in rows) * (SIZE + PAD) + PAD
    h = 36 + sum(heights)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" '
             f'height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="{SURFACE}"/>',
             f'<text x="{PAD}" y="24" font-family="sans-serif" font-size="15" font-weight="600" fill="{INK}">{title}</text>']
    y = 36
    for row, row_h in zip(rows, heights):
        for i, (svg, lines) in enumerate(row):
            x = PAD + i * (SIZE + PAD)
            parts.append(svg.replace("<svg ", f'<svg x="{x}" y="{y}" ', 1))
            for j, line in enumerate(lines):
                parts.append(f'<text x="{x}" y="{y + SIZE + 18 + j * CAP_LINE}" font-family="sans-serif" '
                             f'font-size="12" fill="{INK2 if j else INK}">{line}</text>')
        y += row_h
    parts.append("</svg>")
    out.write_text("\n".join(parts))


def board_svg(board, fill=None, arrows=()):
    return chess.svg.board(board, orientation=board.turn, fill=fill or {}, arrows=list(arrows),
                           size=SIZE, coordinates=False)


def side(board):
    return "White" if board.turn == chess.WHITE else "Black"


@torch.no_grad()
def policy_and_q(model, fens, module=None, fn=None):
    """Legal-move log-probs and Q = P(win) − P(loss) per board, clean or with a residual patch."""
    x = hook.encode_fens(fens)
    out = hook.run_patched(model, x, module, fn) if fn else model(x.to(hook.model_device(model)))
    res = []
    for b, fen in enumerate(fens):
        moves, idx = hook.legal_moves_and_indices(chess.Board(fen))
        lp = hook.legal_log_probs(out["policy"][b], idx).cpu()
        res.append((moves, lp, float(out["wdl"][b, 0] - out["wdl"][b, 2])))
    return res


def top(moves, lp):
    i = int(lp.argmax())
    return moves[i], float(lp[i].exp())


def ablation_examples(model, sae, module, fens, codes, feature, n=4):
    """First test boards (in corpus order) where removing `feature` changes Leela's top move."""
    act = codes.feature(feature)                                         # (N, 64)
    active = [i for i in range(len(fens)) if act[i].amax() > 0][:128]     # one batch on an 8 GB GPU
    sub = [fens[i] for i in active]
    clean = policy_and_q(model, sub)
    abl = policy_and_q(model, sub, module, causal._ablation_fn(sae, torch.full((len(sub),), feature)))
    flips = [k for k in range(len(sub)) if top(*clean[k][:2])[0] != top(*abl[k][:2])[0]]
    panels, rows = [], []
    for k in flips[:n]:
        fen, board = sub[k], chess.Board(sub[k])
        (m0, p0), (m1, p1) = top(*clean[k][:2]), top(*abl[k][:2])
        p0_after = float(abl[k][1][clean[k][0].index(m0)].exp())
        fired = {token_to_square(fen, t): FIRES for t in torch.nonzero(act[active[k]] > 0).flatten().tolist()}
        arrows = [chess.svg.Arrow(m0.from_square, m0.to_square, color=BEFORE),
                  chess.svg.Arrow(m1.from_square, m1.to_square, color=AFTER)]
        panels.append((board_svg(board, fired, arrows), [
            f"{board.san(m0)} ({p0:.0%}) → {board.san(m1)} ({p1:.0%})",
            f"{board.san(m0)} after: {p0_after:.0%} · {side(board)} to move",
            f"value Q {clean[k][2]:+.2f} → {abl[k][2]:+.2f}"]))
        rows.append({"fen": fen, "before": [m0.uci(), p0], "after": [m1.uci(), p1],
                     "q_before": clean[k][2], "q_after": abl[k][2]})
    dq = torch.tensor([a[2] - c[2] for a, c in zip(abl, clean)])
    stats = {"feature": feature, "n_active_boards": len(sub), "top1_changed": len(flips),
             "abs_dq_median": float(dq.abs().median()), "dq_median": float(dq.median()),
             "frac_q_rises": float((dq > 0).float().mean())}
    return panels, rows, stats


def describe(c, board):
    """Class id (side-to-move frame) -> e.g. 'white rook'."""
    color = board.turn if c <= 6 else not board.turn
    return f"{'white' if color else 'black'} {chess.piece_name((c - 1) % 6 + 1)}"


def reconstruction_examples(model, sae, module, n=3):
    t = json.loads((RES / "board_reconstruction.json").read_text())["L7"]["reconstruction_t"]
    tr = lichess.load_corpus(CORPUS, split="val")["fen"].tolist()
    te = lichess.load_corpus(CORPUS, split="test", limit=n)["fen"].tolist()
    c_tr, c_te = encode_corpus(model, sae, module, tr), encode_corpus(model, sae, module, te)
    fmax = br.feature_max(c_tr)
    pred = br.reconstruct(c_te, br.assign_features(c_tr, br.piece_grid(tr), fmax, t), fmax, t)
    y = br.piece_grid(te)
    true_row, rebuilt_row, rows = [], [], []
    for b, fen in enumerate(te):
        board = chess.Board(fen)
        rebuilt, wrong, notes = chess.Board(None), {}, []
        rebuilt.turn = board.turn
        for tok in range(64):
            sq, name = token_to_square(fen, tok), chess.square_name(token_to_square(fen, tok))
            classes = torch.nonzero(pred[b, tok, 1:]).flatten().add(1).tolist()
            truth = int(y[b, tok])
            if classes:                       # draw the true piece if it was among those read, else the first
                c = truth if truth in classes else classes[0]
                rebuilt.set_piece_at(sq, chess.Piece((c - 1) % 6 + 1, board.turn if c <= 6 else not board.turn))
            if classes == ([truth] if truth else []):
                continue
            wrong[sq] = WRONG
            others = " + ".join(describe(c, board) for c in classes if c != truth)
            if not classes:
                notes.append(f"{name}: missed {describe(truth, board)}")
            elif not truth:
                notes.append(f"{name}: extra {others}")
            elif truth in classes:
                notes.append(f"{name}: {describe(truth, board)}, also read as {others.split()[-1]}")
            else:
                notes.append(f"{name}: {describe(truth, board)} read as {others}")
        f1 = float(br.board_f1(pred[b:b + 1], y[b:b + 1])[0])
        status = "exact" if not wrong else f"{len(wrong)} wrong square{'s' * (len(wrong) > 1)}"
        true_row.append((board_svg(board), [f"true board · {side(board)} to move"]))
        rebuilt_row.append((board_svg(rebuilt, wrong), [f"SAE features · {status}"] + notes[:2]
                            + ([f"+{len(notes) - 2} more"] if len(notes) > 2 else [])))
        rows.append({"fen": fen, "rebuilt_fen": rebuilt.board_fen(), "board_f1": f1, "errors": notes})
    return [true_row, rebuilt_row], rows, {"t": t, "n_train_boards": len(tr)}


def main():
    sae = TopKSAE.load("outputs/sae/L7.pt")
    meta = torch.load("outputs/sae/L7.pt", map_location="cpu", weights_only=True)["meta"]
    module, model = meta["module"], hook.load_model()
    feat = {r["property"]: r["feature"] for r in json.loads((RES / "L7/results.json").read_text())["properties"]}
    fens = lichess.load_corpus(CORPUS, split="test", limit=4000)["fen"].tolist()
    codes = encode_corpus(model, sae, module, fens)
    out = {}

    panels, rows, stats = ablation_examples(model, sae, module, fens, codes, feat["enemy_pawn"])
    out["ablation_enemy_pawn"] = {**stats, "boards": rows}
    compose(f"Removing the enemy-pawn feature ({feat['enemy_pawn']}, yellow = where it fires): Leela's top move, "
            "blue before, orange after", [panels], FIG / "example_ablation_enemy_pawn.svg")

    panels, rows, stats = reconstruction_examples(model, sae, module)
    out["reconstruction"] = {**stats, "boards": rows}
    compose("Top: first three test boards · bottom: read from layer-7 SAE features",
            panels, FIG / "example_reconstruction.svg")

    (RES / "example_boards.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

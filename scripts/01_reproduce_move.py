"""Step 1 — verification gate for the hook path. Re-run after ANY change to src/hook.py (DESIGN.md §6c).

    uv run python scripts/01_reproduce_move.py

There is no lc0 binary here to give a reference move, so the gate uses positions with known answers:
  * Lichess puzzles (Lichess/chess-puzzles): the policy head's top legal move vs the puzzle's
    solution move, overall and for mate-in-one puzzles (where ANY mating move counts);
  * castling and promotion positions, which the old hook could not get right;
  * the history-dependence diagnostic, for the record.
Prints a summary and writes outputs/results/gate.json.
"""

import json
from pathlib import Path

import chess
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from lczerolens import LczeroBoard

from src import hook

N_PUZZLES = 1000
OUT = Path("outputs/results/gate.json")


def puzzles(n: int, seed: int = 0) -> pd.DataFrame:
    path = hf_hub_download("Lichess/chess-puzzles", "data/train-00000-of-00003.parquet", repo_type="dataset")
    df = pd.read_parquet(path, columns=["PuzzleId", "FEN", "Moves", "Rating", "Themes"])
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


@torch.no_grad()
def policy_top_moves(model, fens: list[str], batch_size: int = 256) -> list[chess.Move]:
    out = []
    for s in range(0, len(fens), batch_size):
        chunk = fens[s:s + batch_size]
        pol = model(hook.encode_fens(chunk).to(hook.model_device(model)))["policy"]
        for b, fen in enumerate(chunk):
            moves, idx = hook.legal_moves_and_indices(chess.Board(fen))
            out.append(moves[int(pol[b][idx.to(pol.device)].argmax())])
    return out


def main():
    model = hook.load_model()
    print("residual sites:", hook.list_residual_modules(model))

    # Puzzle FEN is the position BEFORE the opponent's setup move; the solver's first move is Moves[1].
    df = puzzles(N_PUZZLES)
    fens, answers, mate1 = [], [], []
    for row in df.itertuples():
        board = chess.Board(row.FEN)
        moves = row.Moves.split()
        board.push_uci(moves[0])
        fens.append(board.fen())
        answers.append(chess.Move.from_uci(moves[1]))
        mate1.append("mateIn1" in list(row.Themes))
    preds = policy_top_moves(model, fens)
    correct = [p == a for p, a in zip(preds, answers)]

    def mates(fen, move):
        b = chess.Board(fen)
        b.push(move)
        return b.is_checkmate()

    mate_ok = [mates(f, p) for f, p, m in zip(fens, preds, mate1) if m]
    by_rating = df.assign(correct=correct).groupby(pd.cut(df.Rating, [0, 1000, 1500, 2000, 2500, 4000]),
                                                     observed=True)["correct"].agg(["mean", "size"])

    special = {
        "ruy_lopez_castles": ("r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5", "e1g1"),
        "white_promotes": ("8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8q"),
        "black_promotes": ("k7/8/8/8/8/8/p6K/8 b - - 0 1", "a2a1q"),
        "mate_in_one_qb7": ("k7/8/1QK5/8/8/8/8/8 w - - 0 1", "b6b7"),
    }
    special_ok = {k: hook.predict_move(model, f).uci() == u for k, (f, u) in special.items()}
    history = [hook.output_depends_on_history(model, f) for f in fens[:100]]

    result = {
        "puzzles": len(fens),
        "puzzle_top1_accuracy": sum(correct) / len(correct),
        "mate_in_one_puzzles": len(mate_ok),
        "mate_in_one_found": sum(mate_ok) / max(len(mate_ok), 1),
        "accuracy_by_rating": {str(k): {"acc": float(v["mean"]), "n": int(v["size"])} for k, v in by_rating.iterrows()},
        "special_positions": special_ok,
        "history_zero_kl_median": float(torch.tensor([h["zero_kl"] for h in history]).median()),
        "history_zero_top1_changed": float(torch.tensor([h["zero_top1_changed"] for h in history]).mean()),
        "history_randomize_kl_median": float(torch.tensor([h["randomize_kl"] for h in history]).median()),
    }
    print(json.dumps(result, indent=2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    assert all(special_ok.values()), f"special positions failed: {special_ok}"
    assert result["mate_in_one_found"] >= 0.9, "policy head should find nearly all mate-in-ones"


if __name__ == "__main__":
    main()

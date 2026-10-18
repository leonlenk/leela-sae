"""Coverage and board reconstruction — the supervised SAE metrics of Karvonen et al. 2024,
"Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game
Models" (arXiv 2408.00113), adapted to Leela's 64 square-tokens.

Board state properties (BSPs): "piece type p is on square s", for 12 piece types (own/enemy x
P N B R Q K, in Leela's side-to-move frame) and 64 squares = 768 BSPs. Because Leela has one token
per square, BSP (s, p) is read off the activation at token s.

A feature f "fires" at threshold t when its activation exceeds t * f_max, f_max being its maximum
activation over the (training) data; t is swept over {0, 0.1, ..., 0.9} as in the paper.

  * COVERAGE(t) = mean over BSPs of the best F1 any single feature achieves for that BSP at t.
    1.0 means every (square, piece) pair has a feature that classifies it perfectly.
  * BOARD RECONSTRUCTION(t): on TRAIN positions, assign feature f to BSP (s, p) if, at token s, f
    predicts p with precision >= 0.95. On TEST positions, predict "p on s" wherever any feature
    assigned to (s, p) fires at token s, and score the predicted board against the true one with F1
    over pieces (empty squares are not a class; a piece predicted on an empty square is a false
    positive). Averaged over test positions. We also report the fraction of test boards
    reconstructed EXACTLY (every square right) — literally "fully reconstructing the board".

Here a single t is chosen on the train positions. The paper's Eq. 5 instead takes the best t for each
test board, which uses the test labels and reads a little higher (`per_board_best_f1`). The paper also
leaves pieces on their starting squares unscored, because in a PGN model they correlate with the
token's position in the string (`start_square_mask`). Coverage here is the best t of the averaged
score; Eq. 3 takes the best t per BSP, again slightly higher.
"""

from __future__ import annotations

import chess
import numpy as np
import torch

from src.eval.board_props import token_order
from src.eval.codes import SparseCodes

N_CLASSES = 13                         # 0 = empty, 1..6 own P N B R Q K, 7..12 enemy P N B R Q K
THRESHOLDS = [i / 10 for i in range(10)]
PIECE_NAMES = ["empty"] + [f"{side}_{chess.piece_name(p)}" for side in ("own", "enemy") for p in chess.PIECE_TYPES]


def piece_grid(fens) -> torch.Tensor:
    """(N, 64) long: class of the piece on each TOKEN's square (0 empty, 1-6 own, 7-12 enemy)."""
    out = np.zeros((len(fens), 64), dtype=np.int64)
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        perm = token_order(board)                               # token t covers square perm[t]
        for t in range(64):
            piece = board.piece_at(int(perm[t]))
            if piece is not None:
                out[i, t] = piece.piece_type + (0 if piece.color == board.turn else 6)
    return torch.from_numpy(out)


def feature_max(codes: SparseCodes) -> torch.Tensor:
    """(F,) maximum activation of every feature over the given codes (0 for never-active)."""
    return torch.zeros(codes.n_features).scatter_reduce_(0, codes.idx.reshape(-1), codes.val.reshape(-1), "amax")


def _counts(codes: SparseCodes, labels: torch.Tensor, fmax: torch.Tensor, t: float, square: int):
    """For token position `square`: fired[f], tp[f, c] (fires AND label == c), positives[c]."""
    idx, val = codes.idx[:, square], codes.val[:, square]               # (N, k)
    on = val > t * fmax[idx]                                            # (N, k)
    y = labels[:, square]                                               # (N,)
    F = codes.n_features
    fired = torch.bincount(idx[on], minlength=F)
    pair = idx * N_CLASSES + y[:, None]                                 # (N, k) joint index (f, class)
    tp = torch.bincount(pair[on], minlength=F * N_CLASSES).reshape(F, N_CLASSES)
    pos = torch.bincount(y, minlength=N_CLASSES)
    return fired, tp, pos


def coverage(codes: SparseCodes, labels: torch.Tensor, fmax: torch.Tensor, t: float) -> dict:
    """Mean over the BSPs that occur in `labels` of the best single-feature F1 at threshold t."""
    best = torch.full((64, N_CLASSES), float("nan"))
    for s in range(64):
        fired, tp, pos = _counts(codes, labels, fmax, t, s)
        f1 = 2 * tp.double() / (fired[:, None] + pos[None, :]).double().clamp_min(1)   # (F, 13)
        best[s] = f1.max(0).values.float()
        best[s, pos == 0] = float("nan")                                # BSP never occurs: skip
    best[:, 0] = float("nan")                                           # empty is not a BSP
    per_class = {PIECE_NAMES[c]: float(np.nanmean(best[:, c].numpy())) for c in range(1, N_CLASSES)}
    return {"coverage": float(np.nanmean(best.numpy())), "per_piece": per_class,
            "n_bsps": int((~torch.isnan(best)).sum())}


def assign_features(codes: SparseCodes, labels: torch.Tensor, fmax: torch.Tensor, t: float,
                    min_precision: float = 0.95, min_fires: int = 1) -> torch.Tensor:
    """(64, F, 13) bool: feature f is assigned to BSP (square s, class c) — precision >= 0.95 on these
    (training) positions at token s. Class 0 (empty) is never assigned."""
    out = torch.zeros(64, codes.n_features, N_CLASSES, dtype=torch.bool)
    for s in range(64):
        fired, tp, _ = _counts(codes, labels, fmax, t, s)
        prec = tp.double() / fired[:, None].double().clamp_min(1)
        out[s] = (prec >= min_precision) & (fired[:, None] >= min_fires)
    out[:, :, 0] = False
    return out


def reconstruct(codes: SparseCodes, assigned: torch.Tensor, fmax: torch.Tensor, t: float) -> torch.Tensor:
    """(N, 64, 13) bool: predicted classes on each token (several, or none, may be predicted)."""
    N, _, k = codes.idx.shape
    pred = torch.zeros(N, 64, N_CLASSES, dtype=torch.bool)
    for s in range(64):
        idx, val = codes.idx[:, s], codes.val[:, s]
        on = val > t * fmax[idx]                                        # (N, k)
        pred[:, s] = (assigned[s][idx] & on[..., None]).any(1)           # (N, 13)
    return pred


def _board_counts(pred: torch.Tensor, labels: torch.Tensor, ignore: torch.Tensor | None = None):
    """Per-position tp, fp, fn over pieces. Empty squares are never scored; `ignore` ((64, 13) bool)
    drops further (token, class) entries from both prediction and truth."""
    truth = torch.nn.functional.one_hot(labels, N_CLASSES).bool()       # (N, 64, 13)
    truth[..., 0] = False
    pred = pred.clone()
    pred[..., 0] = False
    if ignore is not None:
        truth &= ~ignore
        pred &= ~ignore
    tp = (pred & truth).sum((1, 2)).double()
    fp = (pred & ~truth).sum((1, 2)).double()
    fn = (~pred & truth).sum((1, 2)).double()
    return tp, fp, fn


def board_f1(pred: torch.Tensor, labels: torch.Tensor, ignore: torch.Tensor | None = None) -> torch.Tensor:
    """(N,) per-position F1 over pieces."""
    tp, fp, fn = _board_counts(pred, labels, ignore)
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def score_boards(pred: torch.Tensor, labels: torch.Tensor, ignore: torch.Tensor | None = None) -> dict:
    """Per-position F1 over pieces (empty squares unscored), its mean, and the exact-board rate."""
    tp, fp, fn = _board_counts(pred, labels, ignore)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    return {"f1": float(f1.mean()), "exact_board_rate": float(((fp == 0) & (fn == 0)).double().mean()),
            "precision": float((tp / (tp + fp).clamp_min(1)).mean()), "recall": float((tp / (tp + fn).clamp_min(1)).mean())}


def start_square_mask() -> torch.Tensor:
    """(64, 13) bool: (token, class) = a piece on its starting square. The same for both colours in
    the side-to-move frame (own back rank on tokens 0-7, own pawns 8-15, enemy pawns 48-55, enemy back
    rank 56-63, queen on the d-file)."""
    mask = torch.zeros(64, N_CLASSES, dtype=torch.bool)
    back = [4, 2, 3, 5, 6, 3, 2, 4]                                     # R N B Q K B N R as own classes
    for f in range(8):
        mask[f, back[f]] = mask[8 + f, 1] = True
        mask[56 + f, back[f] + 6] = mask[48 + f, 7] = True
    return mask


def per_board_best_f1(preds_by_t: list[torch.Tensor], labels: torch.Tensor,
                      ignore: torch.Tensor | None = None) -> float:
    """The paper's Eq. 5: mean over positions of the best F1 over thresholds (one prediction per t)."""
    return float(torch.stack([board_f1(p, labels, ignore) for p in preds_by_t]).max(0).values.mean())


def best_feature_sharing(codes: SparseCodes, labels: torch.Tensor, fmax: torch.Tensor, t: float) -> dict:
    """How many distinct features are the best match (coverage) for the BSPs that occur. With one token
    per square, a square-agnostic "own knight here" feature can be the best match on many squares."""
    best = {}
    for s in range(64):
        fired, tp, pos = _counts(codes, labels, fmax, t, s)
        f1 = 2 * tp.double() / (fired[:, None] + pos[None, :]).double().clamp_min(1)
        for c in range(1, N_CLASSES):
            if pos[c] > 0:
                best[(s, c)] = int(f1[:, c].argmax())
    per_feature = np.bincount(list(best.values()))
    return {"n_bsps": len(best), "n_distinct_best_features": int((per_feature > 0).sum()),
            "max_bsps_per_feature": int(per_feature.max())}


def evaluate(codes_train: SparseCodes, labels_train: torch.Tensor, codes_test: SparseCodes,
             labels_test: torch.Tensor, thresholds=THRESHOLDS, min_fires: int = 1) -> dict:
    """Coverage and board reconstruction across the threshold grid; t for reconstruction chosen on train.

    min_fires=1 is the paper's rule (any feature with >= 95% precision on the training positions).
    With few training positions per (square, piece), features that fire once or twice get assigned by
    luck and add false positives at test time; min_fires > 1 guards against that.
    """
    fmax = feature_max(codes_train)
    per_t = {}
    for t in thresholds:
        assigned = assign_features(codes_train, labels_train, fmax, t, min_fires=min_fires)
        per_t[t] = {
            "coverage_test": coverage(codes_test, labels_test, fmax, t)["coverage"],
            "reconstruction_train": score_boards(reconstruct(codes_train, assigned, fmax, t), labels_train),
            "reconstruction_test": score_boards(reconstruct(codes_test, assigned, fmax, t), labels_test),
            "n_assigned_pairs": int(assigned.sum()),
        }
    t_best = max(thresholds, key=lambda t: per_t[t]["reconstruction_train"]["f1"])
    t_cov = max(thresholds, key=lambda t: per_t[t]["coverage_test"])
    return {
        "coverage": per_t[t_cov]["coverage_test"], "coverage_t": t_cov,
        "coverage_per_piece": coverage(codes_test, labels_test, fmax, t_cov)["per_piece"],
        "reconstruction": per_t[t_best]["reconstruction_test"], "reconstruction_t": t_best,
        "per_threshold": {str(t): v for t, v in per_t.items()},
    }


def score_probe_predictions(pred_class: torch.Tensor, labels: torch.Tensor) -> dict:
    """Board-reconstruction score for a classifier that predicts ONE class per token (e.g. a probe)."""
    return score_boards(torch.nn.functional.one_hot(pred_class, N_CLASSES).bool(), labels)


# --------------------------------------------------------------------------------------------------
# Coverage for the paper's "strategy" property set (G_strategy): board-level facts plus two
# per-square sets. Same definition: best single-feature F1 over the threshold grid.
# --------------------------------------------------------------------------------------------------

def strategy_labels(fens) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """({per-square name: (N, 64) bool}, {board-level name: (N,) bool}), Leela token order.

    Analogues of Karvonen et al.'s chess strategy BSPs: check, can_check, queen, pin, castling
    rights, bishop pair (board-level); legal_moves = squares some legal move lands on, and
    threatened_squares = own pieces attacked by the enemy (per square). Fork and "ambiguous move"
    are omitted.
    """
    from src.eval.board_props import label_arrays
    sq, bd = label_arrays(fens)
    own_piece = torch.zeros(len(fens), 64, dtype=torch.bool)
    for p in ("pawn", "knight", "bishop", "rook", "queen", "king"):
        own_piece |= torch.from_numpy(sq[f"own_{p}"])
    per_square = {"legal_moves": torch.from_numpy(sq["legal_to"]),
                  "threatened_squares": own_piece & torch.from_numpy(sq["attacked_by_enemy"])}
    boards = [chess.Board(f) for f in fens]
    bishop_pair = torch.tensor([len(b.pieces(chess.BISHOP, b.turn)) >= 2 for b in boards])   # side to move
    board = {"check": torch.from_numpy(bd["is_check"]),
             "can_check": torch.from_numpy(sq["check_square"].any(1)),
             "queen": torch.from_numpy(bd["queens_on"]),
             "pin": torch.from_numpy(sq["own_pinned"].any(1) | sq["enemy_pinned"].any(1)),
             "own_can_castle_kingside": torch.from_numpy(bd["own_can_castle_kingside"]),
             "own_can_castle_queenside": torch.from_numpy(bd["own_can_castle_queenside"]),
             "enemy_can_castle_kingside": torch.from_numpy(bd["enemy_can_castle_kingside"]),
             "enemy_can_castle_queenside": torch.from_numpy(bd["enemy_can_castle_queenside"]),
             "bishop_pair": bishop_pair}
    return per_square, board


def always_fires_f1(per_square: dict, board: dict) -> dict:
    """Trivial baseline for strategy coverage: a classifier that always says yes scores
    F1 = 2p / (1 + p) at base rate p (per square, averaged over the squares where it ever occurs)."""
    out = {}
    for name, y in per_square.items():
        p = y.double().mean(0)
        out[name] = float((2 * p / (1 + p))[y.any(0)].mean())
    for name, y in board.items():
        if y.any():
            p = float(y.double().mean())
            out[name] = 2 * p / (1 + p)
    return out


def _best_f1(fired: torch.Tensor, tp: torch.Tensor, positives: int) -> float:
    return float((2 * tp.double() / (fired + positives).double().clamp_min(1)).max())


def strategy_coverage(codes: SparseCodes, per_square: dict, board: dict, fmax: torch.Tensor,
                      thresholds=THRESHOLDS) -> dict:
    """Per property: best single-feature F1, maximised over the threshold grid. Per-square sets are
    64 BSPs each (token s, its own label) averaged; board-level ones use each feature's max over the
    board's 64 tokens."""
    out = {}
    for name, y in per_square.items():
        best = []
        for s in range(64):
            ys = y[:, s]
            if ys.sum() == 0:
                continue
            f1_t = []
            for t in thresholds:
                idx, val = codes.idx[:, s], codes.val[:, s]
                on = val > t * fmax[idx]
                fired = torch.bincount(idx[on], minlength=codes.n_features)
                tp = torch.bincount(idx[on & ys[:, None]], minlength=codes.n_features)
                f1_t.append(_best_f1(fired, tp, int(ys.sum())))
            best.append(f1_t)
        out[name] = float(torch.tensor(best).mean(0).max())             # coverage(t), best t
    bm = codes.board_max()                                              # (N, F)
    for name, y in board.items():
        if y.sum() == 0:
            continue
        f1_t = []
        for t in thresholds:
            on = (bm > t * fmax[None, :]).double()
            f1_t.append(_best_f1(on.sum(0), y.double() @ on, int(y.sum())))
        out[name] = max(f1_t)
    out["mean"] = float(sum(out.values()) / len(out))
    return out

"""Figures + summary tables for RESULTS.md, from the JSON written by scripts 04-09.

    uv run python scripts/10_figures.py

Writes PNGs and board-gallery SVGs to docs/figures/ and prints markdown tables. Missing inputs are
skipped, so it can be re-run as results come in.
"""

import json
from pathlib import Path

import chess
import chess.svg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

FIG = Path("docs/figures")
RES = Path("outputs/results")

# Palette: validated categorical slots 1-3 (blue, orange, aqua) on a light surface; text in ink tokens.
SURFACE, INK, INK2, GRID, AXIS, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#c3c2b7", "#a3a29b"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MARKERS = ["o", "s", "D"]                    # secondary encoding so identity never rests on colour alone

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False,
})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def rescore(res):
    """Recompute each property's verdict with the calibrated causal rule (random-feature q90), so
    results written by older versions of 04_evaluate are judged identically."""
    rnd = [c for c in res["causal_random_features"] if c.get("n_boards", 0) >= 32]
    q = lambda k: float(torch.tensor([c[k] for c in rnd if c[k] == c[k]]).quantile(0.9))
    thr = {"kl_win_rate": q("kl_win_rate"), "target_win_rate": q("target_win_rate")}
    for r in res["properties"]:
        c = r["causal"]
        r["passes"] = {
            "correlational": r["test_f1"] >= 0.5 and r["test_f1"] > r["null_f1_p95"] + 0.1,
            "causal": c.get("n_boards", 0) >= 32 and (c["kl_win_rate"] > thr["kl_win_rate"]
                                                     or (c["target_win_rate"] or 0) > thr["target_win_rate"]),
            "history": r["history"]["survives"],
        }
        r["passes"]["all"] = all(r["passes"].values())
    res["thresholds"] = thr
    return res


def dot(ax, y, x, color, marker, label):
    ax.scatter(x, y, s=42, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=label)


def fig_correlational(res, out):
    rows = [r for r in res["properties"] if "probe_layer" in r]
    groups = [("Computed by the net (derived)", [r for r in rows if r["origin"] == "derived"]),
              ("Present in the input planes", [r for r in rows if r["origin"] == "input"])]
    heights = [len(g) for _, g in groups]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 0.26 * sum(heights) + 2.2), gridspec_kw={"height_ratios": heights})
    for ax, (title, g) in zip(axes, groups):
        g = sorted(g, key=lambda r: r["test_f1"])
        y = np.arange(len(g))
        for i, r in enumerate(g):
            xs = [r["test_f1"], r["probe_layer"]["test_f1"], r["probe_input"]["test_f1"]]
            ax.plot([min(xs), max(xs)], [i, i], color=GRID, lw=2, zorder=1)
        dot(ax, y, [r["test_f1"] for r in g], BLUE, MARKERS[0], "best single SAE feature")
        dot(ax, y, [r["probe_layer"]["test_f1"] for r in g], ORANGE, MARKERS[1], "linear probe, same layer")
        dot(ax, y, [r["probe_input"]["test_f1"] for r in g], AQUA, MARKERS[2], "linear probe, raw input planes")
        ax.set_yticks(y, [r["property"].replace("_", " ") for r in g])
        ax.set_xlim(0, 1.02)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("held-out F1 (feature selected / probe fit on a disjoint split)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_causal(res, out):
    props = {r["property"]: r for r in res["properties"]}
    piece = [r for r in props.values() if r["origin"] == "input" and r["level"] == "square" and r["property"] != "empty"]
    # Derived-concept features: selected for a derived property AND not also the feature some
    # input property selected (e.g. capture_target picks the enemy-pawn feature — that's a piece feature).
    piece_feats = {r["feature"] for r in piece}
    derived, seen = [], set()
    for r in props.values():
        if r["origin"] == "derived" and r["level"] == "square" and r["feature"] not in piece_feats | seen:
            derived.append(r)
            seen.add(r["feature"])
    rnd = [c for c in res["causal_random_features"] if c.get("n_boards", 0) >= 32]
    cats = [("piece-identity features", [r["causal"] for r in piece], BLUE, MARKERS[0]),
            ("derived-concept features", [r["causal"] for r in derived], ORANGE, MARKERS[1]),
            ("random live features", rnd, MUTED, MARKERS[2])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    rng = np.random.default_rng(0)
    for i, (name, cs, color, m) in enumerate(cats):
        jitter = i + rng.uniform(-0.18, 0.18, len(cs))
        dot(a1, jitter, [max(c["kl_median"], 1e-7) for c in cs], color, m, name)
        dot(a2, jitter, [c["kl_win_rate"] for c in cs], color, m, name)
    a1.set_xscale("log")
    a1.set_xlabel("median policy KL after removing it (nats)")
    a2.axvline(res["thresholds"]["kl_win_rate"], color=INK2, lw=1)
    a2.text(res["thresholds"]["kl_win_rate"], 1.02, "random q90", color=INK2, fontsize=8, ha="center",
            transform=a2.get_xaxis_transform())
    a2.set_xlabel("share of boards where it beats a matched control")
    a2.set_xlim(0, 1.02)
    a1.set_yticks(range(3), [c[0] for c in cats])
    a1.invert_yaxis()
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    fig.suptitle("Ablating one feature (error-preserving), layer 7", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout(w_pad=3)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_history(ctrl, out):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    bins = np.linspace(-0.2, 1.0, 61)
    for name, label, color in (("L7", "SAE trained on history-free input", BLUE),
                               ("L7_realhist", "SAE trained with real game history", ORANGE)):
        if name in ctrl:
            c = np.clip(np.nan_to_num(np.array(ctrl[name]["corr"]), nan=-0.2), -0.2, 1.0)
            ax.hist(c, bins=bins, histtype="step", lw=2, color=color, label=label)
    ax.axvline(0.9, color=INK2, lw=1)
    ax.text(0.9, ax.get_ylim()[1] * 0.92, " survival threshold 0.9 ", color=INK2, fontsize=8, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("per-feature activation correlation: real history vs. current board repeated")
    ax.set_ylabel("features")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_layers(results_by_layer, out):
    layers = sorted(results_by_layer)
    props = [r["property"] for r in results_by_layer[layers[0]]["properties"] if r["origin"] == "derived"]
    f1 = {L: {r["property"]: r["test_f1"] for r in results_by_layer[L]["properties"]} for L in layers}
    order = sorted(props, key=lambda p: max(f1[L].get(p, 0) for L in layers))
    fig, ax = plt.subplots(figsize=(7, 0.27 * len(order) + 1.4))
    y = np.arange(len(order))
    for i, p in enumerate(order):
        xs = [f1[L].get(p, np.nan) for L in layers]
        ax.plot([np.nanmin(xs), np.nanmax(xs)], [i, i], color=GRID, lw=2, zorder=1)
    for (L, color, m) in zip(layers, (BLUE, ORANGE, AQUA), MARKERS):
        dot(ax, y, [f1[L].get(p, np.nan) for p in order], color, m, f"layer {L}")
    ax.set_yticks(y, [p.replace("_", " ") for p in order])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("held-out F1 of the best single SAE feature")
    ax.set_title("Derived properties across depth", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def gallery_svg(feature, hits, title, out, n=6, size=200):
    """Row of boards (side-to-move view) with the firing square highlighted; caption per board."""
    from src.viz.features import token_to_square
    pad, cap = 10, 34
    w, h = n * (size + pad) + pad, size + cap + 40
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="{SURFACE}"/>',
             f'<text x="{pad}" y="22" font-family="sans-serif" font-size="15" font-weight="600" fill="{INK}">{title}</text>']
    for i, (fen, token, act) in enumerate(hits[:n]):
        board = chess.Board(fen)
        sq = token_to_square(fen, token)
        svg = chess.svg.board(board, orientation=board.turn, fill={sq: "#f6c343cc"}, size=size, coordinates=False)
        svg = svg.replace("<svg ", f'<svg x="{pad + i * (size + pad)}" y="34" ', 1)
        parts.append(svg)
        side = "White" if board.turn == chess.WHITE else "Black"
        parts.append(f'<text x="{pad + i * (size + pad)}" y="{34 + size + 18}" font-family="sans-serif" font-size="12" '
                     f'fill="{INK2}">{chess.square_name(sq)} · {side} to move · act {act:.2f}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    by_layer = {}
    for L in (3, 7, 11):
        r = load(f"L{L}/results.json")
        probes_ = load(f"probes_L{L}.json")
        if r:
            for row in r["properties"]:                  # per-layer probes replace any stored in results
                row.pop("probe_layer", None), row.pop("probe_input", None)
                if probes_ and row["property"] in probes_:
                    row.update(probes_[row["property"]])
            by_layer[L] = rescore(r)
    if 7 in by_layer:
        fig_correlational(by_layer[7], FIG / "correlational_L7.png")
        fig_causal(by_layer[7], FIG / "causal_L7.png")
    if len(by_layer) >= 2:
        fig_layers(by_layer, FIG / "layers.png")
    ctrl = load("history_control.json")
    if ctrl:
        fig_history(ctrl, FIG / "history.png")

    # Markdown tables for RESULTS.md.
    for L, r in by_layer.items():
        print(f"\n### Layer {L} verdicts (calibrated causal thresholds {r['thresholds']})")
        print("| property | origin | feature | test F1 (P/R) | AUROC | probe layer / input | ablation KL median (ctrl) | KL win | target win | history corr | passes |")
        print("|" + "---|" * 11)
        for p in sorted(r["properties"], key=lambda p: -p["test_f1"]):
            c, pl, pi = p["causal"], p.get("probe_layer", {}), p.get("probe_input", {})
            passed = "all three" if p["passes"]["all"] else ", ".join(k for k, v in p["passes"].items() if v and k != "all") or "none"
            print(f"| {p['property']} | {p['origin']} | {p['feature']} | {p['test_f1']:.2f} ({p['test_precision']:.2f}/{p['test_recall']:.2f}) | "
                  f"{p['test_auroc']:.2f} | {pl.get('test_f1', float('nan')):.2f} / {pi.get('test_f1', float('nan')):.2f} | "
                  f"{c['kl_median']:.1e} ({c['control_kl_median']:.1e}) | {c['kl_win_rate']:.2f} | {c['target_win_rate']:.2f} | "
                  f"{p['history']['corr']:.3f} | {passed} |")
    for name in ("fidelity_table.md",):
        p = RES / name
        if p.exists():
            print("\n" + p.read_text())


if __name__ == "__main__":
    main()

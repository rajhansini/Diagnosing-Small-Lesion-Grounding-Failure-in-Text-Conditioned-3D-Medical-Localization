"""The three figures for Sections 7.13 and 7.14: what the window result was measuring, and what the
read-out rule was discarding.

fig_rq14_factorial          the window x stride grid, and the decomposition of the reported benefit
fig_rq14_pointing_grid      where the two mechanisms separate -- small ET across the whole grid
fig_rq15_accumulation       uniform against centre-weighted accumulation, and what it costs

Every number is recomputed from results/rq12_grounding_*_scores.csv at draw time, and colour follows
the project's fixed Okabe-Ito order, matching make_figures_appendix.py.
"""
import csv
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/net/projects/ranalab/rajhansini/nlp_project"
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

BLUE, ORANGE, GREEN, AMBER, PINK = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"
PURPLE, SKY = "#7B52AB", "#56B4E9"
INK, INK_MUTED, GRID = "#1a1a19", "#5c5c58", "#d8d8d4"

REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
GRID_EDGE = 128


def style_axes(ax, ylabel=None, xlabel=None, title=None, ygrid=True):
    """Apply the project's shared recessive-grid, no-top/right-spine treatment."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(ygrid, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", pad=8)


def caption(fig, text):
    """Place an explanatory caption below the axes, clear of the x-labels."""
    lines = text.count("\n") + 1
    inches_below = 0.42 + 0.16 * (lines - 1)
    fig.text(0.5, -inches_below / fig.get_figheight(), text,
             ha="center", fontsize=8.2, color=INK_MUTED, style="italic")


def save(fig, name):
    """Write a figure into figures/ at the project's fixed dpi and background."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def load(window, stride, weighting="uniform"):
    """Read one sweep condition's score table as a list of dicts, or None if never run."""
    suffix = "" if weighting == "uniform" else f"_{weighting}"
    path = os.path.join(RESULTS_DIR,
                        f"rq12_grounding_window{window}_stride{stride}{suffix}_scores.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def rule(rows, method):
    """Filter a multi-binarizer table down to one binarization rule."""
    return [r for r in rows if r["threshold_method"] == method]


def pooled(rows, method, key="dice"):
    """Mean of one column over every (patient, region) under one rule."""
    v = [float(r[key]) for r in rule(rows, method)]
    return sum(v) / len(v) if v else float("nan")


def hits(rows, region=None, size_bin=None):
    """Pointing-game hits and n under the plateau-centroid rule."""
    sub = rule(rows, "otsu")
    if region:
        sub = [r for r in sub if r["region"] == region]
    if size_bin:
        sub = [r for r in sub if r["size_bin"] == size_bin]
    return sum(int(r["centroid_hit"]) for r in sub), len(sub)


def n_windows(window, stride):
    """Sliding-window forward passes per volume."""
    return ((GRID_EDGE - window) // stride + 1) ** 3


# --------------------------------------------------------------------------------------------
# 1. The factorial, and what the reported benefit was actually measuring
# --------------------------------------------------------------------------------------------

def fig_rq14_factorial():
    """Separate the receptive-field effect from the sampling-density effect.

    Section 7.11's sweep set stride = window/2 at every point, so the two moved together and its
    +0.060 was attributed entirely to the window. Holding one fixed while varying the other shows
    most of it was the stride.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    # -- left: the grid, as two paths away from the published protocol --
    ax = axes[0]
    stride_path = [(32, 16), (32, 8), (32, 4)]
    window_path = [(32, 4), (16, 4), (12, 4), (8, 4)]
    diag_path = [(32, 16), (16, 8), (12, 6), (8, 4)]

    xs = [n_windows(w, s) for w, s in diag_path]
    ys = [pooled(load(w, s), "pct99") for w, s in diag_path]
    ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.6, color="#bcbcb8", zorder=2,
            label="the original sweep (stride = window/2)")

    xs = [n_windows(w, s) for w, s in stride_path]
    ys = [pooled(load(w, s), "pct99") for w, s in stride_path]
    ax.plot(xs, ys, marker="o", markersize=7, linewidth=2.4, color=ORANGE, zorder=4,
            label="stride varied, window fixed at 32³")

    xs = [n_windows(w, s) for w, s in window_path]
    ys = [pooled(load(w, s), "pct99") for w, s in window_path]
    ax.plot(xs, ys, marker="s", markersize=6.5, linewidth=2.4, color=BLUE, zorder=4,
            label="window varied, stride fixed at 4")

    # Explicit per-point offsets: several conditions sit on two of the three paths, and the two
    # densest cells (16³/4, 12³/4) are close enough in both axes to overlap a shared label position.
    offsets = {(32, 16): (26, -4), (32, 8): (0, -15), (32, 4): (-4, -16),
               (16, 8): (0, 10), (16, 4): (14, 6), (12, 6): (0, 10),
               (12, 4): (26, -2), (8, 4): (0, -16)}
    for (w, s), (dx, dy) in offsets.items():
        x, y = n_windows(w, s), pooled(load(w, s), "pct99")
        ax.annotate(f"{w}³/{s}", xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", fontsize=7.4, color=INK_MUTED)
    x0, y0 = n_windows(32, 16), pooled(load(32, 16), "pct99")
    ax.scatter([x0], [y0], s=200, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=5)
    ax.annotate("published\nprotocol", xy=(x0, y0), xytext=(10, 16), textcoords="offset points",
                fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(220, 7.5e4)
    ax.margins(y=0.14)
    style_axes(ax, ylabel="pooled Dice, top-1% threshold",
               xlabel="sliding-window forward passes per volume  (log)")
    ax.set_title("Two different ways to leave the published protocol", fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")

    # -- right: the decomposition --
    ax = axes[1]
    base16, base4 = load(32, 16), load(32, 4)
    d_sampling = pooled(load(32, 4), "pct99") - pooled(base16, "pct99")
    d_window = pooled(load(16, 4), "pct99") - pooled(base4, "pct99")
    d_original = pooled(load(12, 6), "pct99") - pooled(base16, "pct99")

    labels = ["reported as\na window effect\n(32³/16 → 12³/6)",
              "sampling density\nalone\n(32³, stride 16 → 4)",
              "receptive field\nalone\n(stride 4, 32³ → 16³)"]
    vals = [d_original, d_sampling, d_window]
    colors = ["#bcbcb8", ORANGE, BLUE]
    x = np.arange(3)
    ax.bar(x, vals, 0.6, color=colors)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 0.003, f"{v:+.4f}", ha="center", fontsize=9.5,
                color=INK, fontweight="bold")
    ax.axhline(0, color=INK_MUTED, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(vals) * 1.28)
    style_axes(ax, ylabel="Δ pooled Dice, top-1%")
    ax.set_title("The sampling term alone exceeds the whole reported effect",
                 fontsize=10, color=INK, pad=8)

    fig.suptitle("What “a smaller query window helps” was actually measuring",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.03)
    caption(fig,
            "Every condition in Section 7.11's sweep set stride = window/2 (grey), so shrinking the window always also multiplied the\n"
            "windows swept, and the two effects were never separated. Holding the window at 32³ and varying only the stride (orange) recovers\n"
            "+0.0761 Dice, p=1.9×10⁻²⁷ — more than the +0.0724 the original comparison credited to window size. Holding the stride at 4 and\n"
            "varying only the window (blue) is worth +0.0201, p=4.0×10⁻⁵, is not significant at 12³, and turns into a significant loss by 8³.")
    save(fig, "fig_rq14_factorial.png")


# --------------------------------------------------------------------------------------------
# 2. Where the two mechanisms come apart
# --------------------------------------------------------------------------------------------

def fig_rq14_pointing_grid():
    """The one place sampling density cannot substitute for a smaller receptive field.

    Dice improves under both moves, so it cannot distinguish them. The pointing game can: small
    enhancing tumor is at exactly chance under every 32^3 condition however finely the volume is
    sampled, and moves only when the window itself shrinks.
    """
    conds = [(32, 16), (32, 8), (32, 4), (16, 4), (12, 4), (8, 4)]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.5),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    # -- left: hit rate per region, across the grid --
    ax = axes[0]
    x = np.arange(len(conds))
    for reg, color in zip(REGIONS, [BLUE, ORANGE, GREEN]):
        ys = [hits(load(w, s), reg)[0] / hits(load(w, s), reg)[1] for w, s in conds]
        ax.plot(x, ys, marker="o", markersize=6, linewidth=2.0, color=color, label=reg)
    ys = [hits(load(w, s))[0] / hits(load(w, s))[1] for w, s in conds]
    ax.plot(x, ys, marker="D", markersize=5, linewidth=1.6, color=INK_MUTED,
            linestyle="--", label="pooled")
    ax.axvspan(-0.4, 2.4, color=ORANGE, alpha=0.08, zorder=0)
    ax.text(1.0, 0.04, "window fixed at 32³\nonly the stride changes", fontsize=8,
            color="#8a5a00", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}³/{s}" for w, s in conds], fontsize=8.8)
    ax.set_xlim(-0.4, len(conds) - 0.6)
    style_axes(ax, ylabel="pointing hit rate (plateau centroid)", xlabel="window / stride")
    ax.set_title("Sampling more finely lifts every region — except one bin",
                 fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left", ncol=2)

    # -- right: the ET-small bin alone --
    ax = axes[1]
    vals = [hits(load(w, s), "ET", "small")[0] for w, s in conds]
    n = hits(load(32, 16), "ET", "small")[1]
    colors = [ORANGE if w == 32 else BLUE for w, s in conds]
    ax.bar(x, vals, 0.6, color=colors)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 0.08, f"{v}/{n}", ha="center", fontsize=9,
                color=INK, fontweight="bold" if v == 0 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}³/{s}" for w, s in conds], fontsize=8.4)
    ax.set_ylim(0, max(vals) + 1.4)
    ax.set_yticks(range(0, max(vals) + 2))
    style_axes(ax, ylabel=f"small-ET patients pointed at (n={n})")
    ax.set_title("Zero at every 32³ condition, whatever the stride",
                 fontsize=10, color=INK, pad=8)
    ax.plot([], [], "s", color=ORANGE, label="32³ window")
    ax.plot([], [], "s", color=BLUE, label="smaller window")
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    fig.suptitle("The one thing sampling density cannot buy",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.03)
    caption(fig,
            "Dice rises under both interventions, so it cannot tell them apart; the threshold-free pointing game can. Sampling the volume\n"
            "eight times more finely at a fixed 32³ window lifts pooled pointing from 0.423 to 0.573 and helps every region — but leaves small\n"
            "enhancing tumor at 0 of 21, exactly where the 343-window protocol left it. That bin moves only when the window itself shrinks\n"
            "(2 of 21 at 12³, 4 of 21 at 8³), which is why Section 6.3's architectural reading survives the control that dissolves the rest.")
    save(fig, "fig_rq14_pointing_grid.png")


# --------------------------------------------------------------------------------------------
# 3. The accumulation rule
# --------------------------------------------------------------------------------------------

def fig_rq15_accumulation():
    """What the uniform read-out was discarding, at the published protocol and against its cost."""
    u, g = load(32, 16), load(32, 16, "gaussian")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5),
                             gridspec_kw={"width_ratios": [1.15, 1, 1.1]})

    # -- left: every rule, uniform against gaussian --
    ax = axes[0]
    rules = ["otsu", "pct90", "pct95", "pct99", "oracle_volume"]
    labels = ["Otsu", "top 10%", "top 5%", "top 1%", "oracle\nvolume"]
    x = np.arange(len(rules))
    width = 0.38
    uu = [pooled(u, r) for r in rules]
    gg = [pooled(g, r) for r in rules]
    ax.bar(x - width / 2, uu, width, color="#bcbcb8", label="uniform (published)")
    ax.bar(x + width / 2, gg, width, color=PURPLE, label="centre-weighted")
    for xi, (a, b) in enumerate(zip(uu, gg)):
        # Stagger the pair vertically where the two bars are nearly equal, so the labels do not run
        # into each other (top-10% differs by 0.002).
        lift = 0.030 if abs(a - b) < 0.02 else 0.008
        ax.text(xi - width / 2, a + lift, f"{a:.3f}", ha="center", fontsize=7.6, color=INK_MUTED)
        ax.text(xi + width / 2, b + 0.008, f"{b:.3f}", ha="center", fontsize=7.6,
                color=GREEN if b > a else ORANGE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylim(0, 0.53)
    style_axes(ax, ylabel="pooled Dice")
    ax.set_title("Every calibrated rule improves; Otsu alone falls",
                 fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    # -- centre: the plateau, and the bin it was blocking --
    ax = axes[1]
    tu = statistics.median([int(r["peak_tie_count"]) for r in rule(u, "otsu")])
    tg = statistics.median([int(r["peak_tie_count"]) for r in rule(g, "otsu")])
    ax.bar([0, 1], [tu, tg], 0.55, color=["#bcbcb8", PURPLE])
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e4)
    for xi, v in enumerate([tu, tg]):
        ax.text(xi, v * 1.5, f"{int(v):,}", ha="center", fontsize=11,
                color=INK, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["uniform", "centre-weighted"], fontsize=9)
    style_axes(ax, ylabel="median tied-maximum plateau (voxels, log)")
    ax.set_title("The resolution bottleneck, removed", fontsize=10, color=INK, pad=8)
    eu = hits(u, "ET", "small")
    eg = hits(g, "ET", "small")
    ax.text(0.5, 3.2e3, f"small-ET pointed at:  {eu[0]}/{eu[1]}  →  {eg[0]}/{eg[1]}",
            ha="center", fontsize=9.2, color=GREEN, fontweight="bold")

    # -- right: cost --
    ax = axes[2]
    # Two pairs share an x, so every label gets an explicit offset and alignment rather than a
    # shared one: the cheap pair sits at 343 windows and the dense pair at 24,389.
    pts = [("32³/16\nuniform", load(32, 16), 32, 16, "#bcbcb8", "o", (12, -26), "left"),
           ("16³/4\nuniform", load(16, 4), 16, 4, BLUE, "o", (-12, -26), "right"),
           ("32³/16\ncentre-weighted", load(32, 16, "gaussian"), 32, 16, PURPLE, "*", (14, 8), "left"),
           ("16³/4\ncentre-weighted", load(16, 4, "gaussian"), 16, 4, PURPLE, "s", (-12, 6), "right")]
    for name, rows, w, s, color, marker, off, ha in pts:
        if rows is None:
            continue
        size = 460 if marker == "*" else 150
        ax.scatter([n_windows(w, s)], [pooled(rows, "oracle_volume")], s=size, color=color,
                   marker=marker, zorder=3, edgecolor="white", linewidth=1.4)
        ax.annotate(name, xy=(n_windows(w, s), pooled(rows, "oracle_volume")),
                    xytext=off, textcoords="offset points", ha=ha, fontsize=8, color=INK)
    ax.annotate("", xy=(n_windows(32, 16) * 1.35, pooled(load(32, 16, "gaussian"), "oracle_volume")),
                xytext=(n_windows(16, 4) * 0.72, pooled(load(16, 4), "oracle_volume")),
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.6,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(2.6e3, 0.398, "better, and 71× cheaper", fontsize=8.8, color=GREEN,
            fontweight="bold", ha="center")
    ax.set_xscale("log")
    ax.set_xlim(150, 1.6e5)
    ax.set_ylim(0.26, 0.545)
    style_axes(ax, ylabel="pooled Dice, oracle volume",
               xlabel="forward passes per volume  (log)")
    ax.set_title("The read-out fix is free", fontsize=10, color=INK, pad=8)

    fig.suptitle("The block was the read-out, not the model",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.03)
    caption(fig,
            "`sliding_window_heatmap` gave every voxel a window covers the same scalar score, which is what makes the heatmap piecewise-constant\n"
            "over stride³ blocks — the plateau Section 6.3 read as the model's resolution limit. Weighting each window's contribution toward its own\n"
            "centre instead, with the same model, the same windows and no additional forward passes, collapses that plateau from 4,096 voxels to 1,\n"
            "lifts oracle Dice by +0.145 (p=1.7×10⁻²⁷) and takes small enhancing tumor from 0 of 21 to 7 of 21. Otsu moves the other way for the\n"
            "third time in this report, because a smooth high-entropy heatmap is exactly what its between-class-variance criterion handles worst.")
    save(fig, "fig_rq15_accumulation.png")


def main():
    """Draw the three RQ14/RQ15 figures."""
    fig_rq14_factorial()
    fig_rq14_pointing_grid()
    fig_rq15_accumulation()


if __name__ == "__main__":
    main()

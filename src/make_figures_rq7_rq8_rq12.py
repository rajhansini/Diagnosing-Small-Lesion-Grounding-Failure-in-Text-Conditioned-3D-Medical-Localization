"""Figures for the late experiments: RQ7 (text encoder), RQ8 (compositionality), RQ12 (threshold), P'.

Four figures, each carrying one claim that the earlier figures cannot show:

  fig_rq12_threshold_reversal  the project's central correction -- the smaller window's Dice win
                               exists under Otsu and inverts under every better-calibrated rule.
  fig_rq7_encoder              text-encoder effects against the retraining noise floor, per seed,
                               making visible that three of four "effects" change sign across runs.
  fig_rq8_compositionality     how little the heatmap moves when the query's meaning is destroyed.
  fig_pprime_size              whether a fully supervised model collapses on small lesions too.

Colour follows the project's existing region identity (Okabe-Ito blue/orange/green, colourblind-safe
and validated: all adjacent and all-pairs CVD separations clear the deltaE>=8 target). Hues are
assigned in fixed order and never cycled; text stays in ink colours so identity is never carried by
colour alone; grids are recessive.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"

# Fixed categorical order, shared with every other figure in the project.
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
REGION_COLORS = {"ET": BLUE, "TC": ORANGE, "WT": GREEN}
INK, INK_MUTED, GRID = "#1a1a19", "#5c5c58", "#d8d8d4"

REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
SEEDS = [0, 1, 2]


def load(filename):
    """Read a results CSV into dict rows, or return None if it does not exist."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def style_axes(ax, ylabel=None, xlabel=None):
    """Apply the shared recessive-grid, no-top/right-spine treatment."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
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


def paired_mean(rows_a, rows_b, key_filter=None):
    """Mean Dice of two arms over the (patient, region) keys they share."""
    def idx(rows):
        return {(r["patient_id"], r["region"]): float(r["dice"]) for r in rows
                if key_filter is None or key_filter(r)}
    a, b = idx(rows_a), idx(rows_b)
    keys = sorted(set(a) & set(b))
    return float(np.mean([a[k] for k in keys])), float(np.mean([b[k] for k in keys])), len(keys)


def fig_rq12_threshold_reversal():
    """Draw the Otsu-only nature of the smaller window's advantage."""
    w32, w8 = load("rq12_grounding_window32_scores.csv"), load("rq12_grounding_window8_scores.csv")
    if not w32 or not w8:
        print("  skipped fig_rq12_threshold_reversal (missing RQ12 CSVs)")
        return
    methods = [("otsu", "Otsu\n(published protocol)"), ("pct90", "top 10%"), ("pct95", "top 5%"),
               ("pct99", "top 1%\n(deployable)"), ("oracle_volume", "oracle volume\n(diagnostic)")]

    a_vals, b_vals = [], []
    for m, _ in methods:
        a, b, _ = paired_mean(w32, w8, lambda r, m=m: r["threshold_method"] == m)
        a_vals.append(a)
        b_vals.append(b)

    x = np.arange(len(methods))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(x - w / 2 - 0.01, a_vals, w, label="32³ window (baseline)", color=BLUE)
    ax.bar(x + w / 2 + 0.01, b_vals, w, label="8³ window (the “fix”)", color=ORANGE)

    for xi, (a, b) in enumerate(zip(a_vals, b_vals)):
        better = b > a
        ax.annotate(f"{b - a:+.3f}", xy=(xi, max(a, b) + 0.012), ha="center", fontsize=9,
                    color=INK if better else INK_MUTED,
                    fontweight="bold" if better else "normal")
    style_axes(ax, ylabel="Mean Dice (213 patient×region pairs)")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in methods], fontsize=9, color=INK)
    ax.set_ylim(0, max(max(a_vals), max(b_vals)) * 1.22)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax.set_title("The smaller window wins only under Otsu — and loses under every better-calibrated rule",
                 fontsize=11, color=INK, pad=12, loc="left")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_rq12_threshold_reversal.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_rq7_encoder():
    """Draw per-seed RQ7 deltas against the baseline's own retraining noise floor."""
    variants = [("bertbase", "BERT-base\n(general domain)"),
                ("randbert", "random-init BERT\n(frozen)"),
                ("randvec_ortho", "4 random orthonormal\nvectors"),
                ("randvec_aniso", "random vectors,\nPubMedBERT geometry")]

    def base_file(s):
        return "rq1_localization_scores.csv" if s == 0 else f"rq1_localization_scores_seed{s}.csv"

    def var_file(k, s):
        return (f"rq7_{k}_localization_scores.csv" if s == 0
                else f"rq7_{k}_seed{s}_localization_scores.csv")

    bases = {s: load(base_file(s)) for s in SEEDS}
    pooled = [float(np.mean([float(r["dice"]) for r in bases[s]])) for s in SEEDS]
    noise = float(np.std(pooled, ddof=1))

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.axvspan(-noise, noise, color=GRID, alpha=0.75, zorder=0)
    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)

    for yi, (key, label) in enumerate(variants):
        deltas = []
        for s in SEEDS:
            v = load(var_file(key, s))
            a, b, _ = paired_mean(bases[s], v)
            deltas.append(b - a)
        consistent = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        col = ORANGE if consistent else BLUE
        ax.scatter(deltas, [yi] * 3, s=58, color=col, zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.scatter([np.mean(deltas)], [yi], s=190, marker="|", color=col, zorder=4, linewidth=2.4)
        verdict = "consistent across runs" if consistent else "sign flips → noise"
        ax.annotate(verdict, xy=(0.088, yi), fontsize=8.5,
                    color=INK if consistent else INK_MUTED, va="center")

    style_axes(ax, xlabel="Change in pooled Dice vs. PubMedBERT (one dot per training run)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([lbl for _, lbl in variants], fontsize=9, color=INK)
    ax.set_ylim(-0.45, len(variants) + 0.35)
    ax.set_xlim(-0.075, 0.16)
    ax.annotate(f"grey band = retraining noise floor (±{noise:.4f})",
                xy=(-0.072, len(variants) + 0.05), fontsize=8.5, color=INK_MUTED, va="center")
    ax.set_title("Only one text-encoder effect survives being retrained under three seeds",
                 fontsize=11, color=INK, pad=12, loc="left")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_rq7_encoder.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_rq8_compositionality():
    """Draw how little the heatmap moves when the query's meaning is destroyed."""
    rows = load("rq8_compositionality_scores.csv")
    if not rows:
        print("  skipped fig_rq8_compositionality (missing CSV)")
        return
    conds = [("negated", "negated\n(“no enhancing tumor”)"), ("shuffled", "shuffled\nword order"),
             ("swapped", "swapped\nreferent"), ("generic", "generic filler\n(no anatomy)")]

    x = np.arange(len(conds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    for ri, region in enumerate(REGIONS):
        vals = []
        for c, _ in conds:
            sel = [float(r["heatmap_spearman_vs_original"]) for r in rows
                   if r["condition"] == c and r["region"] == region]
            vals.append(float(np.mean(sel)))
        ax.bar(x + (ri - 1) * (w + 0.015), vals, w, label=region, color=REGION_COLORS[region])

    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("ρ = 1.0: query manipulation changed nothing at all", xy=(-0.42, 1.012),
                fontsize=8.5, color=INK_MUTED)
    style_axes(ax, ylabel="Spearman ρ of heatmap vs. the true clinical query")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in conds], fontsize=9, color=INK)
    ax.set_ylim(0, 1.14)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), labelcolor=INK, title=None)
    ax.set_title("Destroying the query's meaning barely moves the model's spatial response",
                 fontsize=11, color=INK, pad=12, loc="left")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_rq8_compositionality.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_pprime_size():
    """Draw the size-stratified comparison of supervised P' against text-conditioned RQ1."""
    pp = load("pprime_supervised_scores.csv")
    rq1 = load("rq1_localization_scores.csv")
    if not pp or not rq1:
        print("  skipped fig_pprime_size (P' scores not available yet)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9), sharey=True)
    for ax, region in zip(axes, REGIONS):
        pv = [float(np.mean([float(r["dice"]) for r in pp
                             if r["region"] == region and r["size_bin"] == b])) for b in BINS]
        tv = [float(np.mean([float(r["dice"]) for r in rq1
                             if r["region"] == region and r["size_bin"] == b])) for b in BINS]
        x = np.arange(3)
        w = 0.36
        ax.bar(x - w / 2 - 0.01, pv, w, color=REGION_COLORS[region],
               label="P′ supervised U-Net")
        ax.bar(x + w / 2 + 0.01, tv, w, color=REGION_COLORS[region], alpha=0.42,
               label="RQ1 text-conditioned")
        style_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(BINS, fontsize=9, color=INK)
        ratio_p = pv[2] / pv[0] if pv[0] > 0 else float("nan")
        ratio_t = tv[2] / tv[0] if tv[0] > 0 else float("nan")
        ax.set_title(f"{region}\nL/S  {ratio_p:.1f}× vs {ratio_t:.1f}×",
                     fontsize=10, color=INK, pad=8)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Mean Dice", color=INK, fontsize=10)
    # Neutral legend swatches: each panel is drawn in its own region hue, so a coloured key taken
    # from the first panel would mislabel the other two. Fill/opacity carries the distinction.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=INK_MUTED, label="P′ supervised U-Net"),
               Patch(facecolor=INK_MUTED, alpha=0.42, label="RQ1 text-conditioned")]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
    fig.suptitle("P′ validation: the same data, split and metric under dense supervision — "
                 "and how its size collapse compares",
                 fontsize=11, color=INK, x=0.012, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = os.path.join(FIGURES_DIR, "fig_pprime_size.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    """Generate all four late-experiment figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_rq12_threshold_reversal()
    fig_rq7_encoder()
    fig_rq8_compositionality()
    fig_pprime_size()


if __name__ == "__main__":
    main()

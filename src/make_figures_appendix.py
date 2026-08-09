"""Nine appendix figures, each carrying a measurement the result tables held but no figure showed.

The project's twenty-nine existing figures cover the dataset, the core result, every ablation and
every correction. What none of them covered were nine things that were computed, stored, and never
looked at:

  fig_iou_dice                  the second overlap metric every CSV has carried since the first run.
                                If the size collapse is the model's and not Dice's algebra, IoU has
                                to show it too.
  fig_per_patient_spread        every bar in this report is a mean over 20-32 skewed, near-zero
                                per-patient scores. This is what is underneath them, and it is the
                                reason the tests are Wilcoxon rather than t-tests.
  fig_effect_sizes              Section 5.1 promised rank-biserial effect sizes and bootstrap
                                intervals alongside every p-value. All 171 were computed; none was
                                ever shown.
  fig_bh_correction             the Benjamini-Hochberg step-up made visible, including the single
                                test in the whole project that the correction demoted.
  fig_checkpoint_sensitivity    RQ7 was evaluated at both the last and the best-validation
                                checkpoint. The comparison was never reported, and it demotes the
                                same result seed replication demoted, independently.
  fig_pointing_rules            three tie-breaking rules were written to the RQ12 tables; the report
                                uses two. The third needs a block-level chance baseline, and gives
                                the sharpest available statement of what small-ET failure *is*.
  fig_otsu_selected_fraction    the report explains the Otsu/calibrated sign disagreement by appeal
                                to histogram shape; the predicted-volume column measures it.
  fig_window_curve_per_region   Section 7.11's pooled 12-16 optimum is an average over three regions
                                that want three different windows.
  fig_cost_benefit              what the recommended intervention costs, against what it buys.

Every number is recomputed from results/*.csv or the preprocessed masks at draw time. Colour follows
the project's fixed Okabe-Ito order, matching make_figures_supplementary.py.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import region_mask

ROOT = "/net/projects/ranalab/rajhansini/nlp_project"
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
PREPROCESSED_DIR = os.path.join(ROOT, "data", "preprocessed")

BLUE, ORANGE, GREEN, AMBER, PINK = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"
PURPLE, SKY = "#7B52AB", "#56B4E9"
REGION_COLORS = {"ET": BLUE, "TC": ORANGE, "WT": GREEN}
INK, INK_MUTED, GRID = "#1a1a19", "#5c5c58", "#d8d8d4"

REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
REGION_LABELS = {"ET": "ET  (enhancing tumor)", "TC": "TC  (tumor core)", "WT": "WT  (whole tumor)"}

RULES = ["otsu", "pct90", "pct95", "pct99", "oracle_volume"]
RULE_LABELS = {"otsu": "Otsu", "pct90": "top 10%", "pct95": "top 5%",
               "pct99": "top 1%", "oracle_volume": "oracle volume"}
RULE_COLORS = {"otsu": ORANGE, "pct90": "#bcbcb8", "pct95": SKY,
               "pct99": BLUE, "oracle_volume": GREEN}

SWEEP = [("32", "16"), ("16", "8"), ("12", "6"), ("8", "4"), ("8", "8"), ("6", "6")]
OVERLAPPED = SWEEP[:4]
GRID_EDGE = 128
TOTAL_VOXELS = GRID_EDGE ** 3


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


def res(name):
    """Load a results CSV by filename."""
    return pd.read_csv(os.path.join(RESULTS_DIR, name))


def sweep_csv(window, stride):
    """Load one condition of the RQ12 window sweep."""
    return res(f"rq12_grounding_window{window}_stride{stride}_scores.csv")


def rule(df, method):
    """Filter a multi-binarizer table down to one rule."""
    return df[df.threshold_method == method]


def caption(fig, text):
    """Place an explanatory caption below the axes, clear of the x-labels.

    The offset scales with the line count because savefig(bbox_inches="tight") crops to content:
    placing the block too close overlaps the axis labels, placing it too far only adds whitespace
    that the crop then removes.
    """
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


# --------------------------------------------------------------------------------------------
# 1. The second overlap metric
# --------------------------------------------------------------------------------------------

def fig_iou_dice():
    """Does the size collapse survive being measured by IoU instead of Dice?

    Dice and IoU are monotone in each other per patient, so this is not an independent metric in
    the way the pointing game is. What it does rule out is that the *ratios* the report quotes are
    an artifact of Dice's particular algebra: IoU is harsher on partial overlap, so if anything it
    should exaggerate the collapse, and it does -- mildly, and in every region.
    """
    base = res("rq1_localization_scores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    width = 0.36
    x = np.arange(9)
    labels, dice_v, iou_v = [], [], []
    for reg in REGIONS:
        for b in BINS:
            sub = base[(base.region == reg) & (base.size_bin == b)]
            labels.append(f"{reg}\n{b}")
            dice_v.append(sub.dice.mean())
            iou_v.append(sub.iou.mean())
    ax.bar(x - width / 2, dice_v, width, color=BLUE, label="Dice")
    ax.bar(x + width / 2, iou_v, width, color=SKY, label="IoU")
    for i, (d, u) in enumerate(zip(dice_v, iou_v)):
        ax.text(i - width / 2, d + 0.004, f"{d:.3f}", ha="center", fontsize=7, color=INK_MUTED)
        ax.text(i + width / 2, u + 0.004, f"{u:.3f}", ha="center", fontsize=7, color=INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    for xv in (2.5, 5.5):
        ax.axvline(xv, color=GRID, linewidth=1.0)
    style_axes(ax, ylabel="mean score on held-out patients")
    ax.set_title("The same monotone collapse under both metrics", fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    arms = [("RQ2", res("rq2_localization_scores.csv")),
            ("RQ3", res("rq3_localization_scores.csv")),
            ("RQ3b", rule(sweep_csv("16", "8"), "otsu")),
            ("RQ3c", res("rq3c_window12_scores.csv")),
            ("RQ4", res("rq4_localization_scores.csv")),
            ("RQ5", res("rq5_localization_scores.csv")),
            ("RQ6", res("rq6_localization_scores.csv"))]
    dd, du, names = [], [], []
    for name, other in arms:
        m = base.merge(other[["patient_id", "region", "dice", "iou"]],
                       on=["patient_id", "region"], suffixes=("_b", "_o"))
        dd.append((m.dice_o - m.dice_b).mean())
        du.append((m.iou_o - m.iou_b).mean())
        names.append(name)
    for name, a, b in zip(names, dd, du):
        color = GREEN if a > 0 else ORANGE
        ax.scatter([a], [b], s=110, color=color, zorder=3, edgecolor="white", linewidth=1.3)
        ax.annotate(name, xy=(a, b), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=8.4, color=INK)
    lim = max(max(np.abs(dd)), max(np.abs(du))) * 1.35
    ax.plot([-lim, lim], [-lim, lim], color=GRID, linewidth=1.0, linestyle="--", zorder=1)
    ax.axhline(0, color=INK_MUTED, linewidth=0.9)
    ax.axvline(0, color=INK_MUTED, linewidth=0.9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    style_axes(ax, ylabel="pooled ΔIoU vs baseline", xlabel="pooled ΔDice vs baseline")
    ax.set_title("7 of 7 arms keep their sign", fontsize=10, color=INK, pad=8)

    fig.suptitle("The verdicts do not depend on which overlap metric is used",
                 fontsize=12, fontweight="bold", color=INK, y=1.02)
    caption(fig,
             "IoU = Dice / (2 − Dice) per patient, so no comparison can reverse; what this rules out is that the\n"
             "large/small ratios are an artifact of Dice's algebra. IoU makes them slightly larger (ET 16.4× vs 15.0×,\n"
             "TC 10.0× vs 9.2×, WT 5.6× vs 4.9×), so the reported figures are the conservative ones.",)
    save(fig, "fig_iou_dice.png")


# --------------------------------------------------------------------------------------------
# 2. What is underneath every bar in this report
# --------------------------------------------------------------------------------------------

def fig_per_patient_spread():
    """Show the per-patient distributions the report's means are computed over.

    Two things this makes visible that no mean-and-error-bar figure can. First, the distributions
    are bounded at zero, right-skewed and pile up near it in the small bins -- which is the reason
    every test in this report is a Wilcoxon signed-rank rather than a paired t-test. Second, the
    intervention's benefit is a shift of the whole distribution, not a few patients moving a lot.
    """
    base = res("rq1_localization_scores.csv")
    better = rule(sweep_csv("12", "6"), "pct99")
    base_p = rule(sweep_csv("32", "16"), "pct99")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), sharey=True)
    rng = np.random.default_rng(0)
    for ax, reg in zip(axes, REGIONS):
        for i, b in enumerate(BINS):
            for j, (df, color, label) in enumerate(
                    [(base_p, INK_MUTED, "32³ window"), (better, REGION_COLORS[reg], "12³ window")]):
                vals = df[(df.region == reg) & (df.size_bin == b)].dice.values
                xpos = i + (j - 0.5) * 0.34
                ax.scatter(xpos + rng.normal(0, 0.035, len(vals)), vals, s=13, alpha=0.55,
                           color=color, edgecolor="none", zorder=2,
                           label=label if (i == 0 and reg == "ET") else None)
                ax.plot([xpos - 0.13, xpos + 0.13], [np.median(vals)] * 2,
                        color=color, linewidth=2.6, zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels(BINS, fontsize=9)
        style_axes(ax, ylabel="per-patient Dice (top-1% threshold)" if reg == "ET" else None)
        ax.set_title(REGION_LABELS[reg], fontsize=10, color=INK, pad=8)
    axes[0].legend(frameon=False, fontsize=8.6, loc="upper left")
    fig.suptitle("Every mean in this report is an average over a bounded, right-skewed, "
                 "20-to-32-patient distribution",
                 fontsize=12, fontweight="bold", color=INK, y=1.03)
    caption(fig,
             "Grey is the published 32³ protocol, colour the recommended 12³ window; thick bars are bin medians.\n"
             "The distributions are bounded at zero and pile up against it in the small bins — which is why every test in\n"
             "this report is a paired Wilcoxon signed-rank rather than a t-test, and why effect sizes are reported with it.",)
    save(fig, "fig_per_patient_spread.png")


# --------------------------------------------------------------------------------------------
# 3. Effect sizes and intervals for the whole family
# --------------------------------------------------------------------------------------------

def fig_effect_sizes():
    """The 171 tests plotted as magnitude against significance, with the useless corner marked.

    A third of the BH-significant tests in this project move Dice by less than 0.01, and almost all
    of those are in the small bins where baseline Dice is 0.010-0.057 and the paired variance is
    tiny. Reporting p-values alone would have made those look like findings.
    """
    fam = res("full_family_statistics.csv")
    # Two rows rather than two columns: the forest plot's arm labels are long, and side by side they
    # run back into the scatter's plotting area.
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 9.0),
                             gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.42})

    ax = axes[0]
    sig = fam.q_pooled < 0.05
    tiny = fam.delta.abs() < 0.01
    ax.axhspan(1e-4, 0.01, color=AMBER, alpha=0.10, zorder=0)
    ax.text(1.5e-11, 0.0072, "  significant but negligible  (|Δ| < 0.01 Dice)", fontsize=8.2,
            color="#8a6d0b", va="center", ha="left")
    ax.axvline(0.05, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.text(0.062, 0.30, "BH q = 0.05", fontsize=8, color=INK_MUTED, rotation=90, va="top")
    ax.scatter(fam.q_pooled[sig & ~tiny], fam.delta.abs()[sig & ~tiny], s=26, color=BLUE,
               alpha=0.75, edgecolor="none", label=f"significant, substantive ({(sig & ~tiny).sum()})")
    ax.scatter(fam.q_pooled[sig & tiny], fam.delta.abs()[sig & tiny], s=26, color=AMBER,
               alpha=0.85, edgecolor="none", label=f"significant, negligible ({(sig & tiny).sum()})")
    ax.scatter(fam.q_pooled[~sig], fam.delta.abs()[~sig], s=26, color="#bcbcb8",
               alpha=0.8, edgecolor="none", label=f"not significant ({(~sig).sum()})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-11, 1.4)
    style_axes(ax, ylabel="|Δ Dice|  (log)", xlabel="BH-adjusted q-value  (log)")
    ax.set_title("A third of the significant results are too small to matter",
                 fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.2, loc="lower left")

    ax = axes[1]
    order = ["RQ12 (8^3 window, pct99)", "RQ3c (8^3 window)", "RQ3b (16^3 window)",
             "RQ2 (size-conditioned text)", "RQ5 (naturalistic text)", "RQ8 (shuffled word order)",
             "RQ7 (random, PubMedBERT geometry)", "RQ3 (multi-scale ensemble)",
             "RQ6 (uniformity fix)", "RQ4 (scale-matched)"]
    short = {"RQ12 (8^3 window, pct99)": "RQ12  8³ window, top-1%",
             "RQ3c (8^3 window)": "RQ3c  8³ window",
             "RQ3b (16^3 window)": "RQ3b  16³ window",
             "RQ2 (size-conditioned text)": "RQ2   size-conditioned text",
             "RQ5 (naturalistic text)": "RQ5   naturalistic text",
             "RQ8 (shuffled word order)": "RQ8   shuffled word order",
             "RQ7 (random, PubMedBERT geometry)": "RQ7   random @ PubMedBERT geom.",
             "RQ3 (multi-scale ensemble)": "RQ3   multi-scale ensemble",
             "RQ6 (uniformity fix)": "RQ6   uniformity fix",
             "RQ4 (scale-matched)": "RQ4   scale-matched retraining"}
    ax.axvline(0, color=INK_MUTED, linewidth=1.0)
    for i, comp in enumerate(order):
        sub = fam[fam.comparison == comp]
        y = len(order) - 1 - i
        lo, hi = sub.ci_lo.min(), sub.ci_hi.max()
        mean = sub.delta.mean()
        color = GREEN if mean > 0 else ORANGE
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.2, alpha=0.45, solid_capstyle="round")
        ax.scatter(sub.delta, [y] * len(sub), s=22, color=color, alpha=0.9,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter([mean], [y], s=80, marker="|", color=INK, zorder=4, linewidth=2.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short[c] for c in reversed(order)], fontsize=8.4, family="monospace")
    style_axes(ax, xlabel="Δ Dice vs the RQ1 baseline  (9 region×bin tests per row)", ygrid=False)
    ax.set_title("Every arm's nine tests, with bootstrap span", fontsize=10, color=INK, pad=8)

    fig.suptitle("Effect sizes and intervals for all 171 paired tests — the columns the report "
                 "computed and never showed",
                 fontsize=12.5, fontweight="bold", color=INK, y=0.965)
    caption(fig,
             "Top: each of the 171 tests by adjusted significance against absolute effect. Bottom: per-arm spread, dots are the nine\n"
             "region×bin deltas, the bar spans the union of their 10,000-sample percentile bootstrap intervals, the tick is the mean.\n"
             "Among the 144 BH-significant tests, 75% carry a rank-biserial effect size above 0.8 — but 47 move Dice by under 0.01.",)
    save(fig, "fig_effect_sizes.png")


# --------------------------------------------------------------------------------------------
# 4. The correction itself
# --------------------------------------------------------------------------------------------

def fig_bh_correction():
    """Draw the Benjamini-Hochberg step-up that every significance claim in the report passes through.

    Also answers a question the report raised and never resolved: the family grew as experiments were
    added, so does a test's verdict depend on experiments run after it? Corrected against the
    27-test family as it stood when RQ3b was written, versus the final 171, no RQ3b verdict changes
    -- and the larger family is if anything marginally more permissive, because BH's threshold
    depends on the whole p-value distribution rather than only on the count.
    """
    fam = res("full_family_statistics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))

    ax = axes[0]
    p = np.sort(fam.p_raw.values)
    m = len(p)
    i = np.arange(1, m + 1)
    ax.plot(i, p, color=BLUE, linewidth=1.8, label="observed p-values, sorted")
    ax.plot(i, i / m * 0.05, color=ORANGE, linewidth=1.6, linestyle="--",
            label="BH threshold  (i/m)·α,  α = 0.05")
    ax.axhline(0.05, color=INK_MUTED, linewidth=1.0, linestyle=":",
               label="uncorrected α = 0.05")
    n_sig = (fam.q_pooled < 0.05).sum()
    ax.axvline(n_sig, color=GREEN, linewidth=1.4)
    ax.text(n_sig - 3, 0.6, f"{n_sig} rejected", fontsize=8.4, color=GREEN,
            rotation=90, ha="right", va="top")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1.6)
    ax.set_xlim(0, m)
    style_axes(ax, ylabel="p-value  (log)", xlabel="rank i among m = 171 tests")
    ax.set_title("145 significant raw → 144 after correction", fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")

    ax = axes[1]
    early = fam[fam.comparison.isin(["RQ2 (size-conditioned text)", "RQ3 (multi-scale ensemble)",
                                     "RQ3b (16^3 window)"])]
    pe = np.sort(early.p_raw.values)
    me = len(pe)
    qe = np.minimum.accumulate((pe * me / np.arange(1, me + 1))[::-1])[::-1]
    lookup = dict(zip(pe, qe))
    rq3b = fam[fam.comparison == "RQ3b (16^3 window)"].copy()
    rq3b["q_early"] = rq3b.p_raw.map(lookup)
    labels = [f"{r.region} {r['bin']}" for _, r in rq3b.iterrows()]
    y = np.arange(len(rq3b))
    # A dumbbell rather than bars: the q-values span seven orders of magnitude, and bars anchored at
    # zero on a log axis render the smallest ones as nothing at all.
    for yi, (_, r) in zip(y, rq3b.iterrows()):
        ax.plot([r.q_early, r.q_pooled], [yi, yi], color=GRID, linewidth=1.6, zorder=1)
    ax.scatter(rq3b.q_early, y, s=52, color="#bcbcb8", zorder=3, edgecolor="white", linewidth=1.0)
    ax.scatter(rq3b.q_pooled, y, s=52, color=BLUE, zorder=3, edgecolor="white", linewidth=1.0)
    top = len(rq3b) - 1
    ax.annotate("family of 27\n(when RQ3b was run)", xy=(rq3b.q_early.iloc[top], top),
                xytext=(0, 16), textcoords="offset points", ha="center",
                fontsize=7.8, color=INK_MUTED)
    ax.annotate("family of 171\n(final)", xy=(rq3b.q_pooled.iloc[top], top),
                xytext=(0, -30), textcoords="offset points", ha="center",
                fontsize=7.8, color=BLUE)
    ax.axvline(0.05, color=ORANGE, linewidth=1.4, linestyle="--")
    ax.text(0.055, 0.1, "q = 0.05", fontsize=8.2, color=ORANGE)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xscale("log")
    ax.set_xlim(1e-11, 3)
    ax.set_ylim(-0.9, len(rq3b) + 0.2)
    style_axes(ax, xlabel="BH-adjusted q-value  (log)", ygrid=False)
    ax.set_title("Does a later experiment change an earlier verdict? No.",
                 fontsize=10, color=INK, pad=8)

    fig.suptitle("The multiple-comparisons correction every claim in this report passes through",
                 fontsize=12, fontweight="bold", color=INK, y=1.03)
    caption(fig,
             "Left: BH rejects the largest i for which p₍ᵢ₎ ≤ (i/m)·α. One test in the project is significant raw and not after\n"
             "correction — RQ3b, ET large, p = 0.048 → q = 0.057 — and the report flags it rather than rounding it up.\n"
             "Right: the pooled family grew as experiments accumulated; no RQ3b verdict depends on which version it is corrected against.",)
    save(fig, "fig_bh_correction.png")


# --------------------------------------------------------------------------------------------
# 5. Checkpoint selection as a second fragility probe
# --------------------------------------------------------------------------------------------

def fig_checkpoint_sensitivity():
    """RQ7's four conditions evaluated at both checkpoints of the same trained runs.

    The report demotes RQ7's single-seed result by retraining under three seeds. This is the
    independent version of the same check: hold the training runs fixed and vary only which epoch's
    weights are scored. The +0.038 'random-init BERT beats PubMedBERT' effect that carried a
    within-run p of 6e-35 is -0.0008 at the best-validation checkpoint of the very same run.
    """
    base = res("rq1_localization_scores.csv").dice.mean()
    variants = [("BERT-base", "bertbase"), ("random-init BERT", "randbert"),
                ("random orthonormal", "randvec_ortho"), ("random @ PubMedBERT geom.", "randvec_aniso")]
    noise = 0.0044   # retraining noise floor, from analyze_rq7_multiseed.py

    fig, ax = plt.subplots(figsize=(10.4, 4.5))
    ax.axhspan(-noise, noise, color="#bcbcb8", alpha=0.35, zorder=0)
    ax.text(3.46, noise, "  retraining noise floor (±0.0044)", fontsize=8, color=INK_MUTED,
            va="bottom", ha="right")
    ax.axhline(0, color=INK_MUTED, linewidth=1.0)

    width = 0.34
    x = np.arange(len(variants))
    lasts, bests = [], []
    for _, v in variants:
        lasts.append(res(f"rq7_{v}_localization_scores.csv").dice.mean() - base)
        bests.append(res(f"rq7_{v}_best_localization_scores.csv").dice.mean() - base)
    ax.bar(x - width / 2, lasts, width, color=BLUE, label="last-epoch checkpoint (the reported comparison)")
    ax.bar(x + width / 2, bests, width, color=AMBER, label="best-validation checkpoint, same runs")
    for xi, (a, b) in enumerate(zip(lasts, bests)):
        ax.text(xi - width / 2, a + (0.0015 if a > 0 else -0.003), f"{a:+.4f}",
                ha="center", fontsize=8, color=INK)
        ax.text(xi + width / 2, b + (0.0015 if b > 0 else -0.003), f"{b:+.4f}",
                ha="center", fontsize=8, color=INK)
    ax.annotate("p ≈ 6×10⁻³⁵ within its run —\nand gone at the other checkpoint",
                xy=(1 + width / 2 + 0.04, lasts[1] * 0.86), xytext=(1.45, 0.050),
                fontsize=8.4, color=ORANGE, ha="left",
                arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in variants], fontsize=9)
    ax.set_ylim(-0.034, 0.064)
    style_axes(ax, ylabel="pooled Δ Dice vs PubMedBERT baseline")
    ax.set_title("A second, independent probe of the same fragility: vary the checkpoint, not the seed",
                 fontsize=11, fontweight="bold", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.4, loc="lower left")
    caption(fig,
             "Both bars are the same four trained runs, scored at their final epoch and at their best validation epoch. Nothing about the\n"
             "text encoder changed between them. One of four conditions changes sign, and it is the one that produced RQ7's headline.\n"
             "Note the asymmetry this exposes elsewhere: RQ2/RQ4/RQ5/RQ6 are all scored at *_best, and the baseline they beat or lose to\n"
             "has no best checkpoint — a selection that favours the ablations, making RQ4's and RQ6's negative verdicts conservative.",)
    save(fig, "fig_checkpoint_sensitivity.png")


# --------------------------------------------------------------------------------------------
# 6. Three tie-breaking rules, and the block-level chance the third one needs
# --------------------------------------------------------------------------------------------

def fig_pointing_rules():
    """How much of the pointing result is the rule for breaking a tied maximum?

    The strided sliding window makes the heatmap piecewise-constant over s³ blocks, so 'the peak' is
    ambiguous by up to the stride. Three rules were recorded. The first two are point rules and share
    the per-voxel chance level; the third asks whether the peak *block* touches the lesion at all,
    which needs its own chance baseline -- computed here from the masks as the fraction of
    stride-aligned blocks the ground truth intersects.
    """
    d = rule(sweep_csv("32", "16"), "otsu")
    rows = []
    for reg in REGIONS:
        for b in BINS:
            sub = d[(d.region == reg) & (d.size_bin == b)]
            chances = []
            for pid in sub.patient_id:
                mask = np.load(os.path.join(PREPROCESSED_DIR, f"{pid}.npz"))["mask"]
                gt = region_mask(mask, reg)
                blocks = gt.reshape(8, 16, 8, 16, 8, 16).any(axis=(1, 3, 5))
                chances.append(blocks.mean())
            rows.append(dict(region=reg, bin=b, n=len(sub),
                             argmax=sub.argmax_hit.mean(), centroid=sub.centroid_hit.mean(),
                             anytied=sub.any_tied_hit.mean(), block_chance=float(np.mean(chances))))
    t = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5),
                            gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    x = np.arange(9)
    width = 0.26
    ax.bar(x - width, t.argmax, width, color="#bcbcb8", label="argmax corner (naive)")
    ax.bar(x, t.centroid, width, color=BLUE, label="plateau centroid (the report's rule)")
    ax.bar(x + width, t.anytied, width, color=PURPLE, label="peak block touches lesion")
    for xi, ch in enumerate(t.block_chance):
        ax.plot([xi + width - 0.12, xi + width + 0.12], [ch, ch], color=INK, linewidth=1.6, zorder=4)
    ax.plot([], [], color=INK, linewidth=1.6, label="chance for the block rule")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.region}\n{r['bin']}" for _, r in t.iterrows()], fontsize=8)
    for xv in (2.5, 5.5):
        ax.axvline(xv, color=GRID, linewidth=1.0)
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    style_axes(ax, ylabel="hit rate on held-out patients")
    ax.set_title("Three ways to break a 4096-voxel tie, at the published 32³/stride-16 protocol",
                 fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left", ncol=2)

    ax = axes[1]
    # How much the choice of rule is worth, per condition. Plotting the three rates against plateau
    # size instead would mix in the effect of window size and read as noise; the spread between the
    # most and least generous rule is the quantity the check is actually about.
    rows = []
    for w, s in SWEEP:
        sub = rule(sweep_csv(w, s), "otsu")
        rows.append((f"{w}³/{s}", int(sub.peak_tie_count.median()),
                     sub.any_tied_hit.mean() - sub.centroid_hit.mean()))
    rows.sort(key=lambda r: -r[1])
    xs = np.arange(len(rows))
    ax.bar(xs, [r[2] for r in rows], 0.58, color=PURPLE, alpha=0.85)
    for xi, r in enumerate(rows):
        ax.text(xi, r[2] + 0.008, f"{r[2]:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r[0]}\n{r[1]:,} vox" for r in rows], fontsize=7.8)
    ax.set_ylim(0, 0.46)
    style_axes(ax, ylabel="hit-rate spread between the most and\nleast generous tie-breaking rule",
               xlabel="window / stride, by tied-plateau size (shrinking →)")
    ax.set_title("How much the choice of rule is worth", fontsize=10, color=INK, pad=8)

    fig.suptitle("What small-enhancing-tumor failure actually is: the right block, the wrong voxel",
                 fontsize=12, fontweight="bold", color=INK, y=1.03)
    caption(fig,
             "For small ET both point rules give 0 of 21 — the model never points at the lesion. But its winning 16³ block *contains* part of\n"
             "the lesion in 9 of 21 cases, 36× the 1.19% block-level chance. The failure is resolution inside a 26 mm block, not a failure to\n"
             "search: which is the architectural reading of this project, measured rather than asserted. The spread at right shows the rule\n"
             "ambiguity shrinking with the stride that causes it — it is a property of the protocol, not a residual doubt about the model.",)
    save(fig, "fig_pointing_rules.png")


# --------------------------------------------------------------------------------------------
# 7. What each binarizer actually selects
# --------------------------------------------------------------------------------------------

def fig_otsu_selected_fraction():
    """The mechanism behind the sign disagreement, measured instead of argued.

    Section 7.11 explains why Otsu prefers the non-overlapping tiling by appeal to histogram shape.
    The predicted-mask column tests it directly: dropping overlap moves Otsu's cut point, shrinking
    the mask by 21% of its own size and tightening its spread across patients. Every other rule
    selects a fixed fraction by construction and cannot move at all.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4),
                            gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    xs = np.arange(len(SWEEP))
    for m in RULES:
        ys = [rule(sweep_csv(w, s), m).pred_voxels.mean() / TOTAL_VOXELS * 100 for w, s in SWEEP]
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=2.0 if m == "otsu" else 1.4,
                color=RULE_COLORS[m], label=RULE_LABELS[m],
                zorder=4 if m == "otsu" else 2)
    ax.axvspan(3.5, len(SWEEP) - 0.5, color=AMBER, alpha=0.10, zorder=0)
    ax.text(4.5, 13.2, "tiling convention\nchanges here", fontsize=8, color="#8a6d0b", ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{w}³/{s}" for w, s in SWEEP], fontsize=8.8)
    ax.set_ylim(0, 14.5)
    style_axes(ax, ylabel="% of the imaged volume marked as lesion",
               xlabel="window / stride")
    ax.set_title("Only Otsu's line moves — the others are fixed by construction",
                 fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.4, loc="center right")

    ax = axes[1]
    a = rule(sweep_csv("8", "4"), "otsu")
    b = rule(sweep_csv("8", "8"), "otsu")
    bins = np.linspace(0, 22, 45)
    ax.hist(a.pred_voxels / TOTAL_VOXELS * 100, bins=bins, color=BLUE, alpha=0.6,
            label=f"8³ / stride 4 (50% overlap)\nmean {a.pred_voxels.mean() / TOTAL_VOXELS * 100:.2f}%, "
                  f"sd {a.pred_voxels.std() / TOTAL_VOXELS * 100:.2f}")
    ax.hist(b.pred_voxels / TOTAL_VOXELS * 100, bins=bins, color=ORANGE, alpha=0.6,
            label=f"8³ / stride 8 (no overlap)\nmean {b.pred_voxels.mean() / TOTAL_VOXELS * 100:.2f}%, "
                  f"sd {b.pred_voxels.std() / TOTAL_VOXELS * 100:.2f}")
    truth = rule(sweep_csv("32", "16"), "oracle_volume").pred_voxels.mean() / TOTAL_VOXELS * 100
    ax.axvline(truth, color=GREEN, linewidth=1.8)
    ax.text(truth + 0.4, ax.get_ylim()[1] * 0.88, f"true lesion\nfraction {truth:.2f}%",
            fontsize=8.2, color=GREEN)
    style_axes(ax, ylabel="patients", xlabel="% of the volume Otsu marks as lesion")
    ax.set_title("Same window, only the stride changed", fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")

    fig.suptitle("Why Otsu and every calibrated rule disagree in sign about the same heatmaps",
                 fontsize=12, fontweight="bold", color=INK, y=1.03)
    caption(fig,
             "Dropping overlap makes the heatmap blocky and low-entropy. That gives Otsu's between-class-variance criterion a cleaner\n"
             "two-class split, so it moves its cut point and returns a mask 21% smaller and less variable — closer to the truth for a reason\n"
             "that has nothing to do with the heatmap being better. Every rule that selects a fixed fraction of voxels instead pays the full\n"
             "price of the resolution just lost, which is why top-1% rejects the same change by −0.099 Dice while Otsu rewards it by +0.017.",)
    save(fig, "fig_otsu_selected_fraction.png")


# --------------------------------------------------------------------------------------------
# 8. The window curve, per region
# --------------------------------------------------------------------------------------------

def fig_window_curve_per_region():
    """Section 7.11's pooled 12-16 optimum, broken out into the three curves it averages.

    The pooled curve is a genuine result about the protocol as a whole, but it hides that tumor core
    peaks at 16 and falls away sharply, whole tumor is still climbing at 8, and enhancing tumor is
    flat across the middle of the range. That is the same regional split the pointing game found, and
    it means 'use a smaller window' is not one recommendation but three.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), sharex=True)
    xs = np.arange(len(OVERLAPPED))
    for ax, m in zip(axes, ["otsu", "pct99", "oracle_volume"]):
        for reg in REGIONS:
            ys = []
            for w, s in OVERLAPPED:
                sub = rule(sweep_csv(w, s), m)
                ys.append(sub[sub.region == reg].dice.mean())
            ax.plot(xs, ys, marker="o", markersize=6, linewidth=2.0,
                    color=REGION_COLORS[reg], label=reg)
            best = int(np.argmax(ys))
            ax.scatter([best], [ys[best]], s=170, facecolor="none",
                       edgecolor=REGION_COLORS[reg], linewidth=1.8, zorder=5)
            ax.annotate(f"{OVERLAPPED[best][0]}³", xy=(best, ys[best]), xytext=(0, 13),
                        textcoords="offset points", ha="center", fontsize=8.4,
                        color=REGION_COLORS[reg], fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{w}³/{s}" for w, s in OVERLAPPED], fontsize=8.8)
        style_axes(ax, ylabel="mean Dice" if m == "otsu" else None, xlabel="window / stride")
        ax.set_title(RULE_LABELS[m], fontsize=10.5, color=INK, pad=8, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("“Use a smaller window” is not one recommendation — the three regions want three "
                 "different windows",
                 fontsize=12, fontweight="bold", color=INK, y=1.03)
    caption(fig,
             "All four points hold the 50%-overlap convention fixed, so this is a window-size comparison and nothing else. Circles mark each\n"
             "region's optimum. Under Otsu everything looks monotone; under the calibrated rules tumor core peaks at 16³ and then falls, whole\n"
             "tumor is still climbing at 8³, and enhancing tumor is flat from 16³ to 8³. Section 7.11's pooled 12³–16³ bracket is the average of\n"
             "these three, and matches none of them exactly — the same regional split the pointing game reported independently.",)
    save(fig, "fig_window_curve_per_region.png")


# --------------------------------------------------------------------------------------------
# 9. What the intervention costs
# --------------------------------------------------------------------------------------------

def fig_cost_benefit():
    """Forward passes per volume against the Dice each sweep condition buys.

    The report calls the smaller window "free" in the sense of needing no retraining, and Section 7.3
    already corrects that to "no retraining is not the same as no compute". This puts a number on it
    and shows the recommended operating point is not the cheapest or the best, but the knee.
    """
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for (w, s) in SWEEP:
        n = ((GRID_EDGE - int(w)) // int(s) + 1) ** 3
        dice = rule(sweep_csv(w, s), "pct99").dice.mean()
        overlapped = int(s) * 2 == int(w)
        color = BLUE if overlapped else "#bcbcb8"
        ax.scatter([n], [dice], s=150, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{w}³ / stride {s}", xy=(n, dice), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=8.4,
                    color=INK if overlapped else INK_MUTED)
    ov = [(((GRID_EDGE - int(w)) // int(s) + 1) ** 3, rule(sweep_csv(w, s), "pct99").dice.mean())
          for w, s in OVERLAPPED]
    ax.plot([o[0] for o in ov], [o[1] for o in ov], color=BLUE, linewidth=1.6, alpha=0.5, zorder=2)
    best_n, best_d = ov[2]
    ax.scatter([best_n], [best_d], s=380, facecolor="none", edgecolor=GREEN, linewidth=2.2, zorder=4)
    ax.annotate("recommended operating point\n+0.060 Dice for 23× the forward passes",
                xy=(best_n, best_d), xytext=(-30, -52), textcoords="offset points",
                fontsize=8.6, color=GREEN, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.3))
    ax.set_xscale("log")
    ax.set_xlim(200, 60000)
    ax.set_ylim(0.22, 0.40)
    style_axes(ax, ylabel="pooled Dice, top-1% threshold",
               xlabel="sliding-window forward passes per volume  (log)")
    ax.set_title("What the one intervention that works actually costs",
                 fontsize=11.5, fontweight="bold", color=INK, pad=10)
    ax.plot([], [], "o", color=BLUE, label="50% overlap (comparable)")
    ax.plot([], [], "o", color="#bcbcb8", label="non-overlapping (confounded)")
    ax.legend(frameon=False, fontsize=8.6, loc="lower left")
    caption(fig,
             "Measured wall-clock on one RTX 2080 Ti, 73 validation patients × 3 regions: 2m47s at 32³/16, 19m17s at 16³/8, 52m at 12³/6,\n"
             "32m at 8³/4. The recommendation costs 10–23× the baseline's forward passes and returns +0.060 Dice under the deployable rule.\n"
             "For comparison, retraining an arm from scratch takes 24 minutes — so the intervention is cheap against retraining, and unlike\n"
             "retraining it needs no labels, no optimiser and no second set of weights. “No retraining” is still not the same as “free”.",)
    save(fig, "fig_cost_benefit.png")


def main():
    """Draw all nine appendix figures."""
    fig_iou_dice()
    fig_per_patient_spread()
    fig_effect_sizes()
    fig_bh_correction()
    fig_checkpoint_sensitivity()
    fig_pointing_rules()
    fig_otsu_selected_fraction()
    fig_window_curve_per_region()
    fig_cost_benefit()


if __name__ == "__main__":
    main()

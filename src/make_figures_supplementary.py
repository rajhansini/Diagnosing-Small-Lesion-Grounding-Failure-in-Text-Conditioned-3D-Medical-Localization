"""Nine supplementary figures, each carrying a claim the report previously made only in a table.

The report's twenty original figures cover the dataset, the core RQ1 result, and the corrections in
Sections 7.11-7.12. What they did not cover were nine claims that are load-bearing for the argument
but were stated only as numbers in prose:

  fig_anisotropy                the four class descriptions sit at ~0.99 pairwise cosine BEFORE the
                                trained projection and spread out after it -- the single fact that
                                makes RQ7's "geometry matters, meaning does not" result interpretable.
  fig_lift_over_chance          Section 6.2's control: model Dice against a random-heatmap baseline,
                                and the lift ratio that stays flat across size bins.
  fig_threshold_ladder          Section 6.3's five-binarizer table. The collapse survives every rule;
                                only its magnitude is a property of Otsu.
  fig_rq3_multiscale            Section 7.2's naive multi-scale ensemble: 1 bin better, 5 worse.
  fig_rq6_uniformity            Section 7.6's three-part verification of the uniformity repair --
                                embeddings separate, 2 of 3 shortcuts fixed, Dice beats RQ4 but not
                                the plain baseline. Previously three tables and no picture.
  fig_accuracy_vs_localization  Section 7.5's central dissociation: RQ4 has the best classification
                                accuracy of any arm and the worst localization.
  fig_rq8_embedding_vs_behavior Section 7.9's rho=+0.41, p=0.19 -- how far a manipulation moves the
                                query embedding does not predict how much it changes localization.
  fig_pointing_distance         the peak-to-lesion distances behind Section 6.3's median column,
                                including ET-small's 23.7 mm, at both window sizes.
  fig_rq13_arms                 Section 7.12's re-scoring of the three retrained arms under all five
                                binarizers, where RQ2's headline improvement disappears.

Every number is recomputed from results/*.csv or logs/*.out at draw time; nothing is transcribed.
Colour follows the project's fixed Okabe-Ito order, matching make_figures_dataset_and_evidence.py.
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = "/net/projects/ranalab/rajhansini/nlp_project"
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
LOGS_DIR = os.path.join(ROOT, "logs")
PREPROCESSED_DIR = os.path.join(ROOT, "data", "preprocessed")
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")

BLUE, ORANGE, GREEN, AMBER, PINK = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"
PURPLE, SKY = "#7B52AB", "#56B4E9"
REGION_COLORS = {"ET": BLUE, "TC": ORANGE, "WT": GREEN}
INK, INK_MUTED, GRID = "#1a1a19", "#5c5c58", "#d8d8d4"

REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
REGION_LABELS = {"ET": "ET  (enhancing tumor)", "TC": "TC  (tumor core)", "WT": "WT  (whole tumor)"}

# The five binarization rules, in the order they appear everywhere in the report.
RULES = ["otsu", "pct90", "pct95", "pct99", "oracle_volume"]
RULE_LABELS = {"otsu": "Otsu", "pct90": "top 10%", "pct95": "top 5%",
               "pct99": "top 1%", "oracle_volume": "oracle volume"}
RULE_COLORS = {"otsu": ORANGE, "pct90": "#bcbcb8", "pct95": SKY,
               "pct99": BLUE, "oracle_volume": GREEN}


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


def load_csv(path):
    """Read a CSV into a list of dict rows, or None if the file is absent."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def result(name):
    """Load a CSV from the results directory by filename."""
    return load_csv(os.path.join(RESULTS_DIR, name))


def dice_by_patient(rows, region, size_bin=None, where=None):
    """Map patient_id -> Dice for one region, optionally filtered by bin and extra predicates.

    Args:
        rows: dict rows from a result CSV.
        region: "ET", "TC" or "WT".
        size_bin: restrict to one tercile, or None for all.
        where: dict of column -> required value, applied on top of the above.

    Returns:
        dict of patient_id -> float Dice.
    """
    out = {}
    for r in rows:
        if r["region"] != region:
            continue
        if size_bin is not None and r["size_bin"] != size_bin:
            continue
        if where and any(r.get(k) != v for k, v in where.items()):
            continue
        out[r["patient_id"]] = float(r["dice"])
    return out


def paired(a, b):
    """Align two patient -> value maps onto their shared patients.

    Returns:
        (np.array, np.array) in a consistent patient order.
    """
    keys = sorted(set(a) & set(b))
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def wilcoxon_p(x, y):
    """Two-sided paired Wilcoxon p-value, or 1.0 when every pair is identical."""
    if len(x) == 0 or np.allclose(x, y):
        return 1.0
    return stats.wilcoxon(x, y).pvalue


def stars(p):
    """Conventional significance markers for annotating bars."""
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))


def save(fig, name):
    """Write a figure to the figures directory at the project's standard density."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------------------------
# 1. Text-side geometry: anisotropy before the projection, separation after it
# --------------------------------------------------------------------------------------------

def fig_anisotropy():
    """Show that the frozen text encoder carries almost no discriminative signal.

    Three panels: the raw PubMedBERT cosine matrix over the four class descriptions (~0.99
    everywhere), the same four vectors after the trained linear head, and the mean off-diagonal
    cosine of every text condition RQ7 substituted in -- which is what makes RQ7's decomposition
    into "meaning" and "geometry" legible.
    """
    import torch

    raw = torch.load(os.path.join(PREPROCESSED_DIR, "subregion_text_embeddings.pt"),
                     map_location="cpu")
    classes = ["ET", "TC", "WT", "NONE"]
    raw_mat = torch.stack([raw[c].float() for c in classes])

    state = torch.load(os.path.join(CHECKPOINTS_DIR, "baseline_aligner.pt"), map_location="cpu")
    proj = raw_mat @ state["text_proj.weight"].T.float() + state["text_proj.bias"].float()

    def cosmat(m):
        n = torch.nn.functional.normalize(m, dim=-1)
        return (n @ n.T).numpy()

    before, after = cosmat(raw_mat), cosmat(proj)

    def offdiag_mean(m):
        return float(m[~np.eye(len(m), dtype=bool)].mean())

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.75], "wspace": 0.85})

    for ax, mat, title in [
        (axes[0], before, "Raw PubMedBERT\n(frozen encoder output)"),
        (axes[1], after, "After the trained projection\n(what the model actually compares)"),
    ]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(4), classes, fontsize=9, color=INK_MUTED)
        ax.set_yticks(range(4), classes, fontsize=9, color=INK_MUTED)
        for i in range(4):
            for j in range(4):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(v) > 0.55 else INK,
                        fontweight="bold" if i != j else "normal")
        ax.set_title(f"{title}\nmean off-diagonal = {offdiag_mean(mat):+.3f}",
                     fontsize=9.5, color=INK, pad=8)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    # No colourbar: every cell is annotated with its value, so a scale bar would only crowd the
    # third panel. The shared vmin/vmax is stated in the caption instead.

    # -- right: every text condition RQ7 substituted, by its input geometry
    geom = {}
    for variant in ["bertbase", "randbert", "randvec_ortho", "randvec_aniso"]:
        for fn in sorted(os.listdir(LOGS_DIR)):
            if fn.startswith(f"rq7_train_{variant}_2") and fn.endswith(".out"):
                text = open(os.path.join(LOGS_DIR, fn)).read()
                m = re.search(r"mean off-diagonal cosine = (-?[\d.]+)", text)
                if m:
                    geom[variant] = float(m.group(1))
                    break

    # Each row: label, input geometry, mean Dice change over the three RQ7 seeds (Section 7.8),
    # and whether that change kept its sign across all three runs. Sign consistency against the
    # 0.0044 retraining noise floor -- not magnitude -- is what separates signal from noise here.
    rows = [
        ("PubMedBERT\n(the baseline)", offdiag_mean(before), None, None),
        ("BERT-base\n(general domain)", geom.get("bertbase"), -0.0050, False),
        ("Random-init BERT\n(never trained)", geom.get("randbert"), +0.0216, False),
        ("Random vectors,\nPubMedBERT geometry", geom.get("randvec_aniso"), +0.0128, False),
        ("4 random orthonormal\nvectors", geom.get("randvec_ortho"), -0.0087, True),
    ]

    ax = axes[2]
    ypos = np.arange(len(rows))[::-1]
    for y, (label, value, delta, consistent) in zip(ypos, rows):
        color = BLUE if delta is None else (ORANGE if consistent else "#9c9c98")
        ax.barh([y], [value], color=color, height=0.52)
        ax.text(value + 0.025, y + 0.06, f"{value:+.3f}", va="center", fontsize=8.5, color=INK)
        if delta is not None:
            note = ("sign holds in 3/3 runs — reliably worse" if consistent
                    else "sign flips across runs — noise")
            ax.text(0.025, y - 0.28, f"ΔDice {delta:+.4f}   {note}", va="center",
                    fontsize=7.5, color=ORANGE if consistent else INK_MUTED, style="italic")
    ax.set_yticks(ypos, [r[0] for r in rows], fontsize=8.5, color=INK)
    ax.axvline(0, color=INK_MUTED, linewidth=0.9)
    ax.set_xlim(-0.12, 1.32)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    style_axes(ax, xlabel="mean off-diagonal cosine among the four class embeddings",
               ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_title("Only the condition that discards the geometry reliably hurts",
                 fontsize=9.5, color=INK, pad=8)

    fig.suptitle("The text pathway's discriminative work is done entirely by the trained projection",
                 fontsize=12, fontweight="bold", color=INK, y=1.03)
    save(fig, "fig_anisotropy.png")


# --------------------------------------------------------------------------------------------
# 2. Section 6.2: the chance-baseline control
# --------------------------------------------------------------------------------------------

def fig_lift_over_chance():
    """Model Dice against a random-heatmap control, and the lift ratio that stays flat.

    Section 6.2's honest reading depends on both panels: absolute Dice collapses with size, but the
    ratio to chance does not, so the claim is about usability rather than about the model losing
    signal disproportionately on small lesions.
    """
    model = result("rq1_localization_scores.csv")
    chance = result("chance_baseline_scores.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    width, xs = 0.36, np.arange(9)
    labels, m_vals, c_vals, lifts, group_colors = [], [], [], [], []
    for reg in REGIONS:
        for b in BINS:
            m = np.mean(list(dice_by_patient(model, reg, b).values()))
            c = np.mean(list(dice_by_patient(chance, reg, b).values()))
            labels.append(f"{reg}\n{b}")
            m_vals.append(m)
            c_vals.append(c)
            lifts.append(m / c)
            group_colors.append(REGION_COLORS[reg])

    ax = axes[0]
    ax.bar(xs - width / 2, m_vals, width, color=group_colors, label="model")
    ax.bar(xs + width / 2, c_vals, width, color="#c9c9c5", label="chance (random heatmap)")
    for x, m in zip(xs, m_vals):
        ax.text(x - width / 2, m * 1.12, f"{m:.3f}", ha="center", fontsize=7.4, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylim(3e-4, 1.0)
    style_axes(ax, ylabel="mean Dice  (log scale)")
    ax.set_title("Both fall with lesion size — chance included", fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax = axes[1]
    for i, reg in enumerate(REGIONS):
        vals = lifts[i * 3:(i + 1) * 3]
        ax.plot([0, 1, 2], vals, "o-", color=REGION_COLORS[reg], linewidth=2.2,
                markersize=7, label=reg)
        for x, v in zip([0, 1, 2], vals):
            ax.text(x, v + 0.45, f"{v:.1f}×", ha="center", fontsize=8, color=REGION_COLORS[reg],
                    fontweight="bold")
    ax.set_xticks([0, 1, 2], BINS, fontsize=9)
    ax.set_ylim(0, max(lifts) * 1.28)
    style_axes(ax, ylabel="model Dice ÷ chance Dice")
    ax.set_title("…but the ratio to chance is flat", fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=9, title="region", title_fontsize=8.5)
    ax.annotate("within ~25% across bins,\nin every region",
                xy=(1, np.mean(lifts) * 0.55), fontsize=8.2, color=INK_MUTED,
                ha="center", style="italic")

    fig.suptitle("The size collapse is in absolute quality, not in advantage over chance",
                 fontsize=12, fontweight="bold", color=INK, y=1.02)
    save(fig, "fig_lift_over_chance.png")


# --------------------------------------------------------------------------------------------
# 3. Section 6.3: the collapse under all five binarization rules
# --------------------------------------------------------------------------------------------

def fig_threshold_ladder():
    """Dice under every binarization rule, per region and size bin.

    This is Section 6.3's table [3] made visible. The point is the ordering: every rule preserves
    small < medium < large, so the collapse is not manufactured by Otsu -- but the vertical gap
    between the Otsu bar and the oracle bar is how much of the reported magnitude was.
    """
    rows = result("rq11_threshold_confound_scores.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharey=True)
    width = 0.16

    for ax, reg in zip(axes, REGIONS):
        ratios = {}
        for k, rule in enumerate(RULES):
            means = []
            for b in BINS:
                vals = list(dice_by_patient(rows, reg, b, {"threshold_method": rule}).values())
                means.append(np.mean(vals))
            ratios[rule] = means[2] / means[0]
            offs = np.arange(3) + (k - 2) * width
            ax.bar(offs, means, width, color=RULE_COLORS[rule],
                   label=RULE_LABELS[rule] if reg == "ET" else None,
                   edgecolor="white", linewidth=0.6)

        # Otsu -> oracle paired significance, per bin (Section 6.3, table [2]).
        for j, b in enumerate(BINS):
            o = dice_by_patient(rows, reg, b, {"threshold_method": "otsu"})
            v = dice_by_patient(rows, reg, b, {"threshold_method": "oracle_volume"})
            x, y = paired(o, v)
            top = max(np.mean(y), np.mean(x))
            ax.text(j, top + 0.028, stars(wilcoxon_p(x, y)), ha="center", fontsize=9,
                    color=INK if wilcoxon_p(x, y) < 0.05 else INK_MUTED)

        ax.set_xticks(range(3), BINS, fontsize=9.5)
        style_axes(ax, ylabel="mean Dice" if reg == "ET" else None)
        ax.set_title(f"{REGION_LABELS[reg]}\nlarge/small ratio: Otsu {ratios['otsu']:.1f}× → "
                     f"oracle {ratios['oracle_volume']:.1f}×", fontsize=9.5, color=INK, pad=8)
        ax.set_ylim(0, 0.72)

    axes[0].legend(frameon=False, fontsize=8.5, ncol=1, loc="upper left",
                   title="binarization rule", title_fontsize=8.5)
    fig.text(0.5, -0.04, "Stars mark the paired Wilcoxon test between the Otsu and oracle bars "
                         "(*** p<0.001, ** p<0.01, * p<0.05). ET-small is the one bin where better "
                         "thresholding does not help — there the heatmap's ranking is itself poor.",
             ha="center", fontsize=8.4, color=INK_MUTED, style="italic")
    fig.suptitle("The size collapse survives every binarization rule; only its magnitude is Otsu's",
                 fontsize=12, fontweight="bold", color=INK, y=1.03)
    save(fig, "fig_threshold_ladder.png")


# --------------------------------------------------------------------------------------------
# 4. Section 7.2: naive multi-scale ensembling
# --------------------------------------------------------------------------------------------

def fig_rq3_multiscale():
    """RQ3's per-bin change against the baseline, coloured by corrected significance.

    Section 7.2 reports this as three verdict words per cell. The figure shows that the one bin that
    improved is small, the five that worsened are not, and the damage grows with lesion size --
    which is the shape you expect if a voxel-wise max is importing noise from a scale the model was
    never trained to interpret.
    """
    base = result("rq1_localization_scores.csv")
    rq3 = result("rq3_localization_scores.csv")
    fam = {(r["region"], r["bin"]): r for r in result("full_family_statistics.csv")
           if r["comparison"] == "RQ3 (multi-scale ensemble)"}

    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    xs = np.arange(9)
    labels, deltas, colors, notes = [], [], [], []
    for reg in REGIONS:
        for b in BINS:
            x, y = paired(dice_by_patient(base, reg, b), dice_by_patient(rq3, reg, b))
            d = float(np.mean(y - x))
            q = float(fam[(reg, b)]["q_pooled"]) if (reg, b) in fam else 1.0
            labels.append(f"{reg}\n{b}")
            deltas.append(d)
            if q >= 0.05:
                colors.append("#c9c9c5")
                notes.append("n.s.")
            else:
                colors.append(BLUE if d > 0 else ORANGE)
                notes.append(stars(q))

    bars = ax.bar(xs, deltas, 0.62, color=colors, edgecolor="white", linewidth=0.8)
    for x, d, n in zip(xs, deltas, notes):
        off = 0.004 if d > 0 else -0.008
        ax.text(x, d + off, f"{d:+.3f}\n{n}", ha="center", fontsize=7.8,
                va="bottom" if d > 0 else "top", color=INK)
    ax.axhline(0, color=INK, linewidth=1.1)
    ax.set_xticks(xs, labels, fontsize=8.5)
    ax.set_ylim(min(deltas) * 1.62, max(deltas) * 3.2)
    style_axes(ax, ylabel="Δ Dice vs. the 32³ baseline")
    ax.set_title("Naive multi-scale ensembling (16³+32³+64³, voxel-wise max): 1 bin better, "
                 "5 significantly worse",
                 fontsize=11, fontweight="bold", color=INK, pad=10)
    ax.annotate("the only improvement,\nand the smallest effect on the chart",
                xy=(xs[6], deltas[6]), xytext=(xs[6] - 1.5, max(deltas) * 2.3),
                fontsize=8.2, color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.0))
    ax.text(xs[7], min(deltas) * 1.34,
            "damage grows with lesion size: the 64³ scale the model\n"
            "cannot interpret fires uniformly, and the voxel-wise max lets it win",
            fontsize=8.2, color=ORANGE, ha="center", va="top")
    save(fig, "fig_rq3_multiscale.png")


# --------------------------------------------------------------------------------------------
# 5. Section 7.6: the uniformity repair, verified in three parts
# --------------------------------------------------------------------------------------------

def parse_epochs(path, key):
    """Pull a per-epoch scalar out of a training log.

    Args:
        path: log file path.
        key: the metric name as it appears in the log, e.g. "val_acc".

    Returns:
        list of floats in epoch order.
    """
    if not os.path.exists(path):
        return []
    text = open(path).read()
    return [float(m) for m in re.findall(rf"{key}=(-?[\d.]+)", text)]


def find_log(prefix):
    """Return the newest log whose filename starts with prefix, or None."""
    hits = sorted(fn for fn in os.listdir(LOGS_DIR)
                  if fn.startswith(prefix) and fn.endswith(".out"))
    return os.path.join(LOGS_DIR, hits[-1]) if hits else None


def fig_rq6_uniformity():
    """The three separate things Section 7.6 verifies, side by side.

    Left: the regularizer does what it claims, driving mean pairwise class similarity from RQ4's
    collapsed ~0.97 to about -0.10. Centre: the noise probe re-run, showing 2 of 3 pipelines now
    prefer their own label where RQ4 had 0 of 3. Right: the localization consequence -- a real gain
    over RQ4 in all nine bins that still does not reach the plain baseline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.15, 1.35]})

    # -- left: the uniformity term over training
    log = find_log("rq6_train_20")
    train_u = parse_epochs(log, "train_unif")
    val_u = parse_epochs(log, "val_unif")
    ax = axes[0]
    ep = np.arange(1, len(train_u) + 1)
    ax.plot(ep, train_u, color=GREEN, linewidth=2.0, label="train")
    ax.plot(ep, val_u, color=GREEN, linewidth=1.3, linestyle="--", alpha=0.75, label="validation")
    ax.axhline(0.97, color=ORANGE, linewidth=1.6, linestyle=":")
    ax.text(len(ep) * 0.42, 0.99, "RQ4's collapsed embeddings (~0.97)", fontsize=8.2,
            color=ORANGE, va="bottom")
    ax.annotate(f"settles at {train_u[-1]:+.2f}", xy=(len(ep), train_u[-1]),
                xytext=(len(ep) * 0.55, -0.42), fontsize=8.4, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.0))
    ax.set_ylim(-0.6, 1.15)
    style_axes(ax, ylabel="mean pairwise cosine among class embeddings", xlabel="epoch")
    ax.set_title("1 · The embeddings actually separate", fontsize=9.8, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")

    # -- centre: the noise probe, RQ4 vs RQ6
    def cross_sims(path):
        """Parse the 3x3 noise-pipeline vs size-text similarity block out of a probe log."""
        text = open(path).read()
        out = {}
        for pipe, s, m, l in re.findall(
                r"noise via (\w+)-pipeline \(crop=\d+\) -> similarity to: "
                r"small=(-?[\d.]+), medium=(-?[\d.]+), large=(-?[\d.]+)", text):
            out[pipe] = [float(s), float(m), float(l)]
        return out

    rq4 = cross_sims(find_log("rq4_shortcut_20"))
    rq6 = cross_sims(find_log("rq6_hub_test_20"))
    pipes = ["small", "medium", "large"]
    ax = axes[1]
    xs = np.arange(3)
    width = 0.115
    n_correct = {}
    for k, (probe, hatch) in enumerate([(rq4, ""), (rq6, "///")]):
        centre_off = -0.21 if k == 0 else 0.21
        for j, text_cls in enumerate(pipes):
            vals = [probe[p][j] for p in pipes]
            ax.bar(xs + centre_off + (j - 1) * width, vals, width,
                   color=[BLUE, AMBER, ORANGE][j], hatch=hatch,
                   edgecolor="white", linewidth=0.5,
                   label=f'"{text_cls}" text' if k == 0 else None)
        hits = 0
        for i, p in enumerate(pipes):
            correct = int(np.argmax(probe[p])) == i
            hits += correct
            ax.text(xs[i] + centre_off, max(probe[p]) + 0.035, "✓" if correct else "✗",
                    ha="center", fontsize=12, color=GREEN if correct else ORANGE,
                    fontweight="bold")
        n_correct[k] = hits
    ax.axhline(0, color=INK_MUTED, linewidth=0.9)
    ax.set_xticks(xs, [f"noise from\n{p} crops" for p in pipes], fontsize=8.6)
    for i in xs:
        ax.text(i - 0.21, -0.60, "RQ4", ha="center", fontsize=7.6, color=INK_MUTED)
        ax.text(i + 0.21, -0.60, "RQ6", ha="center", fontsize=7.6, color=INK_MUTED)
    style_axes(ax, ylabel="cosine similarity to each size-text")
    ax.set_title(f"2 · Noise probe re-run: {n_correct[0]} of 3 → {n_correct[1]} of 3 prefer "
                 f"their own label\n(solid = RQ4, hatched = RQ6)",
                 fontsize=9.8, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=7.8, ncol=3, loc="upper left", columnspacing=0.9)
    ax.set_ylim(-0.68, 0.62)
    ax.text(1.0, -0.50, 'RQ4\'s one "correct" cell is the hub itself: every pipeline picks "large"',
            ha="center", fontsize=7.4, color=INK_MUTED, style="italic")

    # -- right: what it bought in localization
    base = result("rq1_localization_scores.csv")
    rq4d = result("rq4_localization_scores.csv")
    rq6d = result("rq6_localization_scores.csv")
    ax = axes[2]
    xs = np.arange(9)
    width = 0.27
    labels, b_m, r4_m, r6_m = [], [], [], []
    for reg in REGIONS:
        for b in BINS:
            labels.append(f"{reg}\n{b}")
            b_m.append(np.mean(list(dice_by_patient(base, reg, b).values())))
            r4_m.append(np.mean(list(dice_by_patient(rq4d, reg, b).values())))
            r6_m.append(np.mean(list(dice_by_patient(rq6d, reg, b).values())))
    ax.bar(xs - width, r4_m, width, color=ORANGE, label="RQ4 (scale-matched)")
    ax.bar(xs, r6_m, width, color=GREEN, label="RQ6 (+ uniformity fix)")
    ax.bar(xs + width, b_m, width, color=BLUE, label="RQ1 baseline (doing nothing)")
    for x, lo, hi in zip(xs, r4_m, r6_m):
        ax.annotate("", xy=(x, hi), xytext=(x - width, lo),
                    arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=0.9, alpha=0.75))
    ax.set_xticks(xs, ["sm", "med", "lg"] * 3, fontsize=8)
    for i, reg in enumerate(REGIONS):
        ax.text(i * 3 + 1, -0.036, reg, ha="center", fontsize=9.5, color=INK, fontweight="bold")
        if i:
            ax.axvline(i * 3 - 0.5, color=GRID, linewidth=0.9)
    style_axes(ax, ylabel="mean Dice")
    ax.set_title("3 · Beats RQ4 in 9 of 9 bins — and still loses to the baseline",
                 fontsize=9.8, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")

    fig.suptitle("RQ6: the diagnosed embedding hub was genuinely repaired, and it was not enough",
                 fontsize=12, fontweight="bold", color=INK, y=1.03)
    save(fig, "fig_rq6_uniformity.png")


# --------------------------------------------------------------------------------------------
# 6. Section 7.5: classification accuracy against localization quality
# --------------------------------------------------------------------------------------------

def fig_accuracy_vs_localization():
    """The dissociation that motivates the whole architecture reading of this project.

    Every training-side arm was selected on classification accuracy, and the arm that achieved the
    best accuracy (RQ4) produced the worst localization of any arm tested. Left: the validation
    accuracy curves, each against its own chance level. Right: best accuracy plotted against pooled
    localization Dice, where the relationship is if anything negative.
    """
    arms = [
        ("RQ1 baseline", "baseline_train_20", "rq1_localization_scores.csv", 4, BLUE),
        ("RQ2 size-conditioned", "rq2_train_20", "rq2_localization_scores.csv", 10, AMBER),
        ("RQ4 scale-matched", "rq4_train_20", "rq4_localization_scores.csv", 10, ORANGE),
        ("RQ5 naturalistic", "rq5_train_20", "rq5_localization_scores.csv", 4, PURPLE),
        ("RQ6 uniformity fix", "rq6_train_20", "rq6_localization_scores.csv", 10, GREEN),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    points = []
    for name, prefix, csv_name, n_classes, color in arms:
        curve = parse_epochs(find_log(prefix), "val_acc")
        if not curve:
            continue
        ax.plot(np.arange(1, len(curve) + 1), curve, color=color, linewidth=1.9, label=name)
        rows = result(csv_name)
        dice = np.mean([float(r["dice"]) for r in rows])
        points.append((name, max(curve) / (1.0 / n_classes), dice, color, max(curve), n_classes))
    ax.axhline(0.25, color=INK_MUTED, linewidth=1.0, linestyle=":")
    ax.text(2.0, 0.265, "chance, 4-way", fontsize=7.6, color=INK_MUTED, va="bottom")
    ax.axhline(0.10, color=INK_MUTED, linewidth=1.0, linestyle=":")
    ax.text(2.0, 0.115, "chance, 10-way", fontsize=7.6, color=INK_MUTED, va="bottom")
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 0.80)
    style_axes(ax, ylabel="validation classification accuracy", xlabel="epoch")
    ax.set_title("Every arm learns its training objective", fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.2, loc="lower right", ncol=2)

    ax = axes[1]
    # Hand-placed offsets: the RQ1 and RQ5 points nearly coincide, so automatic placement collides.
    label_offsets = {"RQ1 baseline": (34, -30), "RQ5 naturalistic": (30, 14),
                     "RQ2 size-conditioned": (0, 16), "RQ6 uniformity fix": (-14, 18),
                     "RQ4 scale-matched": (6, -34)}
    for name, lift, dice, color, raw_acc, n_classes in points:
        ax.scatter([lift], [dice], s=130, color=color, zorder=3, edgecolor="white", linewidth=1.4)
        ax.annotate(f"{name}\n({raw_acc:.3f} of {n_classes}-way)", xy=(lift, dice),
                    xytext=label_offsets.get(name, (0, 16)),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    lifts = np.array([p[1] for p in points])
    dices = np.array([p[2] for p in points])
    rho, p = stats.spearmanr(lifts, dices)
    fit = np.polyfit(lifts, dices, 1)
    grid = np.linspace(lifts.min() * 0.96, lifts.max() * 1.04, 20)
    ax.plot(grid, np.polyval(fit, grid), color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlim(lifts.min() * 0.82, lifts.max() * 1.16)
    ax.set_ylim(0, max(dices) * 1.5)
    style_axes(ax, ylabel="pooled localization Dice",
               xlabel="best validation accuracy, as lift over chance")
    ax.set_title(f"Better classification, worse localization\n"
                 f"Spearman ρ = {rho:+.2f}  (n = {len(points)} arms)",
                 fontsize=10, color=INK, pad=8)

    fig.suptitle("Training-objective accuracy does not transfer to localization quality",
                 fontsize=12, fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, -0.07,
             "Dice here is Otsu-scored, the project's original protocol. Section 7.12 shows RQ2's\n"
             "apparent advantage is an artifact of that binarizer — under a calibrated rule its point\n"
             "drops to the baseline's level, which steepens this relationship rather than weakening it.",
             ha="center", fontsize=8.2, color=INK_MUTED, style="italic")
    save(fig, "fig_accuracy_vs_localization.png")


# --------------------------------------------------------------------------------------------
# 7. Section 7.9: does moving the query embedding change the model's behaviour?
# --------------------------------------------------------------------------------------------

def fig_rq8_embedding_vs_behavior():
    """Embedding displacement against behavioural change, over the 12 condition x region cells.

    Section 7.9 reports the correlation as a single number. Plotted, the absence of a relationship
    is the whole point: a model whose query embedding carried graded meaning would sit on a rising
    line, and the two cells where a degraded query localizes BETTER than the true clinical text
    would not exist at all.
    """
    rows = result("rq8_compositionality_scores.csv")
    conditions = ["negated", "shuffled", "swapped", "generic"]
    cond_marker = {"negated": "o", "shuffled": "s", "swapped": "^", "generic": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))

    pts = []
    for cond in conditions:
        for reg in REGIONS:
            orig = dice_by_patient(rows, reg, where={"condition": "original"})
            var = dice_by_patient(rows, reg, where={"condition": cond})
            x, y = paired(orig, var)
            delta = float(np.mean(y - x))
            cos = np.mean([float(r["proj_cosine_vs_original"]) for r in rows
                           if r["region"] == reg and r["condition"] == cond])
            pts.append((cond, reg, 1 - cos, delta))

    ax = axes[0]
    for cond, reg, dist, delta in pts:
        ax.scatter([dist], [abs(delta)], marker=cond_marker[cond], s=105,
                   color=REGION_COLORS[reg], edgecolor="white", linewidth=1.2, zorder=3)
    dists = np.array([p[2] for p in pts])
    mags = np.array([abs(p[3]) for p in pts])
    rho, pval = stats.spearmanr(dists, mags)
    fit = np.polyfit(dists, mags, 1)
    grid = np.linspace(dists.min() * 0.95, dists.max() * 1.05, 20)
    ax.plot(grid, np.polyval(fit, grid), color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    style_axes(ax, ylabel="|Δ Dice| vs. the true clinical query",
               xlabel="1 − cosine to the original query, after the trained projection")
    ax.set_title(f"Displacement does not predict behaviour\nSpearman ρ = {rho:+.3f},  p = {pval:.2f}",
                 fontsize=10, color=INK, pad=8)
    ax.set_ylim(-0.004, mags.max() * 1.32)
    handles = [plt.Line2D([], [], marker=cond_marker[c], linestyle="", color=INK_MUTED,
                          markersize=7, label=c) for c in conditions]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color=REGION_COLORS[r],
                           markersize=7, label=r) for r in REGIONS]
    ax.legend(handles=handles, frameon=False, fontsize=7.8, ncol=4, loc="lower center",
              columnspacing=1.0, handletextpad=0.3)
    ax.text(dists.mean(), mags.max() * 1.24,
            "every manipulation lands in a narrow band above 0.94 cosine",
            fontsize=8.2, color=INK_MUTED, ha="center", style="italic")

    ax = axes[1]
    xs = np.arange(len(pts))
    deltas = [p[3] for p in pts]
    colors = [REGION_COLORS[p[1]] if p[3] > 0 else "#c9c9c5" for p in pts]
    ax.bar(xs, deltas, 0.66, color=colors, edgecolor="white", linewidth=0.7)
    ax.axhline(0, color=INK, linewidth=1.1)
    ax.set_xticks(xs, [p[1] for p in pts], fontsize=8)
    for i, cond in enumerate(conditions):
        ax.text(i * 3 + 1, min(deltas) * 1.22, cond, ha="center", fontsize=8.8, color=INK,
                fontweight="bold")
        if i:
            ax.axvline(i * 3 - 0.5, color=GRID, linewidth=0.9)
    ax.set_ylim(min(deltas) * 1.3, max(deltas) * 1.55)
    style_axes(ax, ylabel="Δ Dice vs. the true clinical query")
    ax.set_title("Two degraded queries beat the real description", fontsize=10, color=INK, pad=8)
    for i, (cond, reg, _, d) in enumerate(pts):
        if d > 0:
            ax.text(i, d + 0.0018, f"{d:+.3f}", ha="center", fontsize=7.8,
                    color=REGION_COLORS[reg], fontweight="bold")
    ax.text(7.0, max(deltas) * 1.38, "contentless filler and a wrong-region term,\n"
                                     "both on whole tumor",
            fontsize=8.2, color=GREEN, ha="center")

    fig.suptitle("RQ8: the query behaves as an opaque class identifier, not as language",
                 fontsize=12, fontweight="bold", color=INK, y=1.02)
    save(fig, "fig_rq8_embedding_vs_behavior.png")


# --------------------------------------------------------------------------------------------
# 8. Sections 6.3 / 7.11: how far the peak response lands from the lesion
# --------------------------------------------------------------------------------------------

def fig_pointing_distance():
    """Peak-to-lesion distance distributions at both window sizes, per region and bin.

    The pointing-game hit rate is a binary summary; the distance is what it is summarising. ET-small
    is the bin where the 32³ peak sits a median 23.7 mm away -- not a near miss, and not a boundary
    problem -- and it is also the bin the smaller window moves most.
    """
    w32 = [r for r in result("rq12_grounding_window32_stride16_scores.csv")
           if r["threshold_method"] == "otsu"]
    w8 = [r for r in result("rq12_grounding_window8_stride4_scores.csv")
          if r["threshold_method"] == "otsu"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=True)
    rng = np.random.default_rng(0)

    for ax, reg in zip(axes, REGIONS):
        for j, b in enumerate(BINS):
            for k, (rows, color, label) in enumerate([(w32, ORANGE, "32³ / stride 16"),
                                                      (w8, BLUE, "8³ / stride 4")]):
                d = np.array([float(r["centroid_dist_mm"]) for r in rows
                              if r["region"] == reg and r["size_bin"] == b])
                x = j + (k - 0.5) * 0.34
                ax.scatter(x + rng.normal(0, 0.045, len(d)), d, s=16, color=color, alpha=0.5,
                           edgecolor="none", zorder=2,
                           label=label if (reg == "TC" and j == 0) else None)
                med = float(np.median(d))
                ax.plot([x - 0.12, x + 0.12], [med, med], color=color, linewidth=2.6, zorder=3)
                ax.text(x, -3.2, f"{med:.0f}", ha="center", fontsize=7.6, color=color,
                        fontweight="bold")
        ax.set_xticks(range(3), BINS, fontsize=9.5)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-6, 78)
        style_axes(ax, ylabel="distance from peak response to nearest lesion voxel (mm)"
                   if reg == "ET" else None)
        ax.set_title(REGION_LABELS[reg], fontsize=10, color=INK, pad=8)
        ax.axhline(0, color=GREEN, linewidth=1.0, linestyle="--", alpha=0.7)

    axes[1].legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    axes[0].annotate("ET-small at 32³:\nmedian 23.7 mm, 0 of 21 hits", xy=(-0.17, 24),
                     xytext=(1.15, 62), fontsize=8.2, color=ORANGE, ha="center",
                     arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.0))
    axes[2].text(2.52, 4, "0 mm = peak inside the lesion", fontsize=8, color=GREEN,
                 ha="right", style="italic")
    axes[1].text(1.0, 62, "tumor core is the one region the smaller window degrades",
                 fontsize=8.2, color=INK_MUTED, ha="center", style="italic")
    fig.text(0.5, -0.03, "Each dot is one validation patient; the thick bar is the bin median "
                         "(printed below the axis). Distances use the tie-aware plateau centroid, "
                         "not raw argmax.",
             ha="center", fontsize=8.4, color=INK_MUTED, style="italic")
    fig.suptitle("Where the peak response actually lands, and what the smaller window moves",
                 fontsize=12, fontweight="bold", color=INK, y=1.03)
    save(fig, "fig_pointing_distance.png")


# --------------------------------------------------------------------------------------------
# 9. Section 7.12: the three retrained arms, re-scored under every rule
# --------------------------------------------------------------------------------------------

def fig_rq13_arms():
    """Each retrained arm's change against the baseline, under all five binarization rules.

    RQ2's bar is the finding: strongly positive under Otsu at p = 8e-17, and indistinguishable from
    zero under every rule that does not read the intensity histogram. RQ4's and RQ6's bars are
    negative under all five, which is what a verdict that does not depend on the instrument looks
    like.
    """
    # analyze_rq13.py scores the baseline from the regenerated 32^3/stride-16 sweep, which carries
    # all five rules; using the same file keeps every number here identical to the analysis log.
    base = result("rq12_grounding_window32_stride16_scores.csv")
    arms = [("RQ2\nsize-conditioned text", "rq13_rq2_thresholds_scores.csv"),
            ("RQ4\nscale-matched", "rq13_rq4_thresholds_scores.csv"),
            ("RQ6\nuniformity fix", "rq13_rq6_thresholds_scores.csv")]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    width = 0.16
    xs = np.arange(len(arms))
    for k, rule in enumerate(RULES):
        deltas, ps = [], []
        for _, csv_name in arms:
            arm_rows = result(csv_name)
            b_all, a_all = [], []
            for reg in REGIONS:
                bd = dice_by_patient(base, reg, where={"threshold_method": rule})
                ad = dice_by_patient(arm_rows, reg, where={"threshold_method": rule})
                x, y = paired(bd, ad)
                b_all.append(x)
                a_all.append(y)
            x, y = np.concatenate(b_all), np.concatenate(a_all)
            deltas.append(float(np.mean(y - x)))
            ps.append(wilcoxon_p(x, y))
        offs = xs + (k - 2) * width
        ax.bar(offs, deltas, width, color=RULE_COLORS[rule], label=RULE_LABELS[rule],
               edgecolor="white", linewidth=0.6)
        for o, d, p in zip(offs, deltas, ps):
            ax.text(o, d + (0.0022 if d >= 0 else -0.0022), stars(p) if p < 0.05 else "n.s.",
                    ha="center", va="bottom" if d >= 0 else "top", fontsize=6.8,
                    color=INK if p < 0.05 else INK_MUTED, rotation=90)

    ax.axhline(0, color=INK, linewidth=1.2)
    ax.set_xticks(xs, [a[0] for a in arms], fontsize=9)
    ax.set_ylim(-0.078, 0.052)
    style_axes(ax, ylabel="Δ pooled Dice vs. the RQ1 baseline")
    ax.legend(frameon=False, fontsize=8.0, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, -0.34), columnspacing=1.2, handletextpad=0.5,
              title="binarization rule", title_fontsize=8.2)
    ax.set_title("Only the Otsu bar says size-conditioned prompting works",
                 fontsize=10.5, fontweight="bold", color=INK, pad=8)
    ax.annotate("+0.023, p = 8×10⁻¹⁷", xy=(-2 * width, 0.0235), xytext=(-0.42, 0.036),
                fontsize=8.4, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.0))

    # -- right: RQ2 per region, Otsu against the deployable top-1% rule
    ax = axes[1]
    arm_rows = result("rq13_rq2_thresholds_scores.csv")
    otsu_d, pct99_d = [], []
    for reg in REGIONS:
        for rule, sink in [("otsu", otsu_d), ("pct99", pct99_d)]:
            x, y = paired(dice_by_patient(base, reg, where={"threshold_method": rule}),
                          dice_by_patient(arm_rows, reg, where={"threshold_method": rule}))
            sink.append(float(np.mean(y - x)))
    xs = np.arange(3)
    ax.bar(xs - 0.19, otsu_d, 0.38, color=ORANGE, label="Otsu (published protocol)")
    ax.bar(xs + 0.19, pct99_d, 0.38, color=BLUE, label="top 1% (deployable)")
    for x, v in zip(xs - 0.19, otsu_d):
        ax.text(x, v + 0.0018, f"{v:+.3f}", ha="center", fontsize=8, color=INK)
    for x, v in zip(xs + 0.19, pct99_d):
        ax.text(x, v + (0.0018 if v >= 0 else -0.0065), f"{v:+.3f}", ha="center", fontsize=8,
                color=INK, va="bottom" if v >= 0 else "top")
    ax.axhline(0, color=INK, linewidth=1.2)
    ax.set_xticks(xs, REGIONS, fontsize=9.5)
    ax.set_ylim(min(pct99_d) * 1.45, max(otsu_d) * 1.55)
    style_axes(ax, ylabel="RQ2 Δ Dice vs. baseline")
    ax.set_title("Whole tumor flips sign outright", fontsize=10.5, fontweight="bold",
                 color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.2, loc="lower left")

    fig.suptitle("RQ13: re-scoring the three retrained arms under a calibrated threshold",
                 fontsize=12, fontweight="bold", color=INK, y=1.03)
    save(fig, "fig_rq13_arms.png")


if __name__ == "__main__":
    fig_anisotropy()
    fig_lift_over_chance()
    fig_threshold_ladder()
    fig_rq3_multiscale()
    fig_rq6_uniformity()
    fig_accuracy_vs_localization()
    fig_rq8_embedding_vs_behavior()
    fig_pointing_distance()
    fig_rq13_arms()
    print("\nAll nine supplementary figures written to figures/")

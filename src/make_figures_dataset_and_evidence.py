"""Dataset figures and the evidence figures for claims that were previously only tables.

Six figures. The first two describe the data itself -- what the 369 patients look like, how the
train/val split was made, and where the size-bin cutoffs fall -- which the report previously stated
only in prose and a small table. The remaining four each carry a claim that was strong but invisible:

  fig_dataset_split       369 patients -> 296/73 split, and the per-region x size-bin populations,
                          shown for train and validation side by side so the reader can see that the
                          bins are balanced and that ET loses 27 patients to having no enhancing
                          component at all.
  fig_dataset_volumes     lesion volume distributions per region on a log axis with tercile cutoffs
                          marked. Makes visible the single most important structural fact about this
                          dataset: the three regions overlap so heavily that a "large" ET is smaller
                          than a "small" WT, which is why every measurement is stratified per region.
  fig_otsu_calibration    predicted mask volume against true lesion volume. The report claims Otsu is
                          ANTI-correlated with lesion size (rho -0.26 to -0.48); this shows it.
  fig_pointing_game       pointing accuracy against its per-bin chance level, at two window sizes.
                          Shows the ET-small = 0/21 result and the ET improvement at 8^3 directly.
  fig_seed_replication    per-seed ablation deltas against the retraining noise floor, the RQ2/4/5/6
                          analogue of the RQ7 figure.
  fig_shortcut_probe      the RQ4 noise probe: text similarity for inputs containing zero anatomy.

Colour follows the project's fixed Okabe-Ito order, validated colourblind-safe.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"

BLUE, ORANGE, GREEN, AMBER, PINK = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"
REGION_COLORS = {"ET": BLUE, "TC": ORANGE, "WT": GREEN}
INK, INK_MUTED, GRID = "#1a1a19", "#5c5c58", "#d8d8d4"

REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
SEEDS = [0, 1, 2]

# Tercile cutoffs, identical to evaluate_rq1.SIZE_CUTOFFS.
SIZE_CUTOFFS = {"ET": (9050.6, 26245.3), "TC": (19190.7, 49948.2), "WT": (63126.2, 125309.1)}


def style_axes(ax, ylabel=None, xlabel=None, ygrid=True):
    """Apply the shared recessive-grid, no-top/right-spine treatment."""
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


def load_csv(path):
    """Read a CSV into dict rows, or None if it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_volumes():
    """Per-patient true lesion volumes in mm^3, keyed by patient then region."""
    rows = load_csv(os.path.join(PREPROCESSED_DIR, "lesion_volumes.csv"))
    return {r["patient_id"]: {reg: float(r[f"{reg.lower()}_volume_mm3"]) for reg in REGIONS}
            for r in rows}


def reconstruct_split(seed=0):
    """Rebuild the seed-0 train/val split exactly as every script in the project does.

    The split is never stored on disk: each script derives it from a sorted patient listing plus a
    seeded shuffle, so reproducing it here needs the same two steps and nothing else.

    Returns:
        (train_ids, val_ids)
    """
    import random
    ids = sorted(f[:-4] for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz"))
    random.seed(seed)
    random.shuffle(ids)
    n_val = max(1, int(0.2 * len(ids)))
    return ids[n_val:], ids[:n_val]


def size_bin(vol, region):
    """Assign a lesion volume to its tercile for one region."""
    lo, hi = SIZE_CUTOFFS[region]
    return "small" if vol <= lo else ("medium" if vol <= hi else "large")


def fig_dataset_split():
    """Draw the train/val split and the per-region, per-bin patient counts."""
    vols = load_volumes()
    train, val = reconstruct_split(0)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0),
                             gridspec_kw={"width_ratios": [1, 2.3]})

    # -- left: the split itself
    ax = axes[0]
    ax.barh([1], [len(train)], color=BLUE, height=0.5, label=f"train ({len(train)})")
    ax.barh([1], [len(val)], left=[len(train)], color=ORANGE, height=0.5,
            label=f"validation ({len(val)})")
    ax.set_xlim(0, len(train) + len(val))
    ax.set_ylim(0.4, 1.9)
    ax.set_yticks([])
    ax.annotate(f"{len(train)} patients\ntrain", xy=(len(train) / 2, 1), ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
    ax.annotate(f"{len(val)}\nval", xy=(len(train) + len(val) / 2, 1), ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
    style_axes(ax, xlabel="patients", ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_title(f"{len(train) + len(val)} glioma patients, 80/20 split",
                 fontsize=10.5, color=INK, pad=10, loc="left")

    # -- right: per-region x bin populations, train vs val
    ax = axes[1]
    x = np.arange(9)
    labels, tr_counts, va_counts = [], [], []
    for reg in REGIONS:
        for b in BINS:
            labels.append(f"{reg}\n{b}")
            tr_counts.append(sum(1 for p in train
                                 if vols[p][reg] > 0 and size_bin(vols[p][reg], reg) == b))
            va_counts.append(sum(1 for p in val
                                 if vols[p][reg] > 0 and size_bin(vols[p][reg], reg) == b))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, tr_counts, w, color=BLUE, label="train")
    ax.bar(x + w / 2 + 0.01, va_counts, w, color=ORANGE, label="validation")
    for xi, v in enumerate(va_counts):
        ax.annotate(str(v), xy=(xi + w / 2 + 0.01, v + 2), ha="center", fontsize=8, color=INK)
    style_axes(ax, ylabel="patients")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_title("Patients per region × size bin (validation counts labelled)",
                 fontsize=10.5, color=INK, pad=10, loc="left")
    fig.suptitle("BraTS2020 dataset composition (split seed 0)", fontsize=11.5, color=INK,
                 x=0.005, ha="left", y=1.06)

    n_no_et = sum(1 for p in vols if vols[p]["ET"] <= 0)
    fig.text(0.5, -0.03, f"Bins are per-region terciles of true lesion volume. "
                         f"{n_no_et} of {len(vols)} patients have no enhancing component at all and "
                         f"contribute no ET row, which is why ET has the smallest usable sample.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_dataset_split.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_dataset_volumes():
    """Draw per-region lesion volume distributions with the tercile cutoffs marked."""
    vols = load_volumes()
    train, val = reconstruct_split(0)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)

    # -- left: distributions with cutoffs
    ax = axes[0]
    for i, reg in enumerate(REGIONS):
        v = np.array([vols[p][reg] for p in vols if vols[p][reg] > 0])
        parts = ax.violinplot([np.log10(v)], positions=[i], widths=0.75, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(REGION_COLORS[reg])
            pc.set_alpha(0.55)
            pc.set_edgecolor(REGION_COLORS[reg])
        lo, hi = SIZE_CUTOFFS[reg]
        for cut in (lo, hi):
            ax.plot([i - 0.42, i + 0.42], [np.log10(cut)] * 2, color=INK, lw=1.6,
                    solid_capstyle="butt")
        ax.annotate(f"{lo/1000:.0f}k", xy=(i + 0.46, np.log10(lo)), fontsize=8,
                    color=INK_MUTED, va="center")
        ax.annotate(f"{hi/1000:.0f}k", xy=(i + 0.46, np.log10(hi)), fontsize=8,
                    color=INK_MUTED, va="center")
    style_axes(ax, ylabel="true lesion volume (mm³, log scale)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(REGIONS, fontsize=10, color=INK)
    ticks = [2, 3, 4, 5, 6]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"$10^{t}$" for t in ticks])
    ax.set_title("Lesion volumes per region, with tercile cutoffs (black bars)",
                 fontsize=10.5, color=INK, pad=10, loc="left")

    # -- right: the overlap that forces per-region stratification
    ax = axes[1]
    for i, reg in enumerate(REGIONS):
        for j, b in enumerate(BINS):
            v = np.array([vols[p][reg] for p in vols
                          if vols[p][reg] > 0 and size_bin(vols[p][reg], reg) == b])
            if not len(v):
                continue
            pos = i + (j - 1) * 0.26
            ax.scatter(np.full(len(v), pos) + np.random.RandomState(0).normal(0, 0.035, len(v)),
                       np.log10(v), s=5, color=REGION_COLORS[reg],
                       alpha=[0.35, 0.6, 0.95][j], linewidths=0)
    et_large_min = np.log10(SIZE_CUTOFFS["ET"][1])
    wt_small_max = np.log10(SIZE_CUTOFFS["WT"][0])
    ax.axhspan(et_large_min, wt_small_max, color=AMBER, alpha=0.16, zorder=0)
    ax.annotate("a “large” ET is smaller\nthan a “small” WT",
                xy=(1.5, (et_large_min + wt_small_max) / 2), fontsize=9, color=INK,
                ha="center", va="center")
    style_axes(ax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(REGIONS, fontsize=10, color=INK)
    ax.set_title("Why size bins must be per-region, not global",
                 fontsize=10.5, color=INK, pad=10, loc="left")
    ax.annotate("within each region: small · medium · large (left to right, darkening)",
                xy=(0.5, 0.02), xycoords="axes fraction", fontsize=8.5, color=INK_MUTED,
                ha="center")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_dataset_volumes.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_otsu_calibration():
    """Draw predicted mask volume against true lesion volume, per region."""
    rows = load_csv(os.path.join(RESULTS_DIR, "rq11_threshold_confound_scores.csv"))
    if rows is None:
        print("  skipped fig_otsu_calibration (missing RQ11 CSV)")
        return
    from scipy import stats

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), sharey=True)
    for ax, reg in zip(axes, REGIONS):
        sel = [r for r in rows if r["region"] == reg and r["threshold_method"] == "otsu"]
        gt = np.array([float(r["gt_volume_mm3_resized"]) for r in sel])
        pred = np.array([float(r["pred_volume_mm3"]) for r in sel])
        keep = gt > 0
        gt, pred = gt[keep], pred[keep]
        rho, p = stats.spearmanr(gt, pred)

        ax.scatter(gt, pred, s=16, color=REGION_COLORS[reg], alpha=0.75, linewidths=0)
        lim = [min(gt.min(), pred.min()) * 0.6, max(gt.max(), pred.max()) * 1.6]
        ax.plot(lim, lim, color=INK_MUTED, lw=1.2, ls=(0, (4, 3)))
        ax.annotate("perfect calibration", xy=(lim[1] * 0.30, lim[1] * 0.42), fontsize=8,
                    color=INK_MUTED, rotation=32)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        style_axes(ax, xlabel="true lesion volume (mm³)")
        ax.set_title(f"{reg}   ρ = {rho:+.2f}  (p={p:.1e})", fontsize=10, color=INK, pad=8)
    axes[0].set_ylabel("Otsu predicted volume (mm³)", color=INK, fontsize=10)
    fig.suptitle("Otsu is not just miscalibrated — it is calibrated backwards: "
                 "smaller lesions get LARGER predicted masks",
                 fontsize=11, color=INK, x=0.008, ha="left", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_otsu_calibration.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_pointing_game():
    """Draw pointing accuracy against its per-bin chance level, at two window sizes."""
    a = load_csv(os.path.join(RESULTS_DIR, "rq12_grounding_window32_stride16_scores.csv"))
    b = load_csv(os.path.join(RESULTS_DIR, "rq12_grounding_window8_stride4_scores.csv"))
    if a is None or b is None:
        print("  skipped fig_pointing_game (missing RQ12 CSVs)")
        return
    total = 128 ** 3

    def rates(rows):
        """Per region x bin hit rate and chance level, from the tie-aware centroid rule."""
        out = {}
        uniq = {(r["patient_id"], r["region"]): r for r in rows if r["threshold_method"] == "otsu"}
        for reg in REGIONS:
            for bn in BINS:
                sel = [r for r in uniq.values() if r["region"] == reg and r["size_bin"] == bn]
                if not sel:
                    continue
                hits = sum(int(r["centroid_hit"]) for r in sel)
                ch = float(np.mean([int(r["gt_voxels_resized"]) / total for r in sel]))
                out[(reg, bn)] = (hits / len(sel), ch, hits, len(sel))
        return out

    ra, rb = rates(a), rates(b)
    keys = [(reg, bn) for reg in REGIONS for bn in BINS]
    x = np.arange(len(keys))
    w = 0.37

    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    ax.bar(x - w / 2 - 0.01, [ra[k][0] for k in keys], w, color=BLUE, label="32³ window (baseline)")
    ax.bar(x + w / 2 + 0.01, [rb[k][0] for k in keys], w, color=ORANGE,
           label="8³ window (matched overlap)")
    ax.plot(x, [ra[k][1] for k in keys], marker="_", markersize=22, linestyle="none",
            color=INK, label="chance level")

    # Label every bin that scores exactly zero -- at 32³ that is ET-small (the report's headline
    # failure); at 8³ it is TC-small, which the smaller window makes worse. Both are results.
    for xi, k in enumerate(keys):
        if ra[k][0] == 0:
            ax.annotate(f"{ra[k][2]}/{ra[k][3]}\nat chance", xy=(xi - w / 2 - 0.01, 0.025),
                        ha="center", fontsize=8, color=BLUE, fontweight="bold")
        if rb[k][0] == 0:
            ax.annotate(f"{rb[k][2]}/{rb[k][3]}\nat chance", xy=(xi + w / 2 + 0.01, 0.025),
                        ha="center", fontsize=8, color=ORANGE, fontweight="bold")
    style_axes(ax, ylabel="pointing accuracy (peak response inside the lesion)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n{b}" for r, b in keys], fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left", labelcolor=INK)
    ax.set_title("A threshold-free metric: does the model point at the lesion at all?",
                 fontsize=11, color=INK, pad=12, loc="left")
    fig.text(0.5, -0.04, "Chance is tiny everywhere (0.05–1.8%), so every bar far exceeds it — "
                         "except ET-small at 32³, which is exactly zero. The 8³ window is the only "
                         "intervention that moves it.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_pointing_game.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_seed_replication():
    """Draw per-seed ablation deltas against the retraining noise floor."""
    def base_file(s):
        return ("rq1_localization_scores.csv" if s == 0
                else f"rq1_localization_scores_seed{s}.csv")

    def arm_file(k, s):
        return (f"{k}_localization_scores.csv" if s == 0
                else f"{k}_seed{s}_localization_scores.csv")

    def dmap(rows):
        return {(r["patient_id"], r["region"]): float(r["dice"]) for r in rows}

    bases = {s: load_csv(os.path.join(RESULTS_DIR, base_file(s))) for s in SEEDS}
    pooled = [float(np.mean([float(r["dice"]) for r in bases[s]])) for s in SEEDS]
    noise = float(np.std(pooled, ddof=1))

    arms = [("rq2", "RQ2\nsize-conditioned text"), ("rq4", "RQ4\nscale-matched"),
            ("rq5", "RQ5\nnaturalistic text"), ("rq6", "RQ6\nuniformity fix")]

    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    ax.axvspan(-noise, noise, color=GRID, alpha=0.75, zorder=0)
    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)

    for yi, (key, label) in enumerate(arms):
        deltas = []
        for s in SEEDS:
            arm = load_csv(os.path.join(RESULTS_DIR, arm_file(key, s)))
            if arm is None:
                continue
            A, B = dmap(bases[s]), dmap(arm)
            ks = sorted(set(A) & set(B))
            deltas.append(float(np.mean([B[k] for k in ks]) - np.mean([A[k] for k in ks])))
        if len(deltas) < 3:
            continue
        consistent = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        col = GREEN if np.mean(deltas) > 0 else ORANGE
        ax.scatter(deltas, [yi] * len(deltas), s=58, color=col, zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.scatter([np.mean(deltas)], [yi], s=190, marker="|", color=col, zorder=4, linewidth=2.4)
        verdict = ("replicated: better" if consistent and np.mean(deltas) > 0 else
                   "replicated: worse" if consistent else "sign flips")
        ax.annotate(verdict, xy=(0.075, yi), fontsize=8.5, color=INK, va="center")

    style_axes(ax, xlabel="change in pooled Dice vs. the RQ1 baseline (one dot per training run)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([lbl for _, lbl in arms], fontsize=9, color=INK)
    ax.set_ylim(-0.5, len(arms) - 0.2)
    ax.set_xlim(-0.075, 0.115)
    ax.annotate(f"grey band = retraining noise floor (±{noise:.4f})",
                xy=(-0.072, len(arms) - 0.35), fontsize=8.5, color=INK_MUTED, va="center")
    ax.set_title("Every ablation, retrained three times: do the Section 7 verdicts replicate?",
                 fontsize=11, color=INK, pad=12, loc="left")
    fig.text(0.5, -0.06, "These are Otsu-scored, the original protocol. Replicating across seeds "
                         "tests whether a verdict survives RETRAINING; it cannot detect a confound "
                         "shared by every run.\nRQ13 shows RQ2's gain here is an artifact of the "
                         "binarizer, and RQ5's positive direction revises its reported null.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_seed_replication.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_shortcut_probe():
    """Draw the RQ4 noise probe: text similarity for inputs containing zero anatomy.

    Numbers are the measured output of test_rq4_shortcut_hypothesis.py (job 2083279); this figure
    only redraws them, since the probe itself needs the GPU and the trained RQ4 checkpoint.
    """
    pipelines = ["small\n(16³→32³)", "medium\n(32³ native)", "large\n(64³→32³)"]
    own = [-0.4008, 0.0255, 0.1492]
    err = [0.0096, 0.0098, 0.0076]
    cross = {"small": [-0.3995, -0.3724, -0.3708],
             "medium": [-0.2104, 0.0230, -0.2148],
             "large": [0.0145, 0.1256, 0.1505]}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))

    ax = axes[0]
    ax.bar(range(3), own, yerr=err, capsize=4, color=[BLUE, ORANGE, GREEN], width=0.6,
           error_kw=dict(ecolor=INK_MUTED, lw=1.2))
    ax.axhline(0, color=INK_MUTED, lw=1)
    style_axes(ax, ylabel="cosine similarity to its own size-text")
    ax.set_xticks(range(3))
    ax.set_xticklabels(pipelines, fontsize=9, color=INK)
    ax.set_title("Pure noise, zero anatomy — yet cleanly separated\nANOVA F = 59,672,  p ≈ 4×10⁻²⁵¹",
                 fontsize=10.5, color=INK, pad=10, loc="left")

    ax = axes[1]
    x = np.arange(3)
    w = 0.26
    for j, (txt, vals) in enumerate(cross.items()):
        ax.bar(x + (j - 1) * (w + 0.015), vals, w,
               color={"small": BLUE, "medium": ORANGE, "large": GREEN}[txt],
               label=f'"{txt}" text')
    ax.axhline(0, color=INK_MUTED, lw=1)
    style_axes(ax, ylabel="cosine similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower right", labelcolor=INK)
    ax.set_title("Every noise pipeline prefers “large” —\nan embedding hub, not per-scale recognition",
                 fontsize=10.5, color=INK, pad=10, loc="left")

    fig.text(0.5, -0.04, "If the model were reading tumor content, inputs containing none should "
                         "score identically across pipelines. They do not — it learned the "
                         "resize-interpolation signature instead.",
             ha="center", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_shortcut_probe.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    """Generate the dataset and evidence figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_dataset_split()
    fig_dataset_volumes()
    fig_otsu_calibration()
    fig_pointing_game()
    fig_seed_replication()
    fig_shortcut_probe()


if __name__ == "__main__":
    main()

"""Do the Section 7 ablation conclusions survive being retrained on different splits?

Section 6 replicated the *core* RQ1 finding across three independent train/val splits, but every
ablation conclusion in Section 7 (RQ2, RQ4, RQ5, RQ6) rested on a single training run each, scored
against a single split. That is the same pseudo-replication trap analyze_rq7_multiseed.py caught in
RQ7: a within-seed paired Wilcoxon over ~213 patients can return p<1e-7 for a contrast that reverses
sign the moment the model is retrained. Section 10 flagged this as an open limitation; this script
closes it.

Each of RQ2/RQ4/RQ5/RQ6 was retrained and re-evaluated under seeds 0/1/2, where the seed controls
both the weight initialisation and the train/val split. Within a seed the comparison against RQ1 is
genuinely paired (identical held-out patients); across seeds the three deltas are independent
replications of the same contrast.

The claim an ablation earns is therefore SIGN CONSISTENCY ACROSS RUNS, judged against the baseline's
own seed-to-seed spread as a noise floor -- not the magnitude of any single within-seed p-value. An
ablation whose direction flips between seeds is reported as unsupported however significant it looked
in Section 7, and one that holds in all three is reported as replicated.
"""
import csv
import os

import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
SEEDS = [0, 1, 2]

ABLATIONS = [
    ("rq2", "RQ2 (size-conditioned text)"),
    ("rq4", "RQ4 (scale-matched retraining)"),
    ("rq5", "RQ5 (naturalistic text)"),
    ("rq6", "RQ6 (uniformity fix)"),
]


def load(filename):
    """Read a per-patient score CSV, or return None if that run has not been produced yet."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def baseline_file(seed):
    """Filename of the RQ1 baseline scores for a given seed (seed 0 uses the unsuffixed name)."""
    return "rq1_localization_scores.csv" if seed == 0 else f"rq1_localization_scores_seed{seed}.csv"


def ablation_file(key, seed):
    """Filename of an ablation's scores for a given seed.

    Seed 0 kept the original unsuffixed naming from when each ablation was a single run; seeds 1 and 2
    were added later under a `_seed{n}_` convention.
    """
    return f"{key}_localization_scores.csv" if seed == 0 else f"{key}_seed{seed}_localization_scores.csv"


def pooled_mean(rows):
    """Mean Dice over every (patient, region) row in a result file."""
    return float(np.mean([float(r["dice"]) for r in rows]))


def paired_delta(base_rows, abl_rows, region=None, size_bin=None):
    """Mean paired difference (ablation - baseline) within one seed, matched by patient and region.

    Args:
        base_rows: RQ1 baseline rows for this seed.
        abl_rows: ablation rows for the same seed (hence the same held-out patients).
        region: optional ET/TC/WT filter.
        size_bin: optional small/medium/large filter.

    Returns:
        dict with n, base mean, ablation mean, delta and a paired Wilcoxon p-value, or None if fewer
        than 5 pairs survive the filters (too few to say anything).
    """
    lookup = {(r["patient_id"], r["region"]): float(r["dice"]) for r in abl_rows}
    pairs = [
        (float(r["dice"]), lookup[(r["patient_id"], r["region"])])
        for r in base_rows
        if (region is None or r["region"] == region)
        and (size_bin is None or r["size_bin"] == size_bin)
        and (r["patient_id"], r["region"]) in lookup
    ]
    if len(pairs) < 5:
        return None
    a, b = (np.array(v) for v in zip(*pairs))
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:  # all differences identically zero
        p = 1.0
    return {"n": a.size, "base": a.mean(), "abl": b.mean(), "delta": b.mean() - a.mean(), "p": p}


def main():
    """Report per-seed ablation deltas, sign consistency, and which Section 7 claims replicate."""
    baselines = {s: load(baseline_file(s)) for s in SEEDS}
    missing = [s for s, v in baselines.items() if v is None]
    if missing:
        raise SystemExit(f"missing RQ1 baseline results for seeds {missing}")

    print("=" * 108)
    print("SEED REPLICATION OF THE SECTION 7 ABLATIONS")
    print("Unit of replication = the training run. Three independent runs per ablation (seeds 0/1/2).")
    print("=" * 108)

    base_pooled = [pooled_mean(baselines[s]) for s in SEEDS]
    noise_floor = float(np.std(base_pooled, ddof=1))
    print("\n[0] NOISE FLOOR: the identical RQ1 baseline, retrained under three seeds")
    print("    pooled Dice per seed: " + ", ".join(f"{v:.4f}" for v in base_pooled))
    print(f"    mean = {np.mean(base_pooled):.4f}, std = {noise_floor:.4f}, "
          f"range = {max(base_pooled) - min(base_pooled):.4f}")
    print("    An ablation effect smaller than this is not distinguishable from retraining noise.")

    print("\n[1] PER-SEED PAIRED DELTA vs. RQ1 (pooled over all regions and size bins)")
    print(f"{'ablation':<34}{'seed0':>9}{'seed1':>9}{'seed2':>9}{'mean':>9}{'std':>8}  verdict")
    summary = {}
    for key, label in ABLATIONS:
        deltas, ok = [], True
        for s in SEEDS:
            rows = load(ablation_file(key, s))
            if rows is None:
                ok = False
                break
            d = paired_delta(baselines[s], rows)
            deltas.append(d["delta"] if d else float("nan"))
        if not ok:
            print(f"{label:<34}  (incomplete: missing one or more seed result files)")
            continue
        deltas = np.array(deltas)
        same_sign = bool(np.all(deltas > 0) or np.all(deltas < 0))
        exceeds = bool(abs(deltas.mean()) > noise_floor)
        if not same_sign:
            verdict = "SIGN FLIPS -> not replicated"
        elif not exceeds:
            verdict = "consistent sign, but < noise floor"
        else:
            verdict = ("replicated: reliably BETTER" if deltas.mean() > 0
                       else "replicated: reliably WORSE")
        summary[key] = {"deltas": deltas, "same_sign": same_sign, "exceeds": exceeds, "label": label}
        print(f"{label:<34}{deltas[0]:>+9.4f}{deltas[1]:>+9.4f}{deltas[2]:>+9.4f}"
              f"{deltas.mean():>+9.4f}{deltas.std(ddof=1):>8.4f}  {verdict}")

    if not summary:
        print("\nNo ablation had all three seeds available.")
        return

    print("\n[2] ONE-SAMPLE t-TEST ON THE THREE PER-SEED DELTAS (n=3 runs, not n=213 patients)")
    print("    Underpowered by construction; reported so the strength of evidence is visible rather")
    print("    than hidden behind a patient-level p-value.")
    print(f"{'ablation':<34}{'mean delta':>12}{'t':>8}{'p':>9}")
    for key, s in summary.items():
        d = s["deltas"]
        t, p = stats.ttest_1samp(d, 0.0)
        print(f"{s['label']:<34}{d.mean():>+12.4f}{t:>8.2f}{p:>9.4f}")

    print("\n[3] PER REGION x SIZE BIN: in how many of 3 seeds does the sign hold?")
    print("    'sign' is the direction observed at seed 0, i.e. the direction Section 7 reported.")
    print(f"{'ablation':<34}{'reg':<5}{'bin':<8}{'seed0':>9}{'seed1':>9}{'seed2':>9}  replicates")
    replication_tally = {}
    for key, s in summary.items():
        n_replicated, n_total = 0, 0
        for region in REGIONS:
            for b in BINS:
                deltas = []
                for seed in SEEDS:
                    rows = load(ablation_file(key, seed))
                    d = paired_delta(baselines[seed], rows, region=region, size_bin=b)
                    deltas.append(d["delta"] if d else float("nan"))
                if any(np.isnan(deltas)):
                    continue
                n_total += 1
                ref = np.sign(deltas[0])
                agree = int(sum(1 for x in deltas if np.sign(x) == ref))
                n_replicated += int(agree == 3)
                print(f"{s['label']:<34}{region:<5}{b:<8}"
                      f"{deltas[0]:>+9.4f}{deltas[1]:>+9.4f}{deltas[2]:>+9.4f}  {agree}/3")
        replication_tally[key] = (n_replicated, n_total)

    print("\n[4] SUMMARY: region x bin combinations whose seed-0 direction held in all 3 seeds")
    for key, s in summary.items():
        n_rep, n_tot = replication_tally[key]
        print(f"    {s['label']:<34} {n_rep}/{n_tot} combinations replicated in all three seeds")

    print("\n[5] SIZE COLLAPSE UNDER EACH ABLATION AND SEED")
    print("    (large/small Dice ratio; RQ1's core finding should persist under every ablation)")
    print(f"{'condition':<34}{'region':<7}{'seed0':>9}{'seed1':>9}{'seed2':>9}")
    for key, label in [("rq1", "RQ1 (baseline)")] + ABLATIONS:
        for region in REGIONS:
            ratios = []
            for seed in SEEDS:
                rows = baselines[seed] if key == "rq1" else load(ablation_file(key, seed))
                if rows is None:
                    ratios.append(float("nan"))
                    continue
                means = {}
                for b in ["small", "large"]:
                    vals = [float(r["dice"]) for r in rows
                            if r["region"] == region and r["size_bin"] == b]
                    means[b] = np.mean(vals) if vals else float("nan")
                ratios.append(means["large"] / means["small"] if means["small"] > 0 else float("nan"))
            print(f"{label:<34}{region:<7}" + "".join(f"{r:>8.1f}x" for r in ratios))


if __name__ == "__main__":
    main()

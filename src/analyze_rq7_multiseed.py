"""RQ7 multi-seed analysis: does the text encoder affect localization, or was that run-to-run noise?

The single-seed RQ7 analysis produced patient-paired p-values as small as 1e-35, which vastly
overstates the evidence. Each text condition was trained exactly once, so the unit of randomization
is the *training run*, not the patient; treating 213 patients from one run as 213 independent
observations is pseudo-replication. The check that exposed this: the PubMedBERT baseline, retrained
under three different seeds, varies by 0.0086 pooled Dice on its own -- comparable to several of the
between-condition differences being claimed.

This script does the analysis properly. Every condition now has three independent training runs
(seeds 0/1/2), and each seed defines its own train/val split, so:

  - within a seed, variant vs. PubMedBERT is genuinely paired (identical held-out patients), and
  - across seeds, the three deltas are independent replications of the same contrast.

The evidence for an effect is therefore CONSISTENCY ACROSS SEEDS, not the size of a within-seed
p-value. A contrast that flips sign between seeds is noise no matter how significant it looks in
any single run. We report the per-seed deltas, their mean and spread, and compare that spread
against the baseline's own seed-to-seed variability as the noise floor.
"""
import csv
import os

import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
SEEDS = [0, 1, 2]

VARIANTS = [
    ("bertbase", "BERT-base (general domain)"),
    ("randbert", "random-init BERT (frozen)"),
    ("randvec_ortho", "4 random orthonormal vectors"),
    ("randvec_aniso", "random vectors, PubMedBERT geometry"),
]


def load(filename):
    """Read a per-patient score CSV from results/ into a list of dict rows.

    Args:
        filename: basename of the CSV inside the results directory.

    Returns:
        List of dicts, one per (patient, region) row.
    """
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def baseline_file(seed):
    """Filename of the PubMedBERT baseline scores for a seed (seed 0 is unsuffixed)."""
    return "rq1_localization_scores.csv" if seed == 0 else f"rq1_localization_scores_seed{seed}.csv"


def variant_file(key, seed):
    """Filename of a text-encoder variant's scores for a seed (seed 0 is unsuffixed)."""
    return f"rq7_{key}_localization_scores.csv" if seed == 0 else f"rq7_{key}_seed{seed}_localization_scores.csv"


def pooled_mean(rows):
    """Mean Dice over every (patient, region) row in a result file."""
    return float(np.mean([float(r["dice"]) for r in rows]))


def paired_delta(base_rows, var_rows, region=None, size_bin=None):
    """Mean paired difference (variant - baseline) within one seed, matched by patient and region."""
    lookup = {(r["patient_id"], r["region"]): float(r["dice"]) for r in var_rows}
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
    return {"n": a.size, "base": a.mean(), "var": b.mean(), "delta": b.mean() - a.mean()}


def main():
    """Judge each text condition by sign consistency across training runs, not within-run p-values."""
    baselines = {s: load(baseline_file(s)) for s in SEEDS}
    missing_base = [s for s, v in baselines.items() if v is None]
    if missing_base:
        raise SystemExit(f"missing baseline results for seeds {missing_base}")

    print("=" * 104)
    print("RQ7 MULTI-SEED: is the text encoder's effect on localization real, or run-to-run noise?")
    print("Unit of replication = the training run. Three independent runs per condition.")
    print("=" * 104)

    base_pooled = [pooled_mean(baselines[s]) for s in SEEDS]
    noise_floor = float(np.std(base_pooled, ddof=1))
    print("\n[0] NOISE FLOOR: the identical PubMedBERT model, retrained under three seeds")
    print(f"    pooled Dice per seed: " + ", ".join(f"{v:.4f}" for v in base_pooled))
    print(f"    mean = {np.mean(base_pooled):.4f}, std = {noise_floor:.4f}, range = {max(base_pooled)-min(base_pooled):.4f}")
    print("    Any between-condition effect smaller than this is not interpretable.")

    print("\n[1] PER-SEED PAIRED DELTA vs. PubMedBERT (pooled over all regions and bins)")
    print(f"{'text condition':<38}{'seed0':>9}{'seed1':>9}{'seed2':>9}{'mean':>9}{'std':>8}  consistent?")
    summary = {}
    for key, label in VARIANTS:
        deltas = []
        ok = True
        for s in SEEDS:
            var_rows = load(variant_file(key, s))
            if var_rows is None:
                ok = False
                break
            d = paired_delta(baselines[s], var_rows)
            deltas.append(d["delta"] if d else float("nan"))
        if not ok:
            print(f"{label:<38}  (incomplete: missing one or more seed result files)")
            continue
        deltas = np.array(deltas)
        same_sign = bool(np.all(deltas > 0) or np.all(deltas < 0))
        exceeds = bool(abs(deltas.mean()) > noise_floor)
        verdict = ("all 3 same sign" if same_sign else "SIGN FLIPS -> noise")
        if same_sign and exceeds:
            verdict += ", > noise floor"
        elif same_sign:
            verdict += ", but < noise floor"
        summary[key] = {"deltas": deltas, "same_sign": same_sign, "exceeds": exceeds, "label": label}
        print(f"{label:<38}{deltas[0]:>+9.4f}{deltas[1]:>+9.4f}{deltas[2]:>+9.4f}"
              f"{deltas.mean():>+9.4f}{deltas.std(ddof=1):>8.4f}  {verdict}")

    if not summary:
        print("\nNo variant had all three seeds available; rerun once the evaluations finish.")
        return

    print("\n[2] VERDICT PER CONDITION")
    for key, s in summary.items():
        d = s["deltas"]
        if not s["same_sign"]:
            v = "NOT SUPPORTED -- the effect changes sign across training runs"
        elif not s["exceeds"]:
            v = "NOT SUPPORTED -- consistent in sign but smaller than baseline seed noise"
        else:
            direction = "BETTER than" if d.mean() > 0 else "WORSE than"
            v = f"SUPPORTED -- reliably {direction} PubMedBERT (mean {d.mean():+.4f}, {abs(d.mean())/noise_floor:.1f}x noise floor)"
        print(f"    {s['label']:<38} {v}")

    print("\n[3] ONE-SAMPLE t-TEST ON THE THREE PER-SEED DELTAS (n=3 runs, not n=213 patients)")
    print("    Underpowered by construction; reported so the weakness of the evidence is visible")
    print("    rather than hidden behind a patient-level p-value.")
    print(f"{'text condition':<38}{'mean delta':>12}{'t':>8}{'p':>9}")
    for key, s in summary.items():
        d = s["deltas"]
        t, p = stats.ttest_1samp(d, 0.0)
        print(f"{s['label']:<38}{d.mean():>+12.4f}{t:>8.2f}{p:>9.4f}")

    print("\n[4] SIZE COLLAPSE ACROSS SEEDS AND TEXT CONDITIONS")
    print("    (large/small Dice ratio; the core RQ1 finding should hold everywhere if it is real)")
    print(f"{'text condition':<38}{'region':<7}{'seed0':>9}{'seed1':>9}{'seed2':>9}")
    all_conditions = [("pubmedbert", "PubMedBERT (baseline)")] + VARIANTS
    for key, label in all_conditions:
        for region in REGIONS:
            ratios = []
            for s in SEEDS:
                rows = baselines[s] if key == "pubmedbert" else load(variant_file(key, s))
                if rows is None:
                    ratios.append(float("nan"))
                    continue
                means = {}
                for b_ in ["small", "large"]:
                    vals = [float(r["dice"]) for r in rows if r["region"] == region and r["size_bin"] == b_]
                    means[b_] = np.mean(vals) if vals else float("nan")
                ratios.append(means["large"] / means["small"] if means["small"] > 0 else float("nan"))
            print(f"{label:<38}{region:<7}" + "".join(f"{r:>8.1f}x" for r in ratios))


if __name__ == "__main__":
    main()

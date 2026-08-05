"""RQ7 analysis: how much does the text encoder contribute to *localization*?

The training logs already show that 4-way patch classification accuracy is statistically
indistinguishable across every text condition, including four random vectors carrying no language
at all. Classification is the easier task, though, and the project's actual claim is about
localization, so this script runs the comparison that matters: paired per-patient Dice, PubMedBERT
against each ablated text encoder, on the identical held-out split and identical protocol.

The framing is deliberately an EQUIVALENCE test rather than a difference test. The interesting
outcome here is a null -- "replacing PubMedBERT with random vectors costs nothing" -- and a
non-significant p-value alone cannot support that, since absence of evidence at n=21-32 per bin is
weak evidence of absence. So the argument rests on bootstrap confidence intervals on the paired
difference: if the CI is tight and contains zero, the conditions are demonstrably equivalent within
a stated tolerance. If it is wide, the honest conclusion is that this experiment cannot tell them
apart, which is a different and weaker claim.

Also reports the two-way split the ablation ladder was designed to separate:
  semantics  -- does meaningful text beat meaningless vectors? (pubmedbert vs randvec_aniso,
                which holds the embedding geometry fixed at PubMedBERT's own)
  geometry   -- does the embedding's anisotropy matter at all? (randvec_ortho vs randvec_aniso,
                which holds semantic content fixed at zero and varies only the geometry)
"""
import csv
import os

import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
BOOTSTRAP_RESAMPLES = 10000
SEED = 0
EQUIVALENCE_MARGIN = 0.01  # Dice units; below this a difference is not clinically meaningful

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


def as_lookup(rows):
    """Index rows by (patient_id, region) so two runs can be paired patient-by-patient.

    Args:
        rows: score rows from one run.

    Returns:
        dict mapping (patient_id, region) -> Dice.
    """
    return {(r["patient_id"], r["region"]): float(r["dice"]) for r in rows}


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation, the effect size paired with Wilcoxon.

    Reported alongside every p-value because with n=20-32 per bin a significant result can still be
    negligible in magnitude. Ranges from -1 (every pair moved down) to +1 (every pair moved up).

    Args:
        a: baseline values.
        b: comparison values, paired elementwise with a.

    Returns:
        float in [-1, 1].
    """
    diffs = np.asarray(b) - np.asarray(a)
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    pos, neg = ranks[diffs > 0].sum(), ranks[diffs < 0].sum()
    return float((pos - neg) / (pos + neg)) if (pos + neg) > 0 else 0.0


def bootstrap_ci(a, b, rng):
    """Percentile bootstrap 95% confidence interval on the mean paired difference.

    Used instead of a normal-theory interval because per-patient Dice is bounded, skewed, and often
    near zero for small lesions, so the sampling distribution of the mean is not close to Gaussian.

    Args:
        a: baseline values.
        b: comparison values, paired elementwise with a.
        rng: seeded numpy Generator, so intervals are reproducible.

    Returns:
        (low, high) bounds of the 95% interval on mean(b - a).
    """
    diffs = np.asarray(b) - np.asarray(a)
    idx = rng.integers(0, diffs.size, size=(BOOTSTRAP_RESAMPLES, diffs.size))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(baseline_rows, other_rows, rng, region=None, size_bin=None):
    """Compare one text-encoder variant against PubMedBERT over matched patients.

    Args:
        baseline_rows: PubMedBERT rows.
        other_rows: the variant's rows.
        rng: seeded numpy Generator for the bootstrap interval.
        region: optional ET/TC/WT filter.
        size_bin: optional small/medium/large filter.

    Returns:
        dict of n, means, delta, CI, effect size and Wilcoxon p, or None if too few pairs.
    """
    other = as_lookup(other_rows)
    pairs = [
        (float(r["dice"]), other[(r["patient_id"], r["region"])])
        for r in baseline_rows
        if (region is None or r["region"] == region)
        and (size_bin is None or r["size_bin"] == size_bin)
        and (r["patient_id"], r["region"]) in other
    ]
    if len(pairs) < 5:
        return None
    a, b = (np.array(v) for v in zip(*pairs))
    p = stats.wilcoxon(a, b).pvalue if not np.allclose(a, b) else 1.0
    lo, hi = bootstrap_ci(a, b, rng)
    return {"n": a.size, "base": a.mean(), "other": b.mean(), "delta": b.mean() - a.mean(),
            "lo": lo, "hi": hi, "eff": rank_biserial(a, b), "p": p}


def main():
    """Report whether swapping or destroying the text encoder changes localization at all.

    Includes an equivalence test rather than only a difference test: with these sample sizes, failing
    to reject the null is weak evidence, whereas showing the whole 95% interval lies inside a +-0.01
    Dice margin actively rules out a meaningful difference.
    """
    rng = np.random.default_rng(SEED)
    baseline = load("rq1_localization_scores.csv")
    if baseline is None:
        raise SystemExit("missing rq1_localization_scores.csv")

    loaded = {}
    for key, label in VARIANTS:
        rows = load(f"rq7_{key}_localization_scores.csv")
        if rows is None:
            print(f"MISSING: rq7_{key}_localization_scores.csv -- skipping {label}")
            continue
        loaded[key] = rows

    print("=" * 108)
    print("RQ7: does the text encoder contribute to LOCALIZATION?")
    print("Baseline = PubMedBERT (results/rq1_localization_scores.csv). Same split, same protocol.")
    print("=" * 108)

    print("\n[1] POOLED OVER ALL REGIONS AND SIZE BINS")
    print(f"{'text condition':<38}{'n':>4}{'PubMedBERT':>12}{'variant':>10}{'delta':>9}"
          f"{'95% CI':>21}{'effect':>8}{'p':>10}")
    pooled = {}
    for key, label in VARIANTS:
        if key not in loaded:
            continue
        c = compare(baseline, loaded[key], rng)
        pooled[key] = c
        ci = f"[{c['lo']:+.4f},{c['hi']:+.4f}]"
        print(f"{label:<38}{c['n']:>4}{c['base']:>12.4f}{c['other']:>10.4f}{c['delta']:>+9.4f}"
              f"{ci:>21}{c['eff']:>+8.3f}{c['p']:>10.2e}")

    print(f"\n[2] EQUIVALENCE TEST (margin = +-{EQUIVALENCE_MARGIN} Dice)")
    print("    'equivalent' means the entire 95% CI on the paired difference lies inside the margin,")
    print("    i.e. we can rule out a meaningful difference -- not merely fail to detect one.")
    for key, label in VARIANTS:
        if key not in pooled:
            continue
        c = pooled[key]
        inside = abs(c["lo"]) < EQUIVALENCE_MARGIN and abs(c["hi"]) < EQUIVALENCE_MARGIN
        if inside:
            verdict = "EQUIVALENT to PubMedBERT"
        elif c["lo"] > 0:
            verdict = "BETTER than PubMedBERT"
        elif c["hi"] < 0:
            verdict = "WORSE than PubMedBERT"
        else:
            verdict = "INCONCLUSIVE (CI spans the margin)"
        print(f"    {label:<38} CI [{c['lo']:+.4f},{c['hi']:+.4f}] -> {verdict}")

    print("\n[3] PER REGION x SIZE BIN")
    print(f"{'text condition':<32}{'reg':<5}{'bin':<8}{'n':>4}{'PubMedBERT':>12}{'variant':>10}"
          f"{'delta':>9}{'effect':>8}{'p_raw':>10}{'q_BH':>9}")
    detail = []
    for key, label in VARIANTS:
        if key not in loaded:
            continue
        for region in REGIONS:
            for b_ in BINS:
                c = compare(baseline, loaded[key], rng, region=region, size_bin=b_)
                if c is None:
                    continue
                c.update({"label": label, "region": region, "bin": b_})
                detail.append(c)
    if detail:
        qs = stats.false_discovery_control([d["p"] for d in detail], method="bh")
        for d, q in zip(detail, qs):
            d["q"] = q
            print(f"{d['label']:<32}{d['region']:<5}{d['bin']:<8}{d['n']:>4}{d['base']:>12.4f}"
                  f"{d['other']:>10.4f}{d['delta']:>+9.4f}{d['eff']:>+8.3f}{d['p']:>10.2e}{q:>9.4f}")
        n_sig = sum(d["q"] < 0.05 for d in detail)
        print(f"\n    {n_sig}/{len(detail)} region x bin comparisons differ from PubMedBERT after BH correction.")

    print("\n[4] DECOMPOSITION: semantics vs. geometry")
    if "randvec_aniso" in loaded:
        c = pooled["randvec_aniso"]
        print("    SEMANTICS  (PubMedBERT vs random vectors at PubMedBERT's own geometry):")
        print(f"      delta = {c['delta']:+.4f} Dice, CI [{c['lo']:+.4f},{c['hi']:+.4f}], p={c['p']:.2e}")
        print("      Geometry is held fixed, so this isolates the contribution of meaning alone.")
    if "randvec_ortho" in loaded and "randvec_aniso" in loaded:
        c = compare(loaded["randvec_aniso"], loaded["randvec_ortho"], rng)
        print("    GEOMETRY   (orthonormal vs PubMedBERT-anisotropic, both semantically empty):")
        print(f"      delta = {c['delta']:+.4f} Dice, CI [{c['lo']:+.4f},{c['hi']:+.4f}], p={c['p']:.2e}")
        print("      Semantic content is held fixed at zero, so this isolates embedding geometry alone.")

    print("\n[5] SIZE COLLAPSE UNDER EACH TEXT CONDITION (does the core RQ1 finding depend on the encoder?)")
    print(f"{'text condition':<38}{'region':<7}{'small':>9}{'medium':>9}{'large':>9}{'L/S':>8}")
    all_conditions = [("pubmedbert", "PubMedBERT (baseline)", baseline)] + [
        (k, lbl, loaded[k]) for k, lbl in VARIANTS if k in loaded]
    for _, label, rows in all_conditions:
        for region in REGIONS:
            means = {}
            for b_ in BINS:
                vals = [float(r["dice"]) for r in rows if r["region"] == region and r["size_bin"] == b_]
                means[b_] = np.mean(vals) if vals else float("nan")
            ratio = means["large"] / means["small"] if means["small"] > 0 else float("nan")
            print(f"{label:<38}{region:<7}{means['small']:>9.4f}{means['medium']:>9.4f}"
                  f"{means['large']:>9.4f}{ratio:>7.1f}x")


if __name__ == "__main__":
    main()

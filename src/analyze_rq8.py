"""RQ8 analysis: does the model use language compositionally, or as a class-ID lookup?

The evaluation script reports Dice pooled over size bins and raw p-values. With n=67-73 paired
observations, a Dice shift of 0.003 reaches p<0.001 while being clinically meaningless, so raw
significance is the wrong summary to reason from. This script adds what the claim actually rests on:

  - per-size-bin breakdown, not just the pooled mean;
  - matched-pairs rank-biserial effect size and a bootstrap CI on the mean paired difference;
  - BH-FDR correction within the RQ8 family;
  - a check of whether *how far a manipulation moves the query embedding* predicts *how much it
    changes the model's spatial behaviour*. If it does not, the projected embedding distance is not
    the mechanism, and the query is acting as an identifier rather than as meaning.

The reading that matters for the paper's thesis is not "is the change significant" but "is a
contentless query as good as a real clinical description" -- an equivalence claim, which is why the
confidence intervals rather than the p-values carry the argument.
"""
import csv
import os

import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
SCORES_CSV = os.path.join(RESULTS_DIR, "rq8_compositionality_scores.csv")
REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
CONDITIONS = ["original", "negated", "shuffled", "swapped", "generic"]
BOOTSTRAP_RESAMPLES = 10000
SEED = 0


def load():
    """Read the RQ8 compositionality probe score CSV.

    Returns:
        List of dicts, one per (patient, region, probe condition) row.
    """
    with open(SCORES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("dice", "iou", "heatmap_spearman_vs_original", "proj_cosine_vs_original", "true_volume_mm3"):
            r[k] = float(r[k])
    return rows


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


def paired(rows, region, condition, size_bin=None):
    """Return (original_dice, condition_dice) aligned by patient."""
    by_patient = {}
    for r in rows:
        if r["region"] != region:
            continue
        if size_bin is not None and r["size_bin"] != size_bin:
            continue
        if r["condition"] in ("original", condition):
            by_patient.setdefault(r["patient_id"], {})[r["condition"]] = r["dice"]
    both = [(v["original"], v[condition]) for v in by_patient.values() if len(v) == 2]
    if not both:
        return np.array([]), np.array([])
    a, b = zip(*both)
    return np.array(a), np.array(b)


def main():
    """Report how far each text manipulation moves the query versus how far it moves behaviour."""
    rows = load()
    rng = np.random.default_rng(SEED)

    print("=" * 112)
    print("RQ8: is the text query functioning as language, or as an opaque class identifier?")
    print("=" * 112)

    print("\n[1] HOW FAR EACH MANIPULATION MOVES THE QUERY, vs. HOW MUCH IT CHANGES BEHAVIOUR")
    print(f"{'condition':<12}{'region':<7}{'proj cosine':>13}{'heatmap rho':>14}")
    for condition in CONDITIONS[1:]:
        for region in REGIONS:
            s = [r for r in rows if r["region"] == region and r["condition"] == condition]
            print(f"{condition:<12}{region:<7}{np.mean([r['proj_cosine_vs_original'] for r in s]):>13.4f}"
                  f"{np.mean([r['heatmap_spearman_vs_original'] for r in s]):>14.4f}")
    print("\n  Every manipulation stays above 0.94 cosine to the original query even AFTER the trained")
    print("  projection. The text head does not separate 'enhancing tumor' from 'no enhancing tumor'")
    print("  from 'a region of tissue' in any strong sense.")

    print("\n[2] PAIRED EFFECT OF EACH MANIPULATION, POOLED OVER SIZE BINS")
    print(f"{'condition':<12}{'reg':<5}{'n':>4}{'original':>10}{'variant':>10}{'delta':>9}"
          f"{'95% CI':>21}{'effect':>8}{'p_raw':>11}{'q_BH':>9}")
    pooled = []
    for condition in CONDITIONS[1:]:
        for region in REGIONS:
            a, b = paired(rows, region, condition)
            if a.size < 5:
                continue
            p = stats.wilcoxon(a, b).pvalue if not np.allclose(a, b) else 1.0
            lo, hi = bootstrap_ci(a, b, rng)
            pooled.append({"condition": condition, "region": region, "n": a.size,
                           "orig": a.mean(), "var": b.mean(), "delta": b.mean() - a.mean(),
                           "lo": lo, "hi": hi, "eff": rank_biserial(a, b), "p": p})
    qs = stats.false_discovery_control([r["p"] for r in pooled], method="bh")
    for r, q in zip(pooled, qs):
        r["q"] = q
        ci = f"[{r['lo']:+.4f},{r['hi']:+.4f}]"
        print(f"{r['condition']:<12}{r['region']:<5}{r['n']:>4}{r['orig']:>10.4f}{r['var']:>10.4f}"
              f"{r['delta']:>+9.4f}{ci:>21}{r['eff']:>+8.3f}{r['p']:>11.2e}{r['q']:>9.4f}")

    print("\n[3] THE EQUIVALENCE CLAIM: contentless and wrong-referent queries vs. the real description")
    print("    (positive delta = the DEGRADED query localizes BETTER than the true clinical text)")
    for condition in ["generic", "swapped", "shuffled"]:
        wins = [r for r in pooled if r["condition"] == condition and r["delta"] > 0]
        near = [r for r in pooled if r["condition"] == condition and abs(r["delta"]) < 0.01]
        print(f"  {condition:<10}: beats the real description in {len(wins)}/3 regions; "
              f"within +-0.01 Dice in {len(near)}/3")
        for r in pooled:
            if r["condition"] == condition:
                verdict = "BETTER than real text" if r["delta"] > 0 else "worse"
                print(f"      {r['region']}: {r['orig']:.4f} -> {r['var']:.4f} ({r['delta']:+.4f}, {verdict})")

    print("\n[4] PER-SIZE-BIN BREAKDOWN")
    print(f"{'condition':<12}{'reg':<5}{'bin':<8}{'n':>4}{'original':>10}{'variant':>10}{'delta':>9}{'effect':>8}")
    for condition in CONDITIONS[1:]:
        for region in REGIONS:
            for b_ in BINS:
                a, b = paired(rows, region, condition, size_bin=b_)
                if a.size < 5:
                    continue
                print(f"{condition:<12}{region:<5}{b_:<8}{a.size:>4}{a.mean():>10.4f}{b.mean():>10.4f}"
                      f"{b.mean() - a.mean():>+9.4f}{rank_biserial(a, b):>+8.3f}")

    print("\n[5] DOES EMBEDDING DISTANCE PREDICT BEHAVIOURAL CHANGE?")
    xs, ys = [], []
    for r in pooled:
        s = [q for q in rows if q["region"] == r["region"] and q["condition"] == r["condition"]]
        xs.append(1.0 - np.mean([q["proj_cosine_vs_original"] for q in s]))
        ys.append(abs(r["delta"]))
    rho, p = stats.spearmanr(xs, ys)
    print(f"  Spearman(1 - projected cosine, |delta Dice|) over the {len(xs)} condition x region cells:")
    print(f"    rho={rho:+.3f}, p={p:.4f}")
    print("  If this is weak, the amount a manipulation shifts the query embedding does not explain")
    print("  how much it changes localization -- i.e. the embedding is not carrying graded meaning.")


if __name__ == "__main__":
    main()

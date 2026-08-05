"""Authoritative statistical analysis across every comparison in the project, including RQ7.

This supersedes analyze_all_comparisons.py, which corrected across a 54-test family (RQ2/3/3b/4/5)
while make_figure_significance_heatmap.py corrected across a 96-test family (adding RQ3c's three
window sizes and RQ6). Those two scripts therefore reported different q-values for the same
comparison, and the report's quoted numbers came from the figure. This script defines one family
construction for everything and is the single source of truth.

Three additions beyond the earlier scripts:

1. Effect sizes. The report has been quoting p-values and raw mean differences but no standardized
   effect size. We compute the matched-pairs rank-biserial correlation (Kerby's simple difference
   formula), the natural effect size for a Wilcoxon signed-rank test, alongside a bootstrap
   confidence interval on the mean paired difference. With n=20-32 per bin, a significant p-value
   with a CI spanning nearly zero is a materially different claim from one with a tight CI, and the
   report currently cannot distinguish them.

2. An explicit choice about the correction family. Pooling every test ever run into one BH family
   is maximally conservative, and it has the awkward property that adding a new experiment
   retroactively weakens previously published claims. The standard alternative is to correct within
   pre-registered families, one per research question. Neither is wrong, but the choice must be
   stated rather than left implicit, so both are computed and reported side by side.

3. RQ7's text-encoder ablations are included in the family, so their q-values are computed under
   the same rule as everything else.
"""
import argparse
import csv
import os

import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 0

# (display name, results filename, research-question family). Baseline for every comparison is RQ1.
COMPARISONS = [
    ("RQ2 (size-conditioned text)", "rq2_localization_scores.csv", "RQ2"),
    ("RQ3 (multi-scale ensemble)", "rq3_localization_scores.csv", "RQ3"),
    ("RQ3b (16^3 window)", "rq3b_scale16only_scores.csv", "RQ3"),
    ("RQ3c (12^3 window)", "rq3c_window12_scores.csv", "RQ3"),
    ("RQ3c (8^3 window)", "rq3c_window8_scores.csv", "RQ3"),
    ("RQ3c (6^3 window)", "rq3c_window6_scores.csv", "RQ3"),
    ("RQ4 (scale-matched)", "rq4_localization_scores.csv", "RQ4"),
    ("RQ6 (uniformity fix)", "rq6_localization_scores.csv", "RQ6"),
    ("RQ5 (naturalistic text)", "rq5_localization_scores.csv", "RQ5"),
    ("RQ7 (BERT-base)", "rq7_bertbase_localization_scores.csv", "RQ7"),
    ("RQ7 (random-init BERT)", "rq7_randbert_localization_scores.csv", "RQ7"),
    ("RQ7 (random orthonormal)", "rq7_randvec_ortho_localization_scores.csv", "RQ7"),
    ("RQ7 (random, PubMedBERT geometry)", "rq7_randvec_aniso_localization_scores.csv", "RQ7"),
    # RQ8's probe conditions live in one CSV distinguished by a `condition` column, so each needs a
    # row filter. They belong in this family: Section 6 promises every significance claim anywhere in
    # the report is corrected across the full accumulated set, and RQ8 makes claims of exactly that kind.
    ("RQ8 (negated text)", "rq8_compositionality_scores.csv", "RQ8", {"condition": "negated"}),
    ("RQ8 (shuffled word order)", "rq8_compositionality_scores.csv", "RQ8", {"condition": "shuffled"}),
    ("RQ8 (swapped referent)", "rq8_compositionality_scores.csv", "RQ8", {"condition": "swapped"}),
    ("RQ8 (generic filler)", "rq8_compositionality_scores.csv", "RQ8", {"condition": "generic"}),
    # RQ12 re-scores the winning window under better binarizers; its Otsu arm duplicates RQ3c (8^3)
    # and is excluded, but the pct99 and oracle arms are new claims and are corrected here.
    ("RQ12 (8^3 window, pct99)", "rq12_grounding_window8_scores.csv", "RQ12", {"threshold_method": "pct99"}),
    ("RQ12 (8^3 window, oracle)", "rq12_grounding_window8_scores.csv", "RQ12", {"threshold_method": "oracle_volume"}),
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


def rank_biserial(baseline_vals, other_vals):
    """Matched-pairs rank-biserial correlation (Kerby 2014): the signed proportion of rank mass
    favouring `other`. +1 means every pair improved, -1 means every pair worsened, 0 is a wash.
    Reported because a p-value alone says nothing about magnitude at these sample sizes."""
    diffs = np.asarray(other_vals) - np.asarray(baseline_vals)
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    positive = ranks[diffs > 0].sum()
    negative = ranks[diffs < 0].sum()
    total = positive + negative
    return float((positive - negative) / total) if total > 0 else 0.0


def bootstrap_ci(baseline_vals, other_vals, rng, alpha=0.05):
    """Percentile bootstrap CI on the mean paired difference, resampling patients (not observations)."""
    diffs = np.asarray(other_vals) - np.asarray(baseline_vals)
    n = diffs.size
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_tests(baseline, other, name, family, rng):
    """Run paired Wilcoxon tests of one comparison arm against the RQ1 baseline.

    Args:
        baseline: RQ1 baseline rows.
        other: the comparison arm's rows, over the same held-out patients.
        name: label used in the printed output.
        family: research-question tag, used for the per-RQ correction variant.
        rng: seeded numpy Generator for the bootstrap intervals.

    Returns:
        List of per-comparison result dicts including raw p, effect size and CI.
    """
    lookup = {(r["patient_id"], r["region"]): float(r["dice"]) for r in other}
    results = []
    for region in REGIONS:
        for b in BINS:
            pairs = [
                (float(r["dice"]), lookup[(r["patient_id"], r["region"])])
                for r in baseline
                if r["region"] == region and r["size_bin"] == b and (r["patient_id"], r["region"]) in lookup
            ]
            if len(pairs) < 5:
                continue
            base_vals, other_vals = (np.array(v) for v in zip(*pairs))
            if np.allclose(base_vals, other_vals):
                continue  # identical scores: Wilcoxon is undefined and the comparison is vacuous
            p = stats.wilcoxon(base_vals, other_vals).pvalue
            lo, hi = bootstrap_ci(base_vals, other_vals, rng)
            results.append({
                "comparison": name, "family": family, "region": region, "bin": b, "n": len(pairs),
                "base_mean": base_vals.mean(), "other_mean": other_vals.mean(),
                "delta": other_vals.mean() - base_vals.mean(),
                "ci_lo": lo, "ci_hi": hi,
                "effect_size": rank_biserial(base_vals, other_vals),
                "p_raw": p,
            })
    return results


def main():
    """Run the whole accumulated family of tests under both pooled and per-RQ BH correction.

    Reports both because the choice matters and should be visible: pooling every test into one family
    is more conservative, but has the awkward property that adding a new research question can
    retroactively demote a previously-significant result. Any such demotion is listed explicitly.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_out", default=os.path.join(RESULTS_DIR, "full_family_statistics.csv"))
    args = parser.parse_args()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    baseline = load("rq1_localization_scores.csv")
    if baseline is None:
        raise SystemExit("missing results/rq1_localization_scores.csv")

    all_results = []
    missing = []
    for entry in COMPARISONS:
        name, filename, family = entry[0], entry[1], entry[2]
        row_filter = entry[3] if len(entry) > 3 else None
        other = load(filename)
        if other is None:
            missing.append((name, filename))
            continue
        if row_filter:
            # Some result files hold several arms distinguished by a column (RQ8's probe `condition`,
            # RQ12's `threshold_method`), so select this arm's rows before pairing.
            other = [r for r in other if all(r.get(k) == v for k, v in row_filter.items())]
            if not other:
                missing.append((name, f"{filename} [{row_filter}]"))
                continue
        all_results += paired_tests(baseline, other, name, family, rng)

    if missing:
        print("Skipped (results file not present yet):")
        for name, filename in missing:
            print(f"    {name:<36} {filename}")
        print()

    if not all_results:
        raise SystemExit("no comparisons available")

    # Family construction A: one pooled family across every test (conservative; what the report did).
    pooled_q = stats.false_discovery_control([r["p_raw"] for r in all_results], method="bh")
    for r, q in zip(all_results, pooled_q):
        r["q_pooled"] = q

    # Family construction B: one family per research question (standard pre-registered practice).
    by_family = {}
    for r in all_results:
        by_family.setdefault(r["family"], []).append(r)
    for group in by_family.values():
        qs = stats.false_discovery_control([r["p_raw"] for r in group], method="bh")
        for r, q in zip(group, qs):
            r["q_perrq"] = q

    print("=" * 132)
    print(f"FULL FAMILY: {len(all_results)} paired Wilcoxon tests across {len(by_family)} research questions")
    print("Baseline for every comparison is RQ1 (results/rq1_localization_scores.csv)")
    print("=" * 132)
    header = (f"{'comparison':<36}{'reg':<4}{'bin':<8}{'n':>4}{'base':>8}{'other':>8}{'delta':>9}"
              f"{'95% CI':>20}{'effect':>8}{'p_raw':>10}{'q_pool':>9}{'q_rq':>9}  sig")
    print(header)
    print("-" * 132)
    for r in all_results:
        sig = ""
        if r["q_pooled"] < 0.05:
            sig = "**pooled+perRQ" if r["q_perrq"] < 0.05 else "**pooled"
        elif r["q_perrq"] < 0.05:
            sig = "*perRQ only"
        elif r["p_raw"] < 0.05:
            sig = "raw only"
        ci = f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
        print(f"{r['comparison']:<36}{r['region']:<4}{r['bin']:<8}{r['n']:>4}{r['base_mean']:>8.4f}"
              f"{r['other_mean']:>8.4f}{r['delta']:>+9.4f}{ci:>20}{r['effect_size']:>+8.3f}"
              f"{r['p_raw']:>10.2e}{r['q_pooled']:>9.4f}{r['q_perrq']:>9.4f}  {sig}")

    n_raw = sum(r["p_raw"] < 0.05 for r in all_results)
    n_pooled = sum(r["q_pooled"] < 0.05 for r in all_results)
    n_perrq = sum(r["q_perrq"] < 0.05 for r in all_results)
    print("-" * 132)
    print(f"Significant at raw p<0.05:                      {n_raw}/{len(all_results)}")
    print(f"Significant after BH, one pooled family:        {n_pooled}/{len(all_results)}")
    print(f"Significant after BH, per-research-question:    {n_perrq}/{len(all_results)}")
    print("\nThe pooled family is the more conservative choice and is what the report currently uses.")
    print("Note its awkward property: adding RQ7 enlarges the family and can retroactively demote a")
    print("previously-significant RQ3b/RQ3c result. Any such change is listed below, explicitly.")

    print("\n=== Tests significant at raw p<0.05 but NOT after pooled BH correction ===")
    demoted = [r for r in all_results if r["p_raw"] < 0.05 and r["q_pooled"] >= 0.05]
    if demoted:
        for r in demoted:
            print(f"  {r['comparison']:<36} {r['region']:<4} {r['bin']:<8} p={r['p_raw']:.4f} q_pooled={r['q_pooled']:.4f}")
    else:
        print("  (none)")

    print("\n=== Statistically significant but negligible in magnitude (|delta| < 0.01 Dice) ===")
    print("(real but clinically meaningless effects -- worth separating from substantive ones)")
    trivial = [r for r in all_results if r["q_pooled"] < 0.05 and abs(r["delta"]) < 0.01]
    if trivial:
        for r in trivial:
            print(f"  {r['comparison']:<36} {r['region']:<4} {r['bin']:<8} delta={r['delta']:+.4f} q={r['q_pooled']:.4f}")
    else:
        print("  (none)")

    fieldnames = ["comparison", "family", "region", "bin", "n", "base_mean", "other_mean", "delta",
                  "ci_lo", "ci_hi", "effect_size", "p_raw", "q_pooled", "q_perrq"]
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved machine-readable statistics to {args.csv_out}")


if __name__ == "__main__":
    main()

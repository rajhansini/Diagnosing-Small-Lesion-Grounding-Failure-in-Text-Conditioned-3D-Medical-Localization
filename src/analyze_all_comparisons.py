"""Consolidated statistical validation across every paired comparison run this project:
RQ1 vs {RQ2, RQ3, RQ3b, RQ4, RQ5}. Recomputes every test from the saved CSVs (not copied from
memory) and applies Benjamini-Hochberg FDR correction across the full family of tests, since
running ~45 paired tests at uncorrected alpha=0.05 would be expected to yield spurious "significant"
results by chance alone."""
import csv
import os

import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]


def load(filename):
    with open(os.path.join(RESULTS_DIR, filename), newline="") as f:
        return list(csv.DictReader(f))


def paired_tests(rq1, other, other_name):
    lookup = {(r["patient_id"], r["region"]): float(r["dice"]) for r in other}
    results = []
    for region in REGIONS:
        for b in BINS:
            pairs = [
                (float(r["dice"]), lookup[(r["patient_id"], r["region"])])
                for r in rq1 if r["region"] == region and r["size_bin"] == b and (r["patient_id"], r["region"]) in lookup
            ]
            if len(pairs) < 5:
                continue
            rq1v, otherv = zip(*pairs)
            _, p = stats.wilcoxon(rq1v, otherv)
            direction = "better" if np.mean(otherv) > np.mean(rq1v) else "worse/equal"
            results.append({
                "comparison": f"RQ1 vs {other_name}", "region": region, "bin": b, "n": len(pairs),
                "rq1_mean": np.mean(rq1v), "other_mean": np.mean(otherv), "direction": direction, "p_raw": p,
            })
    return results


def main():
    rq1 = load("rq1_localization_scores.csv")
    rq2 = load("rq2_localization_scores.csv")
    rq3 = load("rq3_localization_scores.csv")
    rq3b = load("rq3b_scale16only_scores.csv")
    rq4 = load("rq4_localization_scores.csv")
    rq5 = load("rq5_localization_scores.csv")

    all_results = []
    all_results += paired_tests(rq1, rq2, "RQ2")
    all_results += paired_tests(rq1, rq3, "RQ3")
    all_results += paired_tests(rq1, rq3b, "RQ3b")
    all_results += paired_tests(rq1, rq4, "RQ4")
    all_results += paired_tests(rq1, rq5, "RQ5")

    # also the comparison the report claimed but never actually tested
    all_results += paired_tests(rq2, rq4, "RQ4_vs_RQ2")
    for r in all_results:
        if r["comparison"] == "RQ1 vs RQ4_vs_RQ2":
            r["comparison"] = "RQ2 vs RQ4"

    p_values = [r["p_raw"] for r in all_results]
    rejected_or_qvals = stats.false_discovery_control(p_values, method="bh")
    for r, q in zip(all_results, rejected_or_qvals):
        r["q_bh"] = q

    print(f"Total tests: {len(all_results)}\n")
    print(f"{'comparison':<12} {'region':<4} {'bin':<7} {'n':<4} {'rq1/base':<10} {'other':<10} {'p_raw':<10} {'q_BH':<10} sig(raw) sig(BH)")
    n_sig_raw = 0
    n_sig_bh = 0
    for r in all_results:
        sig_raw = "*" if r["p_raw"] < 0.05 else ""
        sig_bh = "*" if r["q_bh"] < 0.05 else ""
        n_sig_raw += bool(sig_raw)
        n_sig_bh += bool(sig_bh)
        print(f"{r['comparison']:<12} {r['region']:<4} {r['bin']:<7} {r['n']:<4} {r['rq1_mean']:<10.4f} {r['other_mean']:<10.4f} {r['p_raw']:<10.4f} {r['q_bh']:<10.4f} {sig_raw:<8} {sig_bh}")

    print(f"\nSignificant at raw p<0.05: {n_sig_raw}/{len(all_results)}")
    print(f"Significant at BH-corrected q<0.05: {n_sig_bh}/{len(all_results)}")

    print("\n=== Tests that were significant raw but DID NOT survive BH correction (likely false positives) ===")
    for r in all_results:
        if r["p_raw"] < 0.05 and r["q_bh"] >= 0.05:
            print(f"  {r['comparison']} / {r['region']} / {r['bin']}: p_raw={r['p_raw']:.4f}, q_BH={r['q_bh']:.4f}")


if __name__ == "__main__":
    main()

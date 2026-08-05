"""RQ13 analysis: do the Section 7 rankings of the retrained ablations survive a calibrated threshold?

Sections 7.1, 7.5 and 7.6 judged RQ2, RQ4 and RQ6 against the RQ1 baseline under Otsu alone. Section
7.11 then showed Otsu is not a neutral instrument -- at a fixed window size it and every calibrated
rule disagree in *sign* about which of two heatmaps is better. That makes the retrained arms' verdicts
suspect for a specific, mechanical reason: each builds its heatmap by a voxel-wise MAX over three
size-phrasing queries, and a max over several maps changes the intensity distribution Otsu keys on.
RQ2 maxes three same-scale maps; RQ4 and RQ6 max three *different-scale* maps. Neither resembles the
baseline's single-query heatmap in histogram shape.

This script re-runs each arm-vs-baseline comparison under all five binarization rules and reports
whether the published direction holds. It also checks the Otsu column against the arm's original
`rq{N}_localization_scores.csv` as an end-to-end correctness gate on the re-scoring.
"""
import csv
import os

import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
REGIONS = ["ET", "TC", "WT"]
METHODS = ["otsu", "pct90", "pct95", "pct99", "oracle_volume"]
ARMS = [("rq2", "RQ2 (size-conditioned text)"),
        ("rq4", "RQ4 (scale-matched)"),
        ("rq6", "RQ6 (uniformity fix)")]


def load(filename):
    """Read a results CSV into dict rows, or None if absent."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def dice_map(rows, method=None):
    """Index Dice by (patient_id, region), optionally filtering to one binarization rule."""
    return {(r["patient_id"], r["region"]): float(r["dice"]) for r in rows
            if method is None or r["threshold_method"] == method}


def paired(base_map, arm_map, region=None):
    """Paired Wilcoxon of one arm against the baseline over their shared patients.

    Args:
        base_map: (patient, region) -> Dice for the RQ1 baseline.
        arm_map: same for the ablation arm.
        region: optional ET/TC/WT filter.

    Returns:
        dict with n, both means, delta and p, or None if fewer than 5 shared pairs.
    """
    keys = [k for k in base_map if k in arm_map and (region is None or k[1] == region)]
    if len(keys) < 5:
        return None
    a = np.array([base_map[k] for k in keys])
    b = np.array([arm_map[k] for k in keys])
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return {"n": len(keys), "base": a.mean(), "arm": b.mean(), "delta": b.mean() - a.mean(), "p": p}


def main():
    """Report each retrained arm vs. the baseline under every binarizer, plus the Otsu gate."""
    base_rows = load("rq1_localization_scores.csv")
    base_otsu = dice_map(base_rows)

    print("=" * 100)
    print("RQ13: do the retrained ablations' Section 7 verdicts survive a calibrated threshold?")
    print("=" * 100)

    print("\n[A] CORRECTNESS GATE: does the re-scored Otsu column reproduce the published CSV?")
    print("    Pass criterion is the pooled mean (what Section 7 reports); bit-exact % is a diagnostic.")
    print("    Otsu picks its cut from a 256-bin histogram, so a tie at the cut point can move a few")
    print("    voxels under GPU float non-determinism; these arms max over three heatmaps, which")
    print("    gives more opportunities for such a tie than the single-query baseline had.")
    print(f"    {'arm':<32}{'n':>5}{'bit-exact':>11}{'max|Δ|':>10}{'pooled Δ':>11}  verdict")
    for arm, label in ARMS:
        new = load(f"rq13_{arm}_thresholds_scores.csv")
        old = load(f"{arm}_localization_scores.csv")
        if new is None:
            print(f"    {label:<32} (not evaluated yet)")
            continue
        n_map, o_map = dice_map(new, "otsu"), dice_map(old)
        shared = sorted(set(n_map) & set(o_map))
        diffs = np.array([abs(n_map[k] - o_map[k]) for k in shared])
        exact = float((diffs < 1e-9).mean())
        pooled = abs(np.mean([n_map[k] for k in shared]) - np.mean([o_map[k] for k in shared]))
        # The pooled mean is the pass criterion, because that is the quantity Section 7 actually
        # reports and compares. The bit-exact fraction is shown as a diagnostic, not a gate: it
        # varies by arm purely with how often Otsu's histogram lands on a tie, which depends on how
        # many heatmaps that arm maxes together, so thresholding on it would penalise the ensemble
        # arms for a property of the binarizer rather than for any reproduction error.
        print(f"    {label:<32}{len(shared):>5}{exact:>10.1%}{diffs.max():>10.2e}{pooled:>11.2e}"
              f"  {'PASS' if pooled < 1e-3 else 'FAIL'}")

    print("\n[B] ARM vs. RQ1 BASELINE UNDER EACH BINARIZER (pooled over regions and bins)")
    print(f"  {'arm':<32}{'binarizer':<15}{'base':>8}{'arm':>8}{'delta':>9}{'p':>11}  verdict")
    published = {}
    for arm, label in ARMS:
        new = load(f"rq13_{arm}_thresholds_scores.csv")
        if new is None:
            continue
        for m in METHODS:
            r = paired(base_otsu if m == "otsu" else dice_map(load("rq12_grounding_window32_stride16_scores.csv"), m),
                       dice_map(new, m))
            if not r:
                continue
            v = "better" if r["delta"] > 0.002 else ("worse" if r["delta"] < -0.002 else "~equal")
            if m == "otsu":
                published[arm] = v
            flag = ""
            if m != "otsu" and arm in published and v != published[arm]:
                flag = "   <-- DISAGREES with the Otsu verdict"
            print(f"  {label:<32}{m:<15}{r['base']:>8.4f}{r['arm']:>8.4f}"
                  f"{r['delta']:>+9.4f}{r['p']:>11.2e}  {v}{flag}")
        print()

    print("[C] PER REGION, calibrated rule (top 1%) vs. the published Otsu verdict")
    print(f"  {'arm':<32}{'region':<7}{'otsu Δ':>10}{'pct99 Δ':>10}   same sign?")
    base32 = load("rq12_grounding_window32_stride16_scores.csv")
    for arm, label in ARMS:
        new = load(f"rq13_{arm}_thresholds_scores.csv")
        if new is None:
            continue
        for region in REGIONS:
            r_o = paired(base_otsu, dice_map(new, "otsu"), region=region)
            r_p = paired(dice_map(base32, "pct99"), dice_map(new, "pct99"), region=region)
            if not (r_o and r_p):
                continue
            same = "yes" if np.sign(r_o["delta"]) == np.sign(r_p["delta"]) else "NO"
            print(f"  {label:<32}{region:<7}{r_o['delta']:>+10.4f}{r_p['delta']:>+10.4f}   {same}")
        print()


if __name__ == "__main__":
    main()

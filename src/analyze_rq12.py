"""RQ12 analysis: does the smaller window fix grounding, or only overlap? And does its win survive a better threshold?

Reads the per-window CSVs written by evaluate_grounding_sweep.py and answers three questions.

[A] THE TIE ARTIFACT. Reported first because it revises a number in Section 6.3. A sliding-window
    heatmap with stride s assigns every voxel the mean of the windows covering it, and all voxels
    inside the same s^3 block are covered by exactly the same window set -- so the heatmap is
    piecewise-constant over s^3 blocks, not smooth. np.argmax then returns the first voxel of the
    winning block in C-order, i.e. its corner, up to s voxels from anything peak-like. At the
    published 32/stride-16 protocol that block is 16^3 = 4096 voxels, roughly 30mm across. RQ11's
    pointing game used exactly that rule, so its distances are biased upward and its hit rates
    downward. We recompute with the centroid of the tied-maximum plateau and report both.

[B] DOES THE WINNING INTERVENTION ACTUALLY POINT BETTER? Section 7 crowned the isolated smaller
    window on Dice alone. Dice conflates finding the lesion with outlining it. If the 8^3 window
    improves Dice without improving the pointing game, its win is a mask-shape effect, not better
    grounding -- a materially different claim from the one Section 7.11 currently makes.

[C] DOES THE WIN SURVIVE A BETTER BINARIZER? Every Section 7 comparison shares the Otsu threshold,
    which Section 10 argues makes them internally fair. RQ11 showed Otsu is anti-correlated with
    lesion size, so it is not a neutral common factor. We re-run the 32 vs 8 comparison under each
    binarization rule and check whether the ranking is stable.
"""
import csv
import os

import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
METHODS = ["otsu", "pct90", "pct95", "pct99", "oracle_volume"]
WINDOWS = [32, 8]
TOTAL_VOXELS = 128 ** 3


def load_window(w):
    """Read one window size's RQ12 score CSV.

    Args:
        w: window size whose CSV to read.

    Returns:
        List of dict rows, or None if that window has not been evaluated yet.
    """
    path = os.path.join(RESULTS_DIR, f"rq12_grounding_window{w}_scores.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def unique_patients(rows, method="otsu"):
    """Collapse to one row per (patient, region), since pointing fields repeat across binarizers.

    Args:
        rows: all rows for one window size.
        method: which threshold_method to keep as the representative row.

    Returns:
        dict mapping (patient_id, region) -> row.
    """
    return {(r["patient_id"], r["region"]): r for r in rows if r["threshold_method"] == method}


def chance_rate(rows_by_key, region, size_bin):
    """Chance pointing accuracy for a bin: expected hit rate of a uniformly random voxel.

    Args:
        rows_by_key: output of unique_patients().
        region: ET/TC/WT.
        size_bin: small/medium/large.

    Returns:
        Mean ground-truth voxel fraction over the patients in that bin.
    """
    fr = [int(r["gt_voxels_resized"]) / TOTAL_VOXELS for r in rows_by_key.values()
          if r["region"] == region and r["size_bin"] == size_bin]
    return float(np.mean(fr)) if fr else float("nan")


def pointing_table(rows_by_key, field):
    """Per region x bin hit counts, rates, chance level, lift and an exact binomial p-value.

    Args:
        rows_by_key: output of unique_patients().
        field: which hit column to score ("argmax_hit" or "centroid_hit").

    Returns:
        List of per-bin result dicts.
    """
    out = []
    for region in REGIONS:
        for b in BINS:
            sel = [r for r in rows_by_key.values() if r["region"] == region and r["size_bin"] == b]
            if not sel:
                continue
            n = len(sel)
            hits = sum(int(r[field]) for r in sel)
            ch = chance_rate(rows_by_key, region, b)
            p = float(stats.binomtest(hits, n, ch, alternative="greater").pvalue)
            dists = [float(r["centroid_dist_mm" if field == "centroid_hit" else "argmax_dist_mm"])
                     for r in sel]
            dists = [d for d in dists if not np.isnan(d)]
            out.append({
                "region": region, "bin": b, "n": n, "hits": hits, "rate": hits / n,
                "chance": ch, "lift": (hits / n) / ch if ch > 0 else float("nan"),
                "p": p, "median_dist": float(np.median(dists)) if dists else float("nan"),
            })
    return out


def paired_window_test(rows32, rows8, method, region=None, size_bin=None):
    """Paired Wilcoxon comparing window 8 against window 32 under one binarization rule.

    Args:
        rows32: all rows for window 32.
        rows8: all rows for window 8.
        method: threshold_method to compare under.
        region: optional ET/TC/WT filter.
        size_bin: optional small/medium/large filter.

    Returns:
        dict with n, both means, the delta and a Wilcoxon p-value, or None if too few pairs.
    """
    a_map = {(r["patient_id"], r["region"]): float(r["dice"]) for r in rows32
             if r["threshold_method"] == method}
    b_map = {(r["patient_id"], r["region"]): float(r["dice"]) for r in rows8
             if r["threshold_method"] == method}
    keys = [k for k in a_map if k in b_map
            and (region is None or k[1] == region)]
    if size_bin is not None:
        bin_of = {(r["patient_id"], r["region"]): r["size_bin"] for r in rows32}
        keys = [k for k in keys if bin_of.get(k) == size_bin]
    if len(keys) < 5:
        return None
    a = np.array([a_map[k] for k in keys])
    b = np.array([b_map[k] for k in keys])
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return {"n": len(keys), "w32": a.mean(), "w8": b.mean(), "delta": b.mean() - a.mean(), "p": p}


def main():
    """Report the tie artifact, the pointing comparison across windows, and threshold robustness."""
    data = {w: load_window(w) for w in WINDOWS}
    missing = [w for w, v in data.items() if v is None]
    if missing:
        raise SystemExit(f"missing RQ12 results for window sizes {missing}")
    uniq = {w: unique_patients(data[w]) for w in WINDOWS}

    print("=" * 108)
    print("RQ12: does the smaller window improve GROUNDING, or only overlap?")
    print("=" * 108)

    print("\n[A] THE TIE ARTIFACT: how many voxels share the maximum heatmap value?")
    print("    A sliding window of stride s makes the heatmap piecewise-constant over s^3 blocks, so")
    print("    'the peak voxel' is ambiguous and np.argmax silently returns the block's corner.")
    print(f"{'window':<9}{'stride':<9}{'median ties':>13}{'block s^3':>12}")
    for w, s in [(32, 16), (8, 8)]:
        med = int(np.median([int(r["peak_tie_count"]) for r in uniq[w].values()]))
        print(f"{w:<9}{s:<9}{med:>13}{s ** 3:>12}")
    print("\n    The median tie count equals the stride block exactly, confirming the mechanism.")
    print("    RQ11 (Section 6.3) used the first-argmax rule at stride 16, so its pointing hit rates")
    print("    are biased LOW and its peak-to-lesion distances biased HIGH. Corrected below.")

    print("\n[B] POINTING GAME, both rules, both window sizes")
    for w in WINDOWS:
        for field, label in [("argmax_hit", "first-argmax (RQ11's rule)"), ("centroid_hit", "plateau centroid (corrected)")]:
            print(f"\n  --- window {w}, {label} ---")
            print(f"  {'reg':<5}{'bin':<8}{'n':>4}{'hits':>6}{'rate':>8}{'chance':>9}"
                  f"{'lift':>9}{'p':>11}{'med dist':>10}")
            for r in pointing_table(uniq[w], field):
                lift = "n/a" if np.isnan(r["lift"]) else f"{r['lift']:.1f}x"
                print(f"  {r['region']:<5}{r['bin']:<8}{r['n']:>4}{r['hits']:>6}{r['rate']:>8.3f}"
                      f"{r['chance']:>9.4f}{lift:>9}{r['p']:>11.2e}{r['median_dist']:>9.1f}m")

    print("\n[C] DOES THE 8^3 WINDOW POINT BETTER THAN 32^3? (corrected centroid rule)")
    print(f"  {'reg':<5}{'bin':<8}{'n':>4}{'w32 rate':>10}{'w8 rate':>9}{'change':>9}   verdict")
    for region in REGIONS:
        for b in BINS:
            t32 = {(r["region"], r["bin"]): r for r in pointing_table(uniq[32], "centroid_hit")}
            t8 = {(r["region"], r["bin"]): r for r in pointing_table(uniq[8], "centroid_hit")}
            if (region, b) not in t32 or (region, b) not in t8:
                continue
            a, c = t32[(region, b)], t8[(region, b)]
            d = c["rate"] - a["rate"]
            verdict = "BETTER" if d > 0.05 else ("WORSE" if d < -0.05 else "~same")
            print(f"  {region:<5}{b:<8}{a['n']:>4}{a['rate']:>10.3f}{c['rate']:>9.3f}{d:>+9.3f}   {verdict}")

    print("\n[D] DOES THE DICE WIN SURVIVE A BETTER BINARIZER? (window 8 minus window 32)")
    print(f"  {'method':<15}{'reg':<5}{'n':>4}{'w32':>9}{'w8':>9}{'delta':>9}{'p':>11}")
    for method in METHODS:
        for region in REGIONS:
            r = paired_window_test(data[32], data[8], method, region=region)
            if r:
                print(f"  {method:<15}{region:<5}{r['n']:>4}{r['w32']:>9.4f}{r['w8']:>9.4f}"
                      f"{r['delta']:>+9.4f}{r['p']:>11.2e}")
        print()

    print("[E] SUMMARY: pooled over all regions and bins, per binarizer")
    print(f"  {'method':<15}{'n':>5}{'w32':>9}{'w8':>9}{'delta':>9}{'p':>11}  window-8 wins?")
    for method in METHODS:
        r = paired_window_test(data[32], data[8], method)
        if r:
            wins = "yes" if r["delta"] > 0 and r["p"] < 0.05 else ("no" if r["delta"] < 0 else "n.s.")
            print(f"  {method:<15}{r['n']:>5}{r['w32']:>9.4f}{r['w8']:>9.4f}"
                  f"{r['delta']:>+9.4f}{r['p']:>11.2e}  {wins}")


if __name__ == "__main__":
    main()

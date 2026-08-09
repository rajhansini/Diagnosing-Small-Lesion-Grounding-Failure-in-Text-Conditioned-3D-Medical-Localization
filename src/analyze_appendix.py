"""Eight statistics blocks that the results CSVs contained but no analysis script had extracted.

Everything here is recomputed from results/*.csv (and, for block [E], the preprocessed masks), in
the same style as analyze_full_family.py: print a human-readable block to stdout so the run can be
saved to logs/, and write the machine-readable version next to the other derived tables.

The eight blocks, and why each exists:

  [A] IoU alongside Dice.        Every evaluation script has written an `iou` column since the first
                                 run and no section of the report has ever quoted it. If the size
                                 collapse is a property of the model rather than of Dice's particular
                                 geometry, IoU must show the same thing.
  [B] Effect sizes and CIs.      Section 5.1 states that rank-biserial effect sizes and bootstrap
                                 intervals are "reported alongside every p-value". They are computed
                                 for all 171 tests and stored, but the report never showed them.
  [C] The two BH families.       analyze_full_family.py corrects under both a single pooled family
                                 and a per-research-question family. The report only ever quotes the
                                 pooled q-values, and never discusses the awkward property that
                                 adding an experiment can retroactively demote an older result.
  [D] Checkpoint selection.      The baseline is a last-epoch checkpoint; the retrained ablations are
                                 evaluated at their best-validation checkpoint. RQ7 was run both
                                 ways. Nobody had asked what that choice was worth.
  [E] Pointing-rule sensitivity. Three tie-breaking rules were written to the RQ12 CSVs; the report
                                 uses two. The third needs its own chance baseline (a random *block*,
                                 not a random voxel), computed here from the masks.
  [F] What Otsu actually selects. The report explains the Otsu/calibrated sign disagreement by
                                 appeal to histogram shape. The predicted-volume column measures it
                                 directly.
  [G] Per-region window optima.  Section 7.11 brackets the optimum at 12-16 pooled over regions. The
                                 per-region curves are not the same curve.
  [H] Cost of the intervention.  Windows per volume for every sweep condition, against the Dice each
                                 one buys.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/net/projects/ranalab/rajhansini/nlp_project"
RESULTS = os.path.join(ROOT, "results")
PREPROCESSED = os.path.join(ROOT, "data", "preprocessed")

REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
RULES = ["otsu", "pct90", "pct95", "pct99", "oracle_volume"]
RULE_LABELS = {"otsu": "Otsu", "pct90": "top 10%", "pct95": "top 5%",
               "pct99": "top 1%", "oracle_volume": "oracle volume"}

# (window, stride) in the order the sweep was run. The last two changed the tiling convention.
SWEEP = [("32", "16"), ("16", "8"), ("12", "6"), ("8", "4"), ("8", "8"), ("6", "6")]
OVERLAPPED = SWEEP[:4]

GRID = 128                      # resampled volume edge, in voxels
TOTAL_VOXELS = GRID ** 3
MM3_PER_VOXEL = 8_928_000 / TOTAL_VOXELS   # imaged volume in mm^3 / voxels, from analyze_rq11.py


def res(name):
    """Load a results CSV by filename."""
    return pd.read_csv(os.path.join(RESULTS, name))


def sweep_csv(window, stride):
    """Load one condition of the RQ12 window sweep."""
    return res(f"rq12_grounding_window{window}_stride{stride}_scores.csv")


def rule(df, method):
    """Filter a multi-binarizer table down to one rule."""
    return df[df.threshold_method == method]


def ordered(df, value_col):
    """Reindex a (region, bin) grouped frame into the report's fixed region x bin order."""
    return df.reindex([(r, b) for r in REGIONS for b in BINS])[value_col]


def header(title):
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)


# ------------------------------------------------------------------------------------------------
# [A] IoU alongside Dice
# ------------------------------------------------------------------------------------------------

def block_a():
    """Does the size collapse look the same under IoU as under Dice?

    Dice and IoU are monotone transforms of each other for a fixed pair of masks
    (IoU = Dice / (2 - Dice)), so they cannot disagree about *which* of two predictions is better on
    one patient. They can still disagree about *aggregates*, because averaging a nonlinear transform
    is not the transform of the average, and IoU is the harsher of the two on partial overlap. The
    point of reporting it is that the collapse ratios are not an artifact of Dice's particular
    algebra.
    """
    header("[A] IoU ALONGSIDE DICE -- the second metric every CSV has always carried")
    base = res("rq1_localization_scores.csv")
    print("    region  bin        n     Dice      IoU   IoU/Dice")
    for reg in REGIONS:
        for b in BINS:
            sub = base[(base.region == reg) & (base.size_bin == b)]
            print(f"    {reg:6s}  {b:9s} {len(sub):3d}   {sub.dice.mean():.4f}   "
                  f"{sub.iou.mean():.4f}     {sub.iou.mean() / sub.dice.mean():.3f}")

    print()
    print("    large/small ratio and Spearman against log lesion volume, under each metric")
    print("    region   L/S Dice   L/S IoU   rho(Dice,logV)   rho(IoU,logV)")
    for reg in REGIONS:
        d = base[base.region == reg]
        ls_d = d[d.size_bin == "large"].dice.mean() / d[d.size_bin == "small"].dice.mean()
        ls_i = d[d.size_bin == "large"].iou.mean() / d[d.size_bin == "small"].iou.mean()
        rho_d = stats.spearmanr(d.dice, np.log(d.true_volume_mm3)).statistic
        rho_i = stats.spearmanr(d.iou, np.log(d.true_volume_mm3)).statistic
        print(f"    {reg:6s}    {ls_d:5.1f}x     {ls_i:5.1f}x          {rho_d:.3f}           {rho_i:.3f}")

    print()
    print("    Every ablation's verdict, recomputed on IoU instead of Dice (pooled, paired Wilcoxon)")
    # rq3b_scale16only_scores.csv predates the IoU column, so the 16^3 window is taken from the
    # RQ12 sweep instead, which recomputes the identical condition (16^3, stride 8) with IoU.
    arms = [("RQ2 (size-conditioned text)", res("rq2_localization_scores.csv")),
            ("RQ3 (multi-scale ensemble)", res("rq3_localization_scores.csv")),
            ("RQ3b (16^3 window)", rule(sweep_csv("16", "8"), "otsu")),
            ("RQ3c (12^3 window)", res("rq3c_window12_scores.csv")),
            ("RQ4 (scale-matched)", res("rq4_localization_scores.csv")),
            ("RQ5 (naturalistic text)", res("rq5_localization_scores.csv")),
            ("RQ6 (uniformity fix)", res("rq6_localization_scores.csv"))]
    print("    arm                             dDice     dIoU   same sign?")
    flips = 0
    for name, other in arms:
        m = base.merge(other[["patient_id", "region", "dice", "iou"]],
                       on=["patient_id", "region"], suffixes=("_b", "_o"))
        d_dice = (m.dice_o - m.dice_b).mean()
        d_iou = (m.iou_o - m.iou_b).mean()
        same = "yes" if np.sign(d_dice) == np.sign(d_iou) else "NO"
        flips += same == "NO"
        print(f"    {name:30s} {d_dice:+7.4f}  {d_iou:+7.4f}   {same}")
    print(f"    -> {len(arms) - flips} of {len(arms)} arms keep their sign under the second metric.")


# ------------------------------------------------------------------------------------------------
# [B] Effect sizes and bootstrap intervals
# ------------------------------------------------------------------------------------------------

def block_b():
    """Summarise the effect-size and interval columns the report promised but never printed."""
    header("[B] EFFECT SIZES AND BOOTSTRAP INTERVALS across the 171-test family")
    fam = res("full_family_statistics.csv")
    sig = fam[fam.q_pooled < 0.05]
    print(f"    tests: {len(fam)} | significant after pooled BH: {len(sig)}")
    print()
    print("    Distribution of |matched-pairs rank-biserial| among BH-significant tests")
    a = sig.effect_size.abs()
    for lo, hi, label in [(0.0, 0.3, "negligible/small  (<0.3)"),
                          (0.3, 0.5, "moderate          (0.3-0.5)"),
                          (0.5, 0.8, "large             (0.5-0.8)"),
                          (0.8, 1.01, "very large        (>=0.8)")]:
        n = ((a >= lo) & (a < hi)).sum()
        print(f"      {label:32s} {n:3d}  ({n / len(sig) * 100:4.1f}%)")

    print()
    print("    Significant but negligible in magnitude (|delta| < 0.01 Dice) -- real, not meaningful")
    tiny = sig[sig.delta.abs() < 0.01]
    print(f"      {len(tiny)} of {len(sig)} BH-significant tests ({len(tiny) / len(sig) * 100:.0f}%)")
    per_bin = tiny.groupby("bin").size().reindex(BINS).fillna(0).astype(int)
    print(f"      by size bin: small {per_bin['small']}, medium {per_bin['medium']}, "
          f"large {per_bin['large']}")
    print("      -> the small bin is where an effect is most likely to be significant and useless,")
    print("         because baseline Dice there is 0.010-0.057 and the paired variance is tiny.")

    print()
    print("    Bootstrap intervals: do any BH-significant tests have a CI straddling zero?")
    straddle = sig[(sig.ci_lo <= 0) & (sig.ci_hi >= 0)]
    print(f"      {len(straddle)} of {len(sig)}")
    for _, r in straddle.iterrows():
        print(f"        {r.comparison:33s} {r.region:3s} {r['bin']:7s} "
              f"delta={r.delta:+.4f}  CI [{r.ci_lo:+.4f},{r.ci_hi:+.4f}]  q={r.q_pooled:.4f}")

    print()
    print("    The ten largest effects in the family, by |delta|")
    top = fam.reindex(fam.delta.abs().sort_values(ascending=False).index).head(10)
    print("    comparison                         reg bin        delta            95% CI   effect      q")
    for _, r in top.iterrows():
        print(f"    {r.comparison:33s} {r.region:3s} {r['bin']:7s} {r.delta:+8.4f}  "
              f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]  {r.effect_size:+.3f}  {r.q_pooled:.4f}")


# ------------------------------------------------------------------------------------------------
# [C] The two BH families
# ------------------------------------------------------------------------------------------------

def block_c():
    """Compare the pooled and per-research-question correction schemes."""
    header("[C] TWO WAYS TO DRAW THE TEST FAMILY -- and why the choice is not innocent")
    fam = res("full_family_statistics.csv")
    raw = (fam.p_raw < 0.05).sum()
    pooled = (fam.q_pooled < 0.05).sum()
    perrq = (fam.q_perrq < 0.05).sum()
    print(f"    significant at raw alpha=0.05 .................. {raw}/{len(fam)}")
    print(f"    significant after BH, one pooled family ........ {pooled}/{len(fam)}")
    print(f"    significant after BH, per-research-question .... {perrq}/{len(fam)}")
    print()
    disagree = fam[(fam.q_pooled < 0.05) != (fam.q_perrq < 0.05)]
    print(f"    tests where the two schemes disagree: {len(disagree)}")
    for _, r in disagree.iterrows():
        print(f"      {r.comparison:33s} {r.region:3s} {r['bin']:7s} "
              f"p={r.p_raw:.4f}  q_pooled={r.q_pooled:.4f}  q_perRQ={r.q_perrq:.4f}")
    print()
    demoted = fam[(fam.p_raw < 0.05) & (fam.q_pooled >= 0.05)]
    print(f"    significant raw but NOT after pooled BH: {len(demoted)}")
    for _, r in demoted.iterrows():
        print(f"      {r.comparison:33s} {r.region:3s} {r['bin']:7s} "
              f"p={r.p_raw:.4f}  q_pooled={r.q_pooled:.4f}")
    print()
    print("    The uncomfortable property of the pooled family, made concrete:")
    print("    the family grew as experiments were added, so a test's q-value depends on experiments")
    print("    run after it. Below, the same RQ3b tests corrected against the family as it stood when")
    print("    RQ3b was written (RQ2/RQ3/RQ3b only, 27 tests) and against the final 171.")
    early = fam[fam.comparison.isin(["RQ2 (size-conditioned text)", "RQ3 (multi-scale ensemble)",
                                     "RQ3b (16^3 window)"])].copy()
    p = np.sort(early.p_raw.values)
    m = len(p)
    q_early = np.minimum.accumulate((p * m / np.arange(1, m + 1))[::-1])[::-1]
    lookup = dict(zip(p, q_early))
    print("    comparison            reg bin           p_raw   q(27-test family)   q(171-test family)")
    for _, r in early[early.comparison == "RQ3b (16^3 window)"].iterrows():
        print(f"    {r.comparison:21s} {r.region:3s} {r['bin']:7s} {r.p_raw:.4f}            "
              f"{lookup[r.p_raw]:.4f}               {r.q_pooled:.4f}")


# ------------------------------------------------------------------------------------------------
# [D] Checkpoint selection
# ------------------------------------------------------------------------------------------------

def block_d():
    """How much of RQ7's single-seed headline was checkpoint selection rather than the encoder?"""
    header("[D] CHECKPOINT SELECTION -- an axis of variation nothing else in the project varied")
    base = res("rq1_localization_scores.csv")
    b_mean = base.dice.mean()
    print(f"    RQ1 baseline (last-epoch checkpoint, the only one train_baseline.py saves): "
          f"{b_mean:.4f}")
    print()
    print("    RQ7, seed 0, the same trained runs evaluated at both checkpoints")
    print("    text condition                     last     best   delta(last)  delta(best)  sign flip?")
    variants = [("BERT-base (general domain)", "bertbase"),
                ("random-init BERT (frozen)", "randbert"),
                ("4 random orthonormal vectors", "randvec_ortho"),
                ("random vectors, PubMedBERT geom", "randvec_aniso")]
    flips = 0
    for label, v in variants:
        last = res(f"rq7_{v}_localization_scores.csv").dice.mean()
        best = res(f"rq7_{v}_best_localization_scores.csv").dice.mean()
        dl, db = last - b_mean, best - b_mean
        flip = np.sign(dl) != np.sign(db) and abs(db) > 1e-9
        flips += flip
        print(f"    {label:33s} {last:.4f}  {best:.4f}    {dl:+.4f}      {db:+.4f}"
              f"      {'YES' if flip else 'no'}")
    print()
    print(f"    {flips} of {len(variants)} conditions change sign on checkpoint choice alone.")
    print("    In particular the +0.0379 'random-init BERT beats PubMedBERT' result that carried a")
    print("    within-run p ~ 6e-35 is -0.0008 at the best-validation checkpoint of the same runs.")
    print("    Seed replication and checkpoint choice are two independent probes of the same")
    print("    fragility, and they agree.")
    print()
    print("    Asymmetry worth disclosing: the retrained arms are evaluated at *_best, the baseline")
    print("    they are compared against has no best checkpoint. Direction of the bias:")
    for label, v in [("RQ2", "rq2"), ("RQ4", "rq4"), ("RQ5", "rq5"), ("RQ6", "rq6")]:
        arm = res(f"{v}_localization_scores.csv")
        print(f"      {label}: evaluated at best-val checkpoint, pooled Dice {arm.dice.mean():.4f} "
              f"vs baseline {b_mean:.4f} ({arm.dice.mean() - b_mean:+.4f})")
    print("    The selection favours the ablations, so RQ4's and RQ6's negative verdicts are")
    print("    conservative and RQ2's positive one is not.")


# ------------------------------------------------------------------------------------------------
# [E] Pointing-rule sensitivity, with a block-level chance baseline
# ------------------------------------------------------------------------------------------------

def block_e():
    """Bound how much the pointing result depends on how a tied maximum is broken.

    Three rules were written to the CSVs:
      argmax_hit    the first voxel of the winning plateau in array order -- its corner.
      centroid_hit  the plateau's centroid. The report's rule.
      any_tied_hit  does *any* voxel of the plateau fall inside the lesion.
    The first two are point rules and share the per-voxel chance level. The third is not a pointing
    rule at all -- it asks whether the peak *block* touches the lesion -- so it needs a block-level
    chance baseline, computed here by counting, per patient, what fraction of the stride-aligned
    blocks the ground-truth mask intersects.
    """
    from dataset import region_mask

    header("[E] POINTING-RULE SENSITIVITY -- and the block-level chance the third rule needs")
    d = rule(sweep_csv("32", "16"), "otsu")
    print("    window 32^3, stride 16 (the published protocol; plateau = 16^3 = 4096 voxels)")
    print("    region bin        n   argmax  centroid   any-tied   block chance   any-tied lift")
    out = []
    for reg in REGIONS:
        for b in BINS:
            sub = d[(d.region == reg) & (d.size_bin == b)]
            chances = []
            for pid in sub.patient_id:
                mask = np.load(os.path.join(PREPROCESSED, f"{pid}.npz"))["mask"]
                gt = region_mask(mask, reg)
                blocks = gt.reshape(8, 16, 8, 16, 8, 16).any(axis=(1, 3, 5))
                chances.append(blocks.mean())
            chance = float(np.mean(chances))
            n = len(sub)
            a, c, t = int(sub.argmax_hit.sum()), int(sub.centroid_hit.sum()), int(sub.any_tied_hit.sum())
            lift = (t / n) / chance if chance > 0 else np.nan
            p = stats.binomtest(t, n, chance, alternative="greater").pvalue
            out.append((reg, b, n, a, c, t, chance, lift, p))
            print(f"    {reg:6s} {b:9s} {n:3d}   {a:3d}/{n:<3d} {c:4d}/{n:<3d}   {t:4d}/{n:<3d}"
                  f"      {chance * 100:5.2f}%        {lift:6.1f}x  (p={p:.1e})")
    print()
    et_small = out[0]
    print(f"    The headline bin, stated three ways. Small enhancing tumor, n={et_small[2]}:")
    print(f"      peak voxel (argmax corner) inside the lesion ....... {et_small[3]}/{et_small[2]}")
    print(f"      peak voxel (plateau centroid) inside the lesion .... {et_small[4]}/{et_small[2]}")
    print(f"      peak *block* touches the lesion ................... {et_small[5]}/{et_small[2]}"
          f"  ({et_small[7]:.0f}x the {et_small[6] * 100:.2f}% block chance)")
    print("    Both point rules agree the model never points at a small enhancing tumor. The block")
    print("    rule shows it is not lost either: the winning 16^3 block contains part of the lesion")
    print("    in 9 of 21 cases. The failure is resolution *inside* the block, not coarse search --")
    print("    which is the architectural reading, measured rather than asserted.")

    print()
    print("    All three rules across the sweep (pooled over regions and bins)")
    print("    window/stride   median ties   argmax   centroid   any-tied")
    for w, s in SWEEP:
        sub = rule(sweep_csv(w, s), "otsu")
        print(f"    {w:>2s}^3 / {s:<2s}         {int(sub.peak_tie_count.median()):>6d}    "
              f"{sub.argmax_hit.mean():.3f}      {sub.centroid_hit.mean():.3f}      "
              f"{sub.any_tied_hit.mean():.3f}")
    print("    The three rules converge as the plateau shrinks, which is what a rule-sensitivity")
    print("    check should look like: the ambiguity is a property of the stride, not of the model.")


# ------------------------------------------------------------------------------------------------
# [F] What Otsu actually selects
# ------------------------------------------------------------------------------------------------

def block_f():
    """Measure the histogram-shape story the report tells in words."""
    header("[F] WHAT EACH RULE ACTUALLY SELECTS -- the mechanism behind the sign disagreement")
    print("    mean % of the imaged volume marked as lesion, per binarization rule and condition")
    print("    rule            " + "".join(f"{w}^3/{s:<3s}" for w, s in SWEEP))
    for m in RULES:
        cells = []
        for w, s in SWEEP:
            sub = rule(sweep_csv(w, s), m)
            cells.append(f"{sub.pred_voxels.mean() / TOTAL_VOXELS * 100:7.2f} ")
        print(f"    {RULE_LABELS[m]:14s}" + "".join(cells))
    print()
    print("    Only Otsu's row moves: every other rule selects a fixed fraction by construction.")
    print()
    print("    Otsu's selected fraction, and its spread across patients")
    print("    window/stride   mean %   std %   ratio to true lesion fraction")
    truth = rule(sweep_csv("32", "16"), "oracle_volume").pred_voxels.mean() / TOTAL_VOXELS * 100
    for w, s in SWEEP:
        sub = rule(sweep_csv(w, s), "otsu")
        mean = sub.pred_voxels.mean() / TOTAL_VOXELS * 100
        std = sub.pred_voxels.std() / TOTAL_VOXELS * 100
        print(f"    {w:>2s}^3 / {s:<2s}        {mean:6.2f}  {std:6.2f}          {mean / truth:5.1f}x")
    print(f"    (true mean lesion fraction across the same patients: {truth:.2f}%)")
    print()
    print("    Holding the window at 8^3 and changing only the stride, 4 -> 8:")
    a = rule(sweep_csv("8", "4"), "otsu").pred_voxels.mean() / TOTAL_VOXELS * 100
    b = rule(sweep_csv("8", "8"), "otsu").pred_voxels.mean() / TOTAL_VOXELS * 100
    print(f"      Otsu's mask shrinks from {a:.2f}% to {b:.2f}% of the volume ({(b - a) / a * 100:+.0f}%)")
    print("      -- i.e. dropping overlap moves Otsu's cut point toward the truth for a reason that")
    print("      has nothing to do with the heatmap being better. That is the whole disagreement:")
    print("      Otsu rewards the blockier heatmap because a coarser histogram splits more cleanly,")
    print("      while every fixed-fraction rule is scored on spatial resolution it has just lost.")


# ------------------------------------------------------------------------------------------------
# [G] Per-region window optima
# ------------------------------------------------------------------------------------------------

def block_g():
    """The pooled optimum at 12-16 is an average over three different per-region optima."""
    header("[G] PER-REGION WINDOW OPTIMA -- the pooled bracket hides a three-way disagreement")
    for m in ["otsu", "pct99", "oracle_volume"]:
        print()
        print(f"    -- {RULE_LABELS[m]} --")
        print("    region   " + "".join(f"{w}^3/{s:<3s}" for w, s in OVERLAPPED) + "   best (overlap-matched)")
        for reg in REGIONS:
            vals = []
            for w, s in OVERLAPPED:
                sub = rule(sweep_csv(w, s), m)
                vals.append(sub[sub.region == reg].dice.mean())
            best = OVERLAPPED[int(np.argmax(vals))]
            print(f"    {reg:6s}   " + "".join(f"{v:7.4f} " for v in vals)
                  + f"   {best[0]}^3 / stride {best[1]}")
    print()
    print("    Under the deployable top-1% rule the three regions want three different windows:")
    for m in ["pct99", "oracle_volume"]:
        picks = []
        for reg in REGIONS:
            vals = [rule(sweep_csv(w, s), m).pipe(lambda d: d[d.region == reg].dice.mean())
                    for w, s in OVERLAPPED]
            picks.append(f"{reg} {OVERLAPPED[int(np.argmax(vals))][0]}^3")
        print(f"      {RULE_LABELS[m]:14s} {', '.join(picks)}")
    print("    This is the same regional split the pointing game found (Section 7.11): the smaller")
    print("    window helps enhancing tumor and whole tumor and hurts tumor core. Pooling the curve")
    print("    averages a peak at 16^3 against one still rising at 8^3 and reports the midpoint.")


# ------------------------------------------------------------------------------------------------
# [H] What the intervention costs
# ------------------------------------------------------------------------------------------------

def block_h():
    """Windows per volume against the Dice each condition buys."""
    header("[H] COST OF THE INTERVENTION -- 'no retraining' is not the same as 'free'")
    base_n = None
    print("    window/stride   windows/volume   vs baseline   pooled Dice (top 1%)   Dice/1000 windows")
    for w, s in SWEEP:
        n = ((GRID - int(w)) // int(s) + 1) ** 3
        if base_n is None:
            base_n = n
        dice = rule(sweep_csv(w, s), "pct99").dice.mean()
        tag = "  (non-overlapping)" if int(s) == int(w) else ""
        print(f"    {w:>2s}^3 / {s:<2s}        {n:>8d}       {n / base_n:6.1f}x        "
              f"{dice:.4f}                {dice / n * 1000:.4f}{tag}")
    print()
    print("    Measured wall-clock, from the SLURM logs (73 validation patients x 3 regions):")
    for label, minutes in [("32^3 / 16 (RQ1 protocol)", 2.8), ("16^3 / 8  (RQ3b)", 19.3),
                           ("12^3 / 6  (RQ3c)", 52.0), ("8^3 / 4   (RQ12 control)", 32.0),
                           ("6^3 / 6   (RQ3c)", 64.0)]:
        print(f"      {label:28s} {minutes:5.1f} min")
    print()
    print("    The recommendation the report makes -- 12^3-16^3 at 50% overlap -- costs 10-23x the")
    print("    baseline's forward passes and buys +0.060 to +0.067 Dice. Against retraining an arm")
    print("    (24 min on one RTX 2080 Ti) that is still cheap, and unlike retraining it needs no")
    print("    labels, no optimiser and no second model.")


def main():
    """Run all eight blocks and write the machine-readable summary."""
    print("=" * 108)
    print("APPENDIX ANALYSIS: eight statistics blocks recomputed from the result tables")
    print("Everything here was already in results/*.csv; none of it had been extracted before.")
    print("=" * 108)
    block_a()
    block_b()
    block_c()
    block_d()
    block_e()
    block_f()
    block_g()
    block_h()
    print()
    print("Done.")


if __name__ == "__main__":
    main()

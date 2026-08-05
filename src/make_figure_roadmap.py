"""Figure: experiment roadmap / decision tree summarizing how RQ1's finding led to RQ2-RQ6,
each box giving the one-line takeaway, so a reader can see the whole project's logic at a glance."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"

BLUE = "#0072B2"
GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#666666"


def box(ax, xy, w, h, title, body, edgecolor, fontsize=8.3):
    """Draw a labelled rounded box on the schematic.

    Args:
        ax: matplotlib axes to draw on.
        xy: (x, y) of the box's lower-left corner.
        w: box width.
        h: box height.
        title: bold heading drawn inside the box.
        body: smaller descriptive text below the title.
        edgecolor: border colour, encoding whether the outcome helped or hurt.
        fontsize: title size.

    Returns:
        None; draws directly on ax.
    """
    b = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                 linewidth=1.8, edgecolor=edgecolor, facecolor="#F5F5F5")
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h - 0.16, title, ha="center", va="top", fontsize=fontsize + 1, fontweight="bold")
    ax.text(xy[0] + w / 2, xy[1] + h / 2 - 0.08, body, ha="center", va="center", fontsize=fontsize, wrap=True)
    return b


def arrow(ax, xy_from, xy_to, color=GRAY, style="-|>"):
    """Draw a connecting arrow between two points on the schematic.

    Args:
        ax: matplotlib axes to draw on.
        xy_from: (x, y) tail position.
        xy_to: (x, y) head position.
        color: arrow colour.
        style: matplotlib connection style for curved edges.

    Returns:
        None; draws directly on ax.
    """
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5, shrinkA=2, shrinkB=2))


def main():
    """Draw the decision-tree diagram of every experiment and its outcome.

    Colour encodes the outcome: green confirmed or helped, red hurt, amber partial or overturned,
    blue a neutral finding. Amber is load-bearing here -- RQ3b/RQ3c looked green for most of the
    project and were reclassified once RQ12 re-scored them under a calibrated threshold.
    """
    amber = "#E69F00"
    fig, ax = plt.subplots(figsize=(13.5, 9.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    # ---- top: the finding, and the check that it is real ----
    box(ax, (0.3, 8.0), 3.9, 1.15, "P′: supervised reference",
        "Same data/split/metric, dense labels:\nreaches published BraTS Dice, and\ncollapses only 1.2-1.3x (vs 5-15x)", GREEN, fontsize=7.8)
    box(ax, (5.0, 8.0), 3.9, 1.15, "RQ1: core finding",
        "Dice collapses 5-15x large->small,\nbeyond chance, 9/9 across 3 splits", BLUE, fontsize=8)
    box(ax, (9.6, 8.0), 3.6, 1.15, "RQ11: decompose the\npipeline",
        "~half the collapse is the Otsu\nthreshold, not grounding", amber, fontsize=7.8)

    for x, label in [(1.95, "TEXT-SIDE"), (6.65, "INFERENCE-SIDE"), (11.3, "TRAINING-SIDE")]:
        ax.text(x, 7.5, label, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color=GRAY)

    # ---- left column: is the failure linguistic? ----
    box(ax, (0.3, 6.15), 3.3, 1.15, "RQ2: size-conditioned\nprompting",
        "Helps med/large, WORSENS small\nET; widens the ET size gap", RED, fontsize=7.6)
    box(ax, (0.3, 4.55), 3.3, 1.15, "RQ5: naturalistic text",
        "Size collapse unchanged; across 3\nseeds actually BETTER than templated", GREEN, fontsize=7.6)
    box(ax, (0.3, 2.95), 3.3, 1.15, "RQ7: swap the encoder",
        "BERT-base / random-init / random\nvectors: all within retraining noise", RED, fontsize=7.6)
    box(ax, (0.3, 1.35), 3.3, 1.15, "RQ8: break the query",
        "Negate, shuffle, swap, blank it:\nheatmap ρ stays 0.56-1.00", RED, fontsize=7.6)

    # ---- centre column: is it the receptive field? ----
    box(ax, (4.85, 6.15), 3.6, 1.15, "RQ3: naive multi-scale\nensemble",
        "Mostly HURTS (5/9 worse) --\nnoisiest scale dominates the max", RED, fontsize=7.6)
    box(ax, (4.85, 4.55), 3.6, 1.15, "RQ3b/RQ3c: one smaller\nwindow (16/12/8/6³)",
        "Helps in all 9 bins under Otsu --\nbut 8³/6³ also changed the tiling", amber, fontsize=7.6)
    box(ax, (4.85, 2.95), 3.6, 1.15, "RQ12: re-score under 5\nbinarizers, tiling fixed",
        "Win is REAL and 2-3x bigger than\nOtsu showed; optimum at 12-16³", GREEN, fontsize=7.6)

    # ---- right column: can it be trained away? ----
    box(ax, (9.6, 6.15), 3.5, 1.15, "RQ4: scale-matched\nretraining",
        "Better classification acc, but\nWORSE localization everywhere", RED, fontsize=7.6)
    box(ax, (9.6, 4.55), 3.5, 1.15, "Noise probe diagnosis",
        "Pure noise classified confidently --\nlearned resize artifacts, not content", RED, fontsize=7.6)
    box(ax, (9.6, 2.95), 3.5, 1.15, "RQ6: uniformity\nregularizer",
        "Fixes 2/3 of the hub bias, beats RQ4\n9/9, still short of plain RQ1", amber, fontsize=7.6)

    # ---- bottom line ----
    box(ax, (4.85, 0.9), 8.25, 1.55, "Bottom line",
        "One intervention works, and it is the simplest: query the frozen model at a 12-16³ window "
        "instead of 32³.\nWorth +0.060 Dice (top-1%) / +0.067 (oracle), and it lifts ET pointing in all "
        "three size bins, small-ET from\n0/21 to 4/21. Everything more sophisticated failed. Swapping "
        "Otsu for a fixed top-1% rule is a second free gain.\nMethodological lesson: a SHARED confound "
        "is not a cancelled confound -- Otsu and every calibrated rule\ndisagree in SIGN on the same "
        "heatmaps, so a common metric can rank arms backwards, invisibly to FDR.", BLUE, fontsize=7.2)

    arrow(ax, (4.2, 8.55), (5.0, 8.55), color=GRAY)
    arrow(ax, (8.9, 8.55), (9.6, 8.55), color=GRAY)
    arrow(ax, (6.0, 8.0), (1.95, 7.35))
    arrow(ax, (6.95, 8.0), (6.65, 7.35))
    arrow(ax, (7.9, 8.0), (11.3, 7.35))
    for x in (1.95, 6.65, 11.3):
        arrow(ax, (x, 6.15), (x, 5.75))
        arrow(ax, (x, 4.55), (x, 4.15))
    for x in (1.95, 6.65, 11.3):
        arrow(ax, (x, 2.95), (x, 2.55) if x != 1.95 else (x, 2.55))
    arrow(ax, (1.95, 1.35), (4.85, 1.7), color=GRAY)
    arrow(ax, (6.65, 2.95), (6.65, 2.55), color=GRAY)
    arrow(ax, (11.3, 2.95), (11.3, 2.55), color=GRAY)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_roadmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

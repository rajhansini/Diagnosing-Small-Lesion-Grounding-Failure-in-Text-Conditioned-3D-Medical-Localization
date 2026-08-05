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
    """Draw the decision-tree diagram of every ablation and its outcome."""
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")

    box(ax, (4.7, 7.6), 3.6, 1.1, "RQ1: core finding", "Dice collapses 5-15x from large\nto small lesions, beyond chance", BLUE, fontsize=8)

    box(ax, (0.3, 5.9), 3.2, 1.15, "RQ2: size-conditioned\ntext prompting", "Helps med/large, WORSENS\nsmall ET (opposite of goal)", RED, fontsize=7.8)
    box(ax, (4.7, 5.9), 3.6, 1.15, "RQ3: naive multi-scale\nensemble (no retrain)", "Mostly HURTS (5/9 worse) --\nnoisiest scale dominates the max", RED, fontsize=7.8)
    box(ax, (9.3, 5.9), 3.4, 1.15, "RQ5: naturalistic text\n(robustness check)", "Replicates RQ1 almost exactly --\nnot a templating artifact", GREEN, fontsize=7.8)

    box(ax, (4.7, 4.2), 3.6, 1.1, "RQ3b: isolate ONE\nsmaller window (16³)", "Helps in ALL 9 bins raw p<0.05\n(6/9 survive FDR correction)", GREEN, fontsize=7.8)

    box(ax, (4.7, 2.5), 3.6, 1.1, "RQ3c: push further\n(12/8/6³)", "Keeps improving to 8³, THEN\nplateaus for ET/TC (not WT)", GREEN, fontsize=7.8)

    box(ax, (0.3, 4.2), 3.2, 1.15, "RQ4: train with\nscale-matched patches", "Better classification acc, but\nWORSE localization everywhere", RED, fontsize=7.8)
    box(ax, (0.3, 2.5), 3.2, 1.3, "Noise probe diagnosis", "Pure noise gets classified with\nhigh confidence -- model learned\nresize artifacts, not content", RED, fontsize=7.6)
    box(ax, (0.3, 0.5), 3.2, 1.3, "RQ6: uniformity\nregularizer fix", "Fixes 2/3 of the hub bias, beats\nRQ4 in 9/9 bins, still short of\nplain RQ1 baseline overall", "#E69F00", fontsize=7.6)

    box(ax, (9.0, 2.9), 3.7, 1.5, "Bottom line", "The simplest fix (RQ3b/c: smaller\nwindow, no retraining) beats every\nmore sophisticated attempt tried.\nComplexity introduced new failure\nmodes faster than it fixed the old one.", BLUE, fontsize=7.6)

    arrow(ax, (5.0, 7.6), (1.9, 7.05))
    arrow(ax, (6.5, 7.6), (6.5, 7.05))
    arrow(ax, (8.0, 7.6), (11.0, 7.05))
    arrow(ax, (6.5, 5.9), (6.5, 5.3))
    arrow(ax, (6.5, 4.2), (6.5, 3.6))
    arrow(ax, (1.9, 5.9), (1.9, 5.3))
    arrow(ax, (1.9, 4.2), (1.9, 3.8))
    arrow(ax, (1.9, 2.5), (1.9, 1.8))
    arrow(ax, (3.5, 1.15), (9.0, 3.2), color=GRAY)
    arrow(ax, (8.3, 3.05), (9.0, 3.4), color=GRAY)

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_roadmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

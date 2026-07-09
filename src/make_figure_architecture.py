"""Figure: schematic of the text-volume contrastive alignment + sliding-window localization
pipeline, for the Method section (which otherwise has zero diagrams)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"
LIGHT = "#EDEDED"


def box(ax, xy, w, h, text, color, fontsize=9, textcolor="black"):
    b = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                 linewidth=1.4, edgecolor=color, facecolor=LIGHT)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=textcolor, wrap=True)
    return b


def arrow(ax, xy_from, xy_to, color=GRAY):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 6.5)
    ax.axis("off")

    # text branch
    box(ax, (0.3, 5.2), 2.4, 0.8, "Region description\n(text)", BLUE)
    box(ax, (0.3, 3.9), 2.4, 0.8, "PubMedBERT\n(frozen, mean-pooled)", BLUE)
    box(ax, (0.3, 2.6), 2.4, 0.8, "text_proj\n(trainable linear)", BLUE)
    arrow(ax, (1.5, 5.2), (1.5, 4.7))
    arrow(ax, (1.5, 3.9), (1.5, 3.4))

    # image branch
    box(ax, (4.3, 5.2), 2.4, 0.8, "3D volume patch\n(32³ voxels, 4 modalities)", ORANGE)
    box(ax, (4.3, 3.9), 2.4, 0.8, "MONAI 3D ResNet-10\n(trainable)", ORANGE)
    box(ax, (4.3, 2.6), 2.4, 0.8, "global average pool\n-> single embedding", ORANGE)
    arrow(ax, (5.5, 5.2), (5.5, 4.7))
    arrow(ax, (5.5, 3.9), (5.5, 3.4))

    # shared space
    box(ax, (2.3, 1.3), 4.4, 0.8, "Shared 256-d space\ncosine similarity / temperature -> contrastive loss", GRAY, fontsize=9)
    arrow(ax, (1.5, 2.6), (3.5, 2.1))
    arrow(ax, (5.5, 2.6), (5.5, 2.1))

    # localization branch (right side)
    box(ax, (7.3, 5.2), 2.4, 0.8, "Sliding window over\nfull 128³ volume", GRAY)
    box(ax, (7.3, 3.9), 2.4, 0.8, "Cosine similarity\nto query text at each position", GRAY)
    box(ax, (7.3, 2.6), 2.4, 0.8, "Per-voxel heatmap", GRAY)
    box(ax, (7.3, 1.3), 2.4, 0.8, "Otsu threshold\n-> predicted mask", GRAY)
    box(ax, (7.3, 0.0), 2.4, 0.8, "Dice / IoU\nvs. ground truth", GRAY)
    arrow(ax, (8.5, 5.2), (8.5, 4.7))
    arrow(ax, (8.5, 3.9), (8.5, 3.4))
    arrow(ax, (8.5, 2.6), (8.5, 2.1))
    arrow(ax, (8.5, 1.3), (8.5, 0.8))
    arrow(ax, (6.7, 1.7), (7.3, 4.3), color=GRAY)

    ax.text(5.0, 6.1, "Training (contrastive alignment)", ha="center", fontsize=11, fontweight="bold")
    ax.text(8.5, 6.1, "Inference (localization)", ha="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_architecture.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

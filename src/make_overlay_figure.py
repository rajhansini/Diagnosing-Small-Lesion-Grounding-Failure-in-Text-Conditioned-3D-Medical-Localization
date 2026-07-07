"""Qualitative example figure: T1ce slice with predicted heatmap overlay and ground-truth ET
contour, for a well-localized large lesion vs a poorly-localized small lesion."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"

EXAMPLES = [
    ("large_example_BraTS20_Training_225.npz", "BraTS20_Training_225 -- large ET (true vol 111,250 mm³, Dice 0.31)"),
    ("small_example_BraTS20_Training_298.npz", "BraTS20_Training_298 -- small ET (true vol 94 mm³, Dice 0.0002)"),
]


def best_slice(seg):
    et = seg == 4
    counts = et.sum(axis=(0, 1))  # count per axial (last-axis) slice
    return int(np.argmax(counts))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, (fname, title) in zip(axes, EXAMPLES):
        d = np.load(os.path.join(FIGURES_DIR, fname))
        t1ce, seg, heatmap = d["t1ce"], d["seg"], d["heatmap"]

        z = best_slice(seg)
        img_slice = t1ce[:, :, z]
        heat_slice = heatmap[:, :, z]
        gt_slice = (seg[:, :, z] == 4)

        ax.imshow(np.rot90(img_slice), cmap="gray")
        heat_norm = (heat_slice - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        ax.imshow(np.rot90(heat_norm), cmap="hot", alpha=0.45, vmin=0, vmax=1)
        ax.contour(np.rot90(gt_slice), colors="cyan", linewidths=1.2, levels=[0.5])
        ax.set_title(title, fontsize=9, pad=10)
        ax.axis("off")

    # legend proxies
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor="red", alpha=0.45, label="predicted heatmap (hot = high response)"),
        Line2D([0], [0], color="cyan", lw=1.5, label="ground-truth ET boundary"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = os.path.join(FIGURES_DIR, "fig3_example_overlays.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""
visualize_results.py — Run this after task1.py and task2.py to visualize results.
Usage:
    python visualize_results.py --task1_val result_task1_val.json --task2 result_task2.json --img_dir THE_VALIDATION_IMAGE_DIRECTORY --cluster_dir THE_FACE_CLUSTER_DIRECTORY
"""

import json, os, argparse
import face_recognition
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

# ── Task 1: drawing bounding boxes on a sample of images ──────────────────────
def viz_task1(result_json, img_dir, n_samples=50, out="viz_task1.png"):
    with open(result_json) as f:
        results = json.load(f)

    keys = list(results.keys())
    sample = random.sample(keys, min(n_samples, len(keys)))

    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, fname in zip(axes, sample):
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            ax.axis('off')
            continue
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        boxes = results[fname]
        for (x, y, w, h) in boxes:
            rect = patches.Rectangle((x, y), w, h,
                                      linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
        ax.set_title(f"{fname}\n{len(boxes)} face(s)", fontsize=9)
        ax.axis('off')

    for ax in axes[len(sample):]:
        ax.axis('off')

    plt.suptitle("Task 1 — Face Detection Results", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved → {out}")


# ── Task 2: showing clusters as image grids ───────────────────────────────────
def viz_task2(result_json, img_dir, max_per_cluster=10, out="viz_task2.png"):
    with open(result_json) as f:
        clusters = json.load(f)   # list of lists of filenames

    K = len(clusters)
    cols = max_per_cluster
    fig, axes = plt.subplots(K, cols, figsize=(cols * 1.6, K * 1.8))
    if K == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for r, cluster in enumerate(clusters):
        row_axes = axes[r] if K > 1 else axes[r]
        sample = cluster[:cols]
        for c in range(cols):
            ax = row_axes[c] if cols > 1 else row_axes
            if c < len(sample):
                img_path = os.path.join(img_dir, sample[c])
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(colors[r % 10])
                        spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
        row_axes[0].set_ylabel(f"Cluster {r}\n({len(cluster)} imgs)",
                               fontsize=8, rotation=0, labelpad=50, va='center')

    plt.suptitle("Task 2 — Face Clustering Results", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1_val", default="result_task1_val.json")
    parser.add_argument("--task2",     default="result_task2.json")
    parser.add_argument("--img_dir",   default="val/",            help="val images folder")
    parser.add_argument("--cluster_dir", default="faceCluster_K/", help="clustering images folder")
    parser.add_argument("--samples",   type=int, default=6,        help="images to sample for task1")
    args = parser.parse_args()

    if os.path.exists(args.task1_val) and os.path.exists(args.img_dir):
        viz_task1(args.task1_val, args.img_dir, n_samples=args.samples)
    else:
        print(f"Skipping task1 viz — {args.task1_val} or {args.img_dir} not found")

    if os.path.exists(args.task2) and os.path.exists(args.cluster_dir):
        viz_task2(args.task2, args.cluster_dir)
    else:
        print(f"Skipping task2 viz — {args.task2} or {args.cluster_dir} not found")
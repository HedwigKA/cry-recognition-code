"""Figures for the paper from stage1_results.json and stage2_results.json."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def cm_plot(ax, cm, labels, title):
    cm = np.array(cm, float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)


def main():
    s1 = json.load(open("stage1_results.json"))
    s2 = json.load(open("stage2_results.json"))
    w = s1["window_3class"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cm_plot(ax, w["confusion"], w["labels"], "Stage 1, window level (5-fold)")
    fig.tight_layout(); fig.savefig("fig_stage1_confusion.png", dpi=200)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, k in zip(axs, ["five_weighted", "five_unweighted"]):
        cm_plot(ax, s2["confusions"][k], s2["labels"]["five"],
                f"Stage 2, {k.replace('five_', '')} (3 seeds x 5 folds)")
    fig.tight_layout(); fig.savefig("fig_stage2_confusion.png", dpi=200)

    r = s2["results"]
    schemes = ["five", "three", "binary"]
    variants = [("always_hungry", "Always hungry"), ("baseline_knn", "Energy+spectrum k-NN"),
                ("unweighted", "1D CNN"), ("weighted", "1D CNN + class weight")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(schemes)); bw = 0.2
    for i, (v, name) in enumerate(variants):
        m = [r[f"{s}_{v}"]["macro_f1_mean"] for s in schemes]
        sd = [r[f"{s}_{v}"].get("macro_f1_sd", 0) for s in schemes]
        ax.bar(x + (i - 1.5) * bw, m, bw, yerr=sd, capsize=3, label=name)
    ax.set_xticks(x, ["5 reasons", "3 groups", "hungry vs other"])
    ax.set_ylabel("Macro-F1 (clip level)"); ax.set_ylim(0, 1); ax.legend(fontsize=8, ncol=2)
    ax.set_title("Stage 2: cry reason, grouped 5-fold CV", fontsize=10)
    fig.tight_layout(); fig.savefig("fig_stage2_macro_f1.png", dpi=200)


if __name__ == "__main__":
    main()

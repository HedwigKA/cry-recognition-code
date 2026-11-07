"""Figures for the paper from stage1_results.json, stage2_results.json, and sentinel_demo_results.json.

All generated figures are saved to the output/ directory:
  output/fig_stage1_confusion.png
  output/fig_stage2_confusion.png
  output/fig_stage2_macro_f1.png
  output/fig_context_fusion.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "output"


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


def plot_context_fusion(demo_path: str):
    if not os.path.exists(demo_path):
        return
    demos = [d for d in json.load(open(demo_path)) if d.get("is_crying")]
    if not demos:
        return
    reasons = ["hungry", "belly_pain", "burping", "discomfort", "tired"]
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.2), sharey=True)
    for ax, d in zip(axs.flat, demos[:4]):
        x = np.arange(len(reasons))
        bw = 0.36
        p_aud = [d["stage2_audio_only"][r] for r in reasons]
        p_fus = [d["stage2_context_fused"][r] for r in reasons]
        ax.bar(x - bw / 2, p_aud, bw, label="Stage 2 (Audio Only)", color="#6baed6")
        ax.bar(x + bw / 2, p_fus, bw, label="Stage 2 + Incubator Context", color="#2171b5")
        for i, (va, vf) in enumerate(zip(p_aud, p_fus)):
            ax.text(i - bw / 2, va + 0.02, f"{va:.0%}", ha="center", fontsize=7)
            ax.text(i + bw / 2, vf + 0.02, f"{vf:.0%}", ha="center", fontsize=7, fontweight="bold")
        ax.set_xticks(x, reasons, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Probability", fontsize=8)
        short_title = d["scenario"].split("->")[0].strip()
        ax.set_title(f"{short_title}\n→ Final: {d['final_predicted_reason'].upper()} ({d['final_confidence']:.1%})",
                     fontsize=8.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axs.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Stage 2 Cry Reason: Audio-Only vs Incubator Sensor Context Fusion", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_context_fusion.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s1 = json.load(open("stage1_results.json"))
    s2 = json.load(open("stage2_results.json"))
    w = s1["window_3class"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cm_plot(ax, w["confusion"], w["labels"], "Stage 1, window level (5-fold)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_stage1_confusion.png"), dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, k in zip(axs, ["five_weighted", "five_unweighted"]):
        cm_plot(ax, s2["confusions"][k], s2["labels"]["five"],
                f"Stage 2, {k.replace('five_', '')} (3 seeds x 5 folds)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_stage2_confusion.png"), dpi=200)
    plt.close(fig)

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
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_stage2_macro_f1.png"), dpi=200)
    plt.close(fig)

    plot_context_fusion("sentinel_demo_results.json")
    print("Saved figures to", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()

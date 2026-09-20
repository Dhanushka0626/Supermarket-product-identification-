"""
Figures for the report (dataset, threshold tuning, confusion matrix, cross-validation, examples).

Owner: Member 2
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .report import draw_detections, plot_report, print_report, summarize  # noqa: E402


def class_counts(counts: pd.DataFrame, path: Path) -> None:
    ax = counts.plot.barh(figsize=(10, 7), width=0.8)
    ax.invert_yaxis(); ax.set_xlabel("number of products"); ax.set_title("Products per class")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def threshold_curve(curve: pd.DataFrame, best: float, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(curve.threshold, curve.count_acc, marker="o", label="correct number of products")
    plt.plot(curve.threshold, curve.cAcc, marker="o", label="correct shopping list (cAcc)")
    plt.axvline(best, ls="--", c="r")
    plt.xlabel("confidence threshold"); plt.ylabel("share of images"); plt.title("Threshold tuning (hold-out)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def confusion(y_true, y_pred, names, title: str, path: Path) -> None:
    from sklearn.metrics import confusion_matrix
    n = len(names)
    cm = confusion_matrix(y_true, y_pred, labels=range(n), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def cv_summary(cv: dict, path: Path) -> None:
    summary, folds = cv["summary"], cv["folds"]
    key = ["mAP50", "classification_acc", "product_level_acc", "count_acc", "cAcc"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].bar(key, summary.loc[key, "mean"], yerr=summary.loc[key, "std"].fillna(0), capsize=6, color="#4C78A8",
              label="CV mean ± std")
    ax[0].scatter(key, summary.loc[key, "final_model_test2019"], color="#E45756", zorder=3, s=60,
                  label="final model on test2019")
    ax[0].axhline(0.8, ls="--", c="grey", label="80% target")
    ax[0].set_ylim(0, 1.05); ax[0].legend(); ax[0].set_title(f"{len(folds)}-fold cross-validation")
    for m in key:
        ax[1].plot(folds.fold, folds[m], marker="o", label=m)
    ax[1].set_xticks(folds.fold); ax[1].set_xlabel("fold"); ax[1].set_ylim(0, 1.05)
    ax[1].legend(); ax[1].set_title("Metrics per fold")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def cv_per_class(per_class: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(per_class.index, per_class.AP50_mean, xerr=per_class.AP50_std.fillna(0), capsize=4, color="#72B7B2")
    ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_xlabel("AP50 (mean ± std over folds)"); ax.set_title("Per-class AP50 - cross-validation")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def examples(pipe, groups, meta_names, fig_dir: Path, seed: int = 7) -> None:
    """One annotated example per difficulty level, with its report and chart."""
    rng = np.random.default_rng(seed)
    levels = [g[1] for g in groups]
    for lv in sorted(set(levels)):
        i = int(rng.choice([k for k, l in enumerate(levels) if l == lv]))
        path, _, _, gcls = groups[i]
        proc, dets = pipe(cv2.imread(path))
        cv2.imwrite(str(fig_dir / f"example_{lv}.jpg"), draw_detections(proc, dets, meta_names))
        df = print_report(dets, meta_names, title=f"{Path(path).name} ({lv})")
        true_counts = Counter(meta_names[m] for m in gcls)
        df["true_count"] = df.category.map(true_counts).fillna(0).astype(int)
        df.to_csv(fig_dir / f"example_{lv}_summary.csv", index=False)
        plot_report(summarize(dets, meta_names), fig_dir / f"example_{lv}_chart.png", title=f"{Path(path).name} ({lv})")

"""
Visualization module for classification metrics.

Generates confusion matrix heatmaps, per-class metric bar charts,
and comparison charts across classification tasks.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from f1_metrics import calculate_f1_metrics

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")

matplotlib.use("Agg")


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", ax=None):
    """Plot a confusion matrix as a heatmap."""
    metrics = calculate_f1_metrics(y_true, y_pred, labels)
    cm = metrics["confusion_matrix"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    # Add text annotations in each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    return fig


def plot_per_class_metrics(y_true, y_pred, labels, title="Per-Class Metrics", ax=None):
    """Plot precision, recall, and F1 per class as grouped bar chart."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#2196F3")
    bars2 = ax.bar(x, recall, width, label="Recall", color="#FF9800")
    bars3 = ax.bar(x + width, f1, width, label="F1 Score", color="#4CAF50")

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, height + 0.01,
                f"{height:.2f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_comparison_chart(results, title="Model Comparison"):
    """
    Plot a comparison of macro-averaged metrics across multiple tasks.

    Args:
        results: list of dicts with keys "name", "y_true", "y_pred", "labels"
        title: chart title
    """
    names = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for r in results:
        metrics = calculate_f1_metrics(r["y_true"], r["y_pred"], r["labels"])
        names.append(r["name"])
        f1_scores.append(metrics["f1_macro"])
        precision_scores.append(metrics["precision_macro"])
        recall_scores.append(metrics["recall_macro"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.25

    bars1 = ax.bar(x - width, precision_scores, width, label="Precision (Macro)", color="#2196F3")
    bars2 = ax.bar(x, recall_scores, width, label="Recall (Macro)", color="#FF9800")
    bars3 = ax.bar(x + width, f1_scores, width, label="F1 (Macro)", color="#4CAF50")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, height + 0.01,
                f"{height:.2f}", ha="center", va="bottom", fontsize=10,
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def generate_all_visualizations(results):
    """
    Generate all visualizations and save them as PNG files.

    Args:
        results: list of dicts with keys "name", "y_true", "y_pred", "labels"
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)

    for r in results:
        # Confusion matrix
        fig = plot_confusion_matrix(
            r["y_true"], r["y_pred"], r["labels"],
            title=f"Confusion Matrix — {r['name']}",
        )
        fig.tight_layout()
        filename = os.path.join(CHARTS_DIR, f"confusion_matrix_{r['name'].lower().replace(' ', '_')}.png")
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved {filename}")
        plt.close(fig)

        # Per-class metrics
        fig = plot_per_class_metrics(
            r["y_true"], r["y_pred"], r["labels"],
            title=f"Per-Class Metrics — {r['name']}",
        )
        fig.tight_layout()
        filename = os.path.join(CHARTS_DIR, f"per_class_metrics_{r['name'].lower().replace(' ', '_')}.png")
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved {filename}")
        plt.close(fig)

    # Comparison chart
    if len(results) > 1:
        fig = plot_comparison_chart(results)
        filename = os.path.join(CHARTS_DIR, "comparison_chart.png")
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved {filename}")
        plt.close(fig)

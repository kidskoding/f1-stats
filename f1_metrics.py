"""
F1 Metrics Calculator

Demonstrates various F1 score calculations for classification tasks.
"""

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import numpy as np


def calculate_f1_metrics(y_true: list, y_pred: list, labels: list = None):
    """
    Calculate comprehensive F1 metrics for classification results.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional list of label names for display

    Returns:
        Dictionary containing all F1 metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Determine if binary or multi-class
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    is_binary = len(unique_classes) == 2

    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_per_class": f1_score(y_true, y_pred, average=None),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if is_binary:
        metrics["f1_binary"] = f1_score(y_true, y_pred, average="binary")

    return metrics


def print_metrics_report(y_true: list, y_pred: list, labels: list = None):
    """Print a formatted report of all F1 metrics."""
    metrics = calculate_f1_metrics(y_true, y_pred, labels)

    print("=" * 50)
    print("F1 METRICS REPORT")
    print("=" * 50)

    if "f1_binary" in metrics:
        print(f"\nF1 Score (Binary):    {metrics['f1_binary']:.4f}")

    print(f"\nF1 Score (Macro):     {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Micro):     {metrics['f1_micro']:.4f}")
    print(f"F1 Score (Weighted):  {metrics['f1_weighted']:.4f}")
    print(f"\nPrecision (Macro):    {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):       {metrics['recall_macro']:.4f}")

    print(f"\nF1 Per Class: {metrics['f1_per_class']}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\n" + "=" * 50)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=labels))


def manual_f1_calculation(tp: int, fp: int, fn: int) -> dict:
    """
    Calculate F1 score manually from confusion matrix values.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        Dictionary with precision, recall, and F1 score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    # Example 1: Binary Classification
    print("\n" + "#" * 50)
    print("EXAMPLE 1: Binary Classification (Spam Detection)")
    print("#" * 50)

    y_true_binary = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]  # 1 = spam, 0 = not spam
    y_pred_binary = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

    print_metrics_report(y_true_binary, y_pred_binary, labels=["Not Spam", "Spam"])

    # Example 2: Multi-class Classification
    print("\n" + "#" * 50)
    print("EXAMPLE 2: Multi-class Classification (Sentiment)")
    print("#" * 50)

    # 0 = negative, 1 = neutral, 2 = positive
    y_true_multi = [0, 1, 2, 0, 1, 2, 0, 2, 1, 0, 2, 1]
    y_pred_multi = [0, 2, 2, 0, 1, 1, 0, 2, 0, 0, 2, 1]

    print_metrics_report(y_true_multi, y_pred_multi, labels=["Negative", "Neutral", "Positive"])

    # Example 3: Manual F1 Calculation
    print("\n" + "#" * 50)
    print("EXAMPLE 3: Manual F1 Calculation")
    print("#" * 50)

    # From a confusion matrix: TP=45, FP=5, FN=10
    result = manual_f1_calculation(tp=45, fp=5, fn=10)
    print(f"\nGiven: TP=45, FP=5, FN=10")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1 Score:  {result['f1']:.4f}")

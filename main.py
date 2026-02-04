from f1_metrics import print_metrics_report, manual_f1_calculation
from visualizations import generate_all_visualizations

def main():
    # Binary Classification Example
    print("\n" + "#" * 50)
    print("Binary Classification (Spam Detection)")
    print("#" * 50)

    y_true_binary = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred_binary = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    print_metrics_report(y_true_binary, y_pred_binary, labels=["Not Spam", "Spam"])

    # Multi-class Classification Example
    print("\n" + "#" * 50)
    print("Multi-class Classification (Sentiment)")
    print("#" * 50)

    y_true_multi = [0, 1, 2, 0, 1, 2, 0, 2, 1, 0, 2, 1]
    y_pred_multi = [0, 2, 2, 0, 1, 1, 0, 2, 0, 0, 2, 1]
    print_metrics_report(y_true_multi, y_pred_multi, labels=["Negative", "Neutral", "Positive"])

    # Generate visualizations
    print("\n" + "#" * 50)
    print("Generating Visualizations")
    print("#" * 50)

    results = [
        {
            "name": "Spam Detection",
            "y_true": y_true_binary,
            "y_pred": y_pred_binary,
            "labels": ["Not Spam", "Spam"],
        },
        {
            "name": "Sentiment Analysis",
            "y_true": y_true_multi,
            "y_pred": y_pred_multi,
            "labels": ["Negative", "Neutral", "Positive"],
        },
    ]
    generate_all_visualizations(results)
    print("\nAll visualizations saved as PNG files.")

if __name__ == "__main__":
    main()

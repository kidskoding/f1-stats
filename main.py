from f1_metrics import print_metrics_report, manual_f1_calculation
from visualizations import generate_all_visualizations
from f1_data import load_all_data
from f1_visualizations import generate_all_f1_visualizations

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
    print("\nAll ML visualizations saved as PNG files.")

    # F1 Driver Data Visualizations
    print("\n" + "#" * 50)
    print("F1 Driver Data Visualizations (2015-2024)")
    print("#" * 50)

    print("\nFetching F1 data (cached after first run)...")
    f1_data = load_all_data()

    print(f"\nDataset sizes:")
    for name, df in f1_data.items():
        print(f"  {name}: {len(df)} rows")

    print("\nGenerating F1 visualizations...")
    generate_all_f1_visualizations(f1_data)
    print("\nAll F1 visualizations saved to charts/ directory.")

if __name__ == "__main__":
    main()

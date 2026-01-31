from f1_metrics import print_metrics_report, manual_f1_calculation

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

if __name__ == "__main__":
    main()

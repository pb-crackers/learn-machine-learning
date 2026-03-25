"""
Lab 2.3: Build a Classifier and Compute Evaluation Metrics

Train a logistic regression classifier, compute metrics by hand,
and explore the precision-recall tradeoff.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve,
)


# ================================================================
# PART 1: Load and Explore
# ================================================================

def load_and_explore():
    """Load the breast cancer dataset and examine class distribution."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    target_names = data.target_names

    print("=" * 50)
    print("PART 1: Data Exploration")
    print("=" * 50)
    print(f"Shape: {X.shape}")
    print(f"Classes: {target_names}")

    # TODO: Print the class distribution
    # Hint: use np.unique(y, return_counts=True)
    # Is the dataset balanced?

    # TODO: Split into train/test with stratification
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )

    X_train, X_test, y_train, y_test = None, None, None, None  # Replace
    return X_train, X_test, y_train, y_test, target_names


# ================================================================
# PART 2: Train Classifier
# ================================================================

def train_classifier(X_train, y_train):
    """Build and train a logistic regression pipeline."""
    print("\n" + "=" * 50)
    print("PART 2: Training Classifier")
    print("=" * 50)

    # TODO: Create a pipeline with StandardScaler + LogisticRegression
    # pipeline = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    # ])
    # pipeline.fit(X_train, y_train)

    pipeline = None  # Replace with your code
    return pipeline


# ================================================================
# PART 3: Compute Metrics Manually
# ================================================================

def compute_metrics_manually(y_true, y_pred, target_names):
    """Compute confusion matrix and metrics by hand, then verify with sklearn."""
    print("\n" + "=" * 50)
    print("PART 3: Manual Metrics Computation")
    print("=" * 50)

    # TODO: Compute TP, FP, TN, FN by hand
    # For binary classification with positive class = 1 (malignant):
    # TP = np.sum((y_pred == 1) & (y_true == 1))
    # FP = np.sum((y_pred == 1) & (y_true == 0))
    # TN = np.sum((y_pred == 0) & (y_true == 0))
    # FN = np.sum((y_pred == 0) & (y_true == 1))

    # TODO: Compute metrics from TP, FP, TN, FN
    # accuracy_manual = (TP + TN) / (TP + TN + FP + FN)
    # precision_manual = TP / (TP + FP) if (TP + FP) > 0 else 0
    # recall_manual = TP / (TP + FN) if (TP + FN) > 0 else 0
    # f1_manual = 2 * precision_manual * recall_manual / (precision_manual + recall_manual)

    # TODO: Print your manual calculations
    # print(f"Confusion Matrix:")
    # print(f"  TP={TP}, FP={FP}")
    # print(f"  FN={FN}, TN={TN}")
    # print(f"\nManual Metrics:")
    # print(f"  Accuracy:  {accuracy_manual:.4f}")
    # print(f"  Precision: {precision_manual:.4f}")
    # print(f"  Recall:    {recall_manual:.4f}")
    # print(f"  F1 Score:  {f1_manual:.4f}")

    # TODO: Verify against sklearn
    # print(f"\nSklearn Metrics:")
    # print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    # print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    # print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")
    # print(f"  F1 Score:  {f1_score(y_true, y_pred):.4f}")

    # TODO: Print full classification report
    # print(f"\nClassification Report:")
    # print(classification_report(y_true, y_pred, target_names=target_names))

    pass  # Remove after implementing


# ================================================================
# PART 4: Threshold Analysis
# ================================================================

def threshold_analysis(pipeline, X_test, y_test):
    """Explore how the classification threshold affects precision and recall."""
    print("\n" + "=" * 50)
    print("PART 4: Threshold Analysis")
    print("=" * 50)

    # TODO: Get predicted probabilities for the positive class
    # y_proba = pipeline.predict_proba(X_test)[:, 1]

    # TODO: Compute precision and recall at various thresholds
    # precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # TODO: Find the threshold that maximizes F1
    # f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    # best_idx = np.argmax(f1_scores)
    # best_threshold = thresholds[best_idx]
    # print(f"Best threshold for F1: {best_threshold:.3f}")
    # print(f"  Precision at best: {precisions[best_idx]:.3f}")
    # print(f"  Recall at best:    {recalls[best_idx]:.3f}")
    # print(f"  F1 at best:        {f1_scores[best_idx]:.3f}")

    # TODO: Plot precision and recall vs threshold
    # plt.figure(figsize=(8, 5))
    # plt.plot(thresholds, precisions[:-1], label="Precision")
    # plt.plot(thresholds, recalls[:-1], label="Recall")
    # plt.axvline(x=best_threshold, color="gray", linestyle="--", label=f"Best threshold={best_threshold:.2f}")
    # plt.xlabel("Threshold")
    # plt.ylabel("Score")
    # plt.title("Precision and Recall vs. Classification Threshold")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    pass  # Remove after implementing


# ================================================================
# PART 5: ROC Curve
# ================================================================

def plot_roc_curve(pipeline, X_test, y_test):
    """Plot the ROC curve and compute AUC."""
    print("\n" + "=" * 50)
    print("PART 5: ROC Curve")
    print("=" * 50)

    # TODO: Get predicted probabilities
    # y_proba = pipeline.predict_proba(X_test)[:, 1]

    # TODO: Compute ROC curve and AUC
    # fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    # auc = roc_auc_score(y_test, y_proba)
    # print(f"AUC-ROC: {auc:.4f}")

    # TODO: Plot the ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, linewidth=2, label=f"Logistic Regression (AUC={auc:.3f})")
    # plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Classifier")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    pass  # Remove after implementing


# ================================================================
# MAIN
# ================================================================

def main():
    print("Lab 2.3: Classification and Evaluation Metrics")
    print("=" * 50)

    # Part 1
    X_train, X_test, y_train, y_test, target_names = load_and_explore()

    if X_train is None:
        print("\nPlease complete Part 1 (data loading and splitting) first.")
        return

    # Part 2
    pipeline = train_classifier(X_train, y_train)

    if pipeline is None:
        print("\nPlease complete Part 2 (train classifier) first.")
        return

    # Generate predictions
    y_pred = pipeline.predict(X_test)

    # Part 3
    compute_metrics_manually(y_test, y_pred, target_names)

    # Part 4
    threshold_analysis(pipeline, X_test, y_test)

    # Part 5
    plot_roc_curve(pipeline, X_test, y_test)

    print("\n" + "=" * 50)
    print("Lab complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

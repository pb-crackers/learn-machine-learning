"""
Lab 2.4: Decision Trees vs. Random Forests

Train, visualize, and compare tree-based models.
Fill in the TODO sections to complete each part.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report


# ================================================================
# PART 1: Load and Prepare Data
# ================================================================

def load_data():
    """Load the Wine dataset and split into train/test."""
    print("=" * 50)
    print("PART 1: Data Loading")
    print("=" * 50)

    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names

    print(f"Shape: {X.shape}")
    print(f"Classes: {target_names}")
    print(f"Features: {feature_names}")

    # TODO: Split into train/test (80/20, stratified)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )

    X_train, X_test, y_train, y_test = None, None, None, None  # Replace
    return X_train, X_test, y_train, y_test, feature_names, target_names


# ================================================================
# PART 2: Train and Visualize a Decision Tree
# ================================================================

def train_and_visualize_tree(X_train, y_train, feature_names, target_names):
    """Train a decision tree and visualize it."""
    print("\n" + "=" * 50)
    print("PART 2: Decision Tree Visualization")
    print("=" * 50)

    # TODO: Train a DecisionTreeClassifier with max_depth=3
    # tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    # tree.fit(X_train, y_train)
    tree = None  # Replace

    # TODO: Visualize the tree
    # plt.figure(figsize=(20, 10))
    # plot_tree(
    #     tree,
    #     feature_names=feature_names,
    #     class_names=target_names,
    #     filled=True,
    #     rounded=True,
    #     fontsize=9,
    # )
    # plt.title("Decision Tree (max_depth=3)")
    # plt.tight_layout()
    # plt.show()

    # TODO: Print feature importances
    # importances = tree.feature_importances_
    # for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    #     if imp > 0:
    #         print(f"  {name:30s}: {imp:.4f}")

    return tree


# ================================================================
# PART 3: Overfitting Analysis
# ================================================================

def overfitting_analysis(X_train, y_train):
    """Analyze how max_depth affects overfitting."""
    print("\n" + "=" * 50)
    print("PART 3: Overfitting Analysis")
    print("=" * 50)

    depths = range(1, 21)
    train_scores = []
    cv_scores = []

    for depth in depths:
        # TODO: Train a tree at this depth and record scores
        # tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        # tree.fit(X_train, y_train)
        # train_scores.append(tree.score(X_train, y_train))
        # cv = cross_val_score(tree, X_train, y_train, cv=5, scoring="accuracy")
        # cv_scores.append(cv.mean())
        pass  # Replace with your code

    # TODO: Plot training vs CV accuracy
    # plt.figure(figsize=(8, 5))
    # plt.plot(depths, train_scores, "o-", label="Training Accuracy")
    # plt.plot(depths, cv_scores, "o-", label="CV Accuracy")
    # plt.xlabel("max_depth")
    # plt.ylabel("Accuracy")
    # plt.title("Decision Tree: Overfitting vs. max_depth")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # TODO: Find and print the best depth
    # best_depth = depths[np.argmax(cv_scores)]
    # print(f"Best max_depth: {best_depth} (CV accuracy: {max(cv_scores):.3f})")

    # return best_depth
    return 3  # Placeholder


# ================================================================
# PART 4: Random Forest
# ================================================================

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """Train a random forest and compare with single tree."""
    print("\n" + "=" * 50)
    print("PART 4: Random Forest")
    print("=" * 50)

    # TODO: Train a RandomForestClassifier
    # forest = RandomForestClassifier(n_estimators=100, random_state=42)
    # forest.fit(X_train, y_train)
    forest = None  # Replace

    # TODO: Compare CV accuracy
    # tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    # tree_scores = cross_val_score(tree, X_train, y_train, cv=5)
    # forest_scores = cross_val_score(forest, X_train, y_train, cv=5)
    # print(f"Single Tree CV: {tree_scores.mean():.3f} +/- {tree_scores.std():.3f}")
    # print(f"Random Forest CV: {forest_scores.mean():.3f} +/- {forest_scores.std():.3f}")

    # TODO: Plot feature importances
    # importances = forest.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # plt.figure(figsize=(10, 5))
    # plt.bar(range(len(importances)), importances[indices])
    # plt.xticks(range(len(importances)),
    #            [feature_names[i] for i in indices], rotation=45, ha="right")
    # plt.title("Random Forest Feature Importances")
    # plt.ylabel("Importance")
    # plt.tight_layout()
    # plt.show()

    return forest


# ================================================================
# PART 5: Ensemble Comparison
# ================================================================

def compare_ensembles(X_train, y_train):
    """Compare multiple ensemble methods."""
    print("\n" + "=" * 50)
    print("PART 5: Ensemble Comparison")
    print("=" * 50)

    # TODO: Define models to compare
    models = {
        "Decision Tree (depth=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest (100)": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost (100)": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        ),
    }

    # TODO: Cross-validate each model and print results
    # print(f"{'Model':35s}  {'Accuracy':>10s}")
    # print("-" * 50)
    # for name, model in models.items():
    #     scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    #     print(f"{name:35s}  {scores.mean():.3f} +/- {scores.std():.3f}")

    pass  # Remove after implementing


# ================================================================
# MAIN
# ================================================================

def main():
    print("Lab 2.4: Decision Trees vs. Random Forests")
    print("=" * 50)

    # Part 1
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()

    if X_train is None:
        print("\nPlease complete Part 1 (data loading) first.")
        return

    # Part 2
    tree = train_and_visualize_tree(X_train, y_train, feature_names, target_names)

    # Part 3
    best_depth = overfitting_analysis(X_train, y_train)

    # Part 4
    forest = train_random_forest(X_train, y_train, X_test, y_test, feature_names)

    # Part 5
    compare_ensembles(X_train, y_train)

    # Final test set evaluation
    if forest is not None:
        print("\n" + "=" * 50)
        print("Final Test Set Evaluation (Random Forest)")
        print("=" * 50)
        y_pred = forest.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=target_names))

    print("\n" + "=" * 50)
    print("Lab complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

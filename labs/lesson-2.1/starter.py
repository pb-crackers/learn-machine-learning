"""
Lab 2.1: Complete Toy ML Pipeline with Scikit-Learn

Build a full ML pipeline: load, explore, preprocess, train, evaluate.
Fill in the TODO sections to complete each step.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_and_explore():
    """Part 1: Load the California Housing dataset and explore it."""
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names

    print("=" * 50)
    print("PART 1: Data Exploration")
    print("=" * 50)
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {feature_names}")

    # TODO: Print basic statistics for each feature (min, max, mean, std)
    # Hint: use a loop over columns or convert to a DataFrame

    # TODO: Create a correlation heatmap
    # Hint: np.corrcoef or pandas .corr(), then plt.imshow()

    return X, y, feature_names


def split_data(X, y):
    """Part 2: Split data into train, validation, and test sets."""
    print("\n" + "=" * 50)
    print("PART 2: Data Splitting")
    print("=" * 50)

    # TODO: Split into 60% train, 20% validation, 20% test
    # Hint: Use train_test_split twice
    # First split: separate test set (20%)
    # Second split: separate validation from remaining (25% of 80% = 20% overall)
    X_temp, X_test, y_temp, y_test = None, None, None, None  # Replace with your code
    X_train, X_val, y_train, y_val = None, None, None, None  # Replace with your code

    # TODO: Print the size of each set
    # print(f"Training set:   {X_train.shape[0]} samples")
    # print(f"Validation set: {X_val.shape[0]} samples")
    # print(f"Test set:       {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipelines():
    """Part 3: Create preprocessing + model pipelines."""
    print("\n" + "=" * 50)
    print("PART 3: Building Pipelines")
    print("=" * 50)

    # TODO: Create a dictionary of pipelines, each with StandardScaler + model
    # Example structure:
    # pipelines = {
    #     "Linear Regression": Pipeline([
    #         ("scaler", StandardScaler()),
    #         ("model", LinearRegression()),
    #     ]),
    #     ...
    # }
    pipelines = {}  # Replace with your code

    print(f"Created {len(pipelines)} pipelines")
    return pipelines


def compare_models(pipelines, X_train, y_train):
    """Part 4: Compare models using cross-validation."""
    print("\n" + "=" * 50)
    print("PART 4: Model Comparison (5-Fold CV)")
    print("=" * 50)

    results = {}
    best_name = None
    best_score = -np.inf

    for name, pipeline in pipelines.items():
        # TODO: Run 5-fold cross-validation with scoring="r2"
        # Hint: scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
        scores = None  # Replace with your code

        # TODO: Store results and track the best model
        # mean_score = scores.mean()
        # results[name] = {"mean": mean_score, "std": scores.std()}
        # if mean_score > best_score:
        #     best_score = mean_score
        #     best_name = name

        pass  # Remove this line after adding your code

    # TODO: Print results in a formatted table
    # for name, result in results.items():
    #     print(f"  {name:25s}  R^2 = {result['mean']:.4f} +/- {result['std']:.4f}")

    print(f"\nBest model: {best_name} (R^2 = {best_score:.4f})")
    return best_name


def final_evaluation(pipelines, best_name, X_train, y_train, X_test, y_test):
    """Part 5: Train the best model and evaluate on the test set."""
    print("\n" + "=" * 50)
    print("PART 5: Final Evaluation on Test Set")
    print("=" * 50)

    # TODO: Fit the best pipeline on the full training set
    best_pipeline = pipelines[best_name]
    # best_pipeline.fit(X_train, y_train)

    # TODO: Predict on the test set
    # y_pred = best_pipeline.predict(X_test)

    # TODO: Compute and print test R^2 and RMSE
    # test_r2 = r2_score(y_test, y_pred)
    # test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(f"Test R^2:  {test_r2:.4f}")
    # print(f"Test RMSE: {test_rmse:.4f}")

    # TODO (Stretch): Plot predicted vs actual values
    # plt.figure(figsize=(8, 6))
    # plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    # plt.xlabel("Actual")
    # plt.ylabel("Predicted")
    # plt.title(f"Predicted vs Actual ({best_name})")
    # plt.show()

    pass  # Remove after implementing


def main():
    """Run the complete ML pipeline."""
    # Part 1: Load and explore
    X, y, feature_names = load_and_explore()

    # Part 2: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Part 3: Build pipelines
    pipelines = build_pipelines()

    # Part 4: Compare models
    if pipelines and X_train is not None:
        best_name = compare_models(pipelines, X_train, y_train)

        # Part 5: Final evaluation
        if best_name:
            final_evaluation(pipelines, best_name, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

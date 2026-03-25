"""
Lab 0.2 — Pandas, Matplotlib & the ML Data Pipeline
=====================================================
Complete each function marked with TODO.
Run this file to check your solutions: python starter.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save to files
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory for plots
os.makedirs("plots", exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: Load the California Housing dataset
# ---------------------------------------------------------------------------

def load_housing_data():
    """Load California Housing as a Pandas DataFrame. No changes needed."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame  # DataFrame with features + target ("MedHouseVal")

    # Inject some artificial missing values for the cleaning exercise
    rng = np.random.default_rng(42)
    mask_income = rng.random(len(df)) < 0.05    # ~5% missing
    mask_age = rng.random(len(df)) < 0.03       # ~3% missing
    df.loc[mask_income, "MedInc"] = np.nan
    df.loc[mask_age, "HouseAge"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Exercise 1: Explore the Dataset
# ---------------------------------------------------------------------------

def explore_dataset(df):
    """Print key facts about the dataset.

    TODO: Fill in each variable below using the appropriate Pandas method.

    Args:
        df: pandas DataFrame

    Returns:
        A dict with keys: "shape", "n_missing", "mean_income", "median_house_value"
    """
    # TODO: Number of rows and columns as a tuple, e.g. (20640, 9)
    shape = None

    # TODO: Total number of missing values across the entire DataFrame (a single int)
    # Hint: df.isnull().sum() gives per-column counts
    n_missing = None

    # TODO: Mean of the "MedInc" column, ignoring NaN values (a float)
    mean_income = None

    # TODO: Median of the "MedHouseVal" column (a float)
    median_house_value = None

    return {
        "shape": shape,
        "n_missing": n_missing,
        "mean_income": mean_income,
        "median_house_value": median_house_value,
    }


def test_explore(df):
    """Test exploration. No changes needed."""
    results = explore_dataset(df)
    assert results["shape"] is not None, "shape is None -- did you implement it?"
    assert results["shape"][0] > 20000, f"Expected >20000 rows, got {results['shape'][0]}"
    assert results["shape"][1] >= 9, f"Expected >=9 columns, got {results['shape'][1]}"
    assert results["n_missing"] > 0, "n_missing should be > 0 (we injected NaNs)"
    assert results["mean_income"] is not None, "mean_income is None"
    assert results["median_house_value"] is not None, "median_house_value is None"
    print(f"  Shape: {results['shape']}")
    print(f"  Missing values: {results['n_missing']}")
    print(f"  Mean income: {results['mean_income']:.4f}")
    print(f"  Median house value: {results['median_house_value']:.4f}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: Clean the Data
# ---------------------------------------------------------------------------

def clean_dataset(df):
    """Clean the dataset for ML readiness.

    Steps to implement:
    1. Fill missing values in "MedInc" with the column median
    2. Fill missing values in "HouseAge" with the column median
    3. Remove rows where "MedHouseVal" >= 5.0 (these are capped/clipped values)
    4. Create a new column "RoomsPerHousehold" = "AveRooms" / "AveOccup"

    Args:
        df: pandas DataFrame (will NOT be modified in place -- work on a copy)

    Returns:
        Cleaned pandas DataFrame

    TODO: Implement the four cleaning steps described above.
    """
    df = df.copy()  # don't modify the original

    # TODO Step 1: Fill MedInc NaN with median
    # Hint: df["col"] = df["col"].fillna(df["col"].median())

    # TODO Step 2: Fill HouseAge NaN with median

    # TODO Step 3: Remove capped values (MedHouseVal >= 5.0)
    # Hint: df = df[df["MedHouseVal"] < 5.0]

    # TODO Step 4: Create RoomsPerHousehold column

    return df


def test_clean(df):
    """Test cleaning. No changes needed."""
    df_clean = clean_dataset(df)
    assert df_clean is not None, "clean_dataset returned None"
    assert df_clean.isnull().sum().sum() == 0, (
        f"Still have {df_clean.isnull().sum().sum()} missing values after cleaning"
    )
    assert (df_clean["MedHouseVal"] < 5.0).all(), (
        "Found MedHouseVal >= 5.0 -- capped values not removed"
    )
    assert "RoomsPerHousehold" in df_clean.columns, (
        "Missing column: RoomsPerHousehold"
    )
    print(f"  Cleaned shape: {df_clean.shape}")
    print(f"  Missing values: {df_clean.isnull().sum().sum()}")
    print(f"  Max MedHouseVal: {df_clean['MedHouseVal'].max():.4f}")
    print("  PASSED")
    return df_clean


# ---------------------------------------------------------------------------
# Exercise 3: Visualize
# ---------------------------------------------------------------------------

def plot_target_distribution(df):
    """Create a histogram of the target variable (MedHouseVal).

    TODO:
    - Create a figure with plt.subplots
    - Plot a histogram of df["MedHouseVal"] with 50 bins
    - Add a title, x-label, and y-label
    - Save to "plots/target_distribution.png"
    """
    # TODO: your code here
    pass


def plot_correlation_heatmap(df):
    """Create a correlation heatmap of all numeric columns.

    TODO:
    - Compute the correlation matrix with df.corr()
    - Create a figure (at least 8x6 inches)
    - Use sns.heatmap with annot=True, fmt=".2f", cmap="coolwarm"
    - Add a title
    - Save to "plots/correlation_heatmap.png"
    """
    # TODO: your code here
    pass


def plot_income_vs_value(df):
    """Create a scatter plot of MedInc (x) vs MedHouseVal (y).

    TODO:
    - Create a figure with plt.subplots
    - Scatter plot with alpha=0.2 and s=5 for readability
    - Add title, x-label, y-label
    - Save to "plots/income_vs_value.png"
    """
    # TODO: your code here
    pass


def plot_value_by_age_group(df):
    """Create a box plot of MedHouseVal grouped by HouseAge bins.

    TODO:
    - Create age bins: [0, 15, 30, 45, 60] with labels ["New", "Mid", "Old", "Very Old"]
      using pd.cut on df["HouseAge"]
    - Create a box plot with sns.boxplot (x=age groups, y=MedHouseVal)
    - Add title and labels
    - Save to "plots/value_by_age_group.png"
    """
    # TODO: your code here
    pass


def test_plots(df):
    """Test that all plot files were created. No changes needed."""
    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_income_vs_value(df)
    plot_value_by_age_group(df)

    expected_files = [
        "plots/target_distribution.png",
        "plots/correlation_heatmap.png",
        "plots/income_vs_value.png",
        "plots/value_by_age_group.png",
    ]
    for f in expected_files:
        assert os.path.exists(f), f"Plot not found: {f} -- did you save it?"
        print(f"  Created: {f}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Exercise 4: Prepare for Modeling
# ---------------------------------------------------------------------------

def prepare_for_modeling(df, target_col="MedHouseVal", test_fraction=0.2):
    """Split features/target and apply standardization.

    Steps:
    1. Separate the DataFrame into feature columns (X) and target (y) as NumPy arrays.
    2. Split into training and test sets. Use the first (1 - test_fraction) rows
       for training and the remaining rows for testing (simple split, no shuffle needed).
    3. Compute mean and std from X_train ONLY.
    4. Standardize both X_train and X_test using those training statistics.

    Args:
        df: cleaned pandas DataFrame
        target_col: name of the target column
        test_fraction: fraction of data to use as test set

    Returns:
        A dict with keys: "X_train", "X_test", "y_train", "y_test",
                          "train_mean", "train_std"

    TODO: Implement the steps above. Do NOT use sklearn for this exercise.
    """
    # TODO Step 1: Separate features and target
    # feature_cols = [c for c in df.columns if c != target_col]
    # X = ...
    # y = ...

    # TODO Step 2: Split into train/test
    # split_idx = int(len(X) * (1 - test_fraction))
    # X_train, X_test = ...
    # y_train, y_test = ...

    # TODO Step 3: Compute training statistics
    # train_mean = ...
    # train_std = ...

    # TODO Step 4: Standardize (use broadcasting!)
    # X_train_scaled = ...
    # X_test_scaled = ...

    return {
        "X_train": None,   # replace with X_train_scaled
        "X_test": None,    # replace with X_test_scaled
        "y_train": None,   # replace with y_train
        "y_test": None,    # replace with y_test
        "train_mean": None,  # replace with train_mean
        "train_std": None,   # replace with train_std
    }


def test_prepare(df):
    """Test data preparation. No changes needed."""
    result = prepare_for_modeling(df)

    assert result["X_train"] is not None, "X_train is None -- did you implement prepare_for_modeling?"

    n_total = len(df)
    n_train = result["X_train"].shape[0]
    n_test = result["X_test"].shape[0]

    assert n_train + n_test == n_total, (
        f"Train ({n_train}) + Test ({n_test}) != Total ({n_total})"
    )
    assert n_test == int(n_total * 0.2), (
        f"Test size {n_test} != expected {int(n_total * 0.2)}"
    )

    # Training features should have mean ≈ 0
    train_means = result["X_train"].mean(axis=0)
    assert np.allclose(train_means, 0, atol=1e-10), (
        f"Training feature means should be ~0, max deviation: {np.abs(train_means).max():.6f}"
    )

    print(f"  X_train shape: {result['X_train'].shape}")
    print(f"  X_test shape:  {result['X_test'].shape}")
    print(f"  y_train shape: {result['y_train'].shape}")
    print(f"  y_test shape:  {result['y_test'].shape}")
    print(f"  Train feature means (should be ~0): {train_means.round(6)}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Run all exercises
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Lab 0.2: Pandas, Matplotlib & the ML Data Pipeline")
    print("=" * 60)

    df = load_housing_data()

    print("\nExercise 1: Explore the Dataset")
    test_explore(df)

    print("\nExercise 2: Clean the Data")
    df_clean = test_clean(df)

    print("\nExercise 3: Visualize")
    test_plots(df_clean)

    print("\nExercise 4: Prepare for Modeling")
    test_prepare(df_clean)

    print("\n" + "=" * 60)
    print("All exercises passed! Great work.")
    print("=" * 60)

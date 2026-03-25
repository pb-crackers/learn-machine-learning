"""
Lab 5.1 Starter — Data Pipeline & Exploration
================================================
Follow the numbered steps below. Replace the placeholder comments with your
own code. Each step maps to a section in the lesson.

Usage:
    python starter.py

Or run interactively in a Jupyter notebook by copying cells.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Create output directories
os.makedirs("plots", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# =============================================================================
# STEP 1: Load the Dataset
# =============================================================================
# Choose ONE of the options below and uncomment it, or write your own loader.

# Option A: California Housing (regression)
# from sklearn.datasets import fetch_california_housing
# housing = fetch_california_housing(as_frame=True)
# df = housing.frame
# TARGET_COL = "MedHouseVal"

# Option B: Heart Disease from CSV (classification)
# df = pd.read_csv("data/heart.csv")
# TARGET_COL = "target"

# Option C: Wine Quality from CSV (classification or regression)
# df = pd.read_csv("data/winequality-red.csv", sep=";")
# TARGET_COL = "quality"

# Option D: Your own dataset
# df = pd.read_csv("data/your_dataset.csv")
# TARGET_COL = "your_target_column"

# --- Uncomment one option above, then continue ---

# Verify the dataset loaded
# print(f"Dataset shape: {df.shape}")
# print(f"Target column: {TARGET_COL}")


# =============================================================================
# STEP 2: Inspect the Data
# =============================================================================
# Run each command and write your observations as comments.

def inspect_data(df):
    """Print basic information about the DataFrame."""
    print("=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nStatistical summary:\n{df.describe()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nMissing percentage:\n{(df.isnull().sum() / len(df) * 100).round(2)}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


# Uncomment when ready:
# inspect_data(df)

# ANSWER THESE QUESTIONS (write your answers as comments):
# Q1: How many samples and features?
# A1:
#
# Q2: What is the target variable? Classification or regression?
# A2:
#
# Q3: Are there missing values? Which columns?
# A3:
#
# Q4: Are data types correct? Any columns stored as wrong type?
# A4:
#
# Q5: Are there duplicates? Should they be removed?
# A5:


# =============================================================================
# STEP 3: Exploratory Data Analysis
# =============================================================================

def plot_numeric_distributions(df, save_path="plots/feature_distributions.png"):
    """Histogram of every numeric column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        df[col].hist(bins=30, ax=axes[i], edgecolor="black")
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_matrix(df, save_path="plots/correlation_matrix.png"):
    """Heatmap of pairwise correlations for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_target_distribution(df, target_col, save_path="plots/target_distribution.png"):
    """Distribution of the target variable."""
    plt.figure(figsize=(8, 5))
    if df[target_col].nunique() <= 10:
        df[target_col].value_counts().plot(kind="bar", edgecolor="black")
        plt.ylabel("Count")
    else:
        df[target_col].hist(bins=30, edgecolor="black")
        plt.ylabel("Frequency")
    plt.title(f"Distribution of {target_col}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_features_vs_target(df, target_col, top_n=6,
                            save_path="plots/features_vs_target.png"):
    """Scatter or box plots of top correlated features vs target."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = (
        df[numeric_cols].corr()[target_col]
        .drop(target_col, errors="ignore")
        .abs()
        .sort_values(ascending=False)
    )
    top_features = correlations.head(top_n).index

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        if df[target_col].nunique() <= 10:
            # Classification: box plot
            df.boxplot(column=feat, by=target_col, ax=axes[i])
        else:
            # Regression: scatter plot
            axes[i].scatter(df[feat], df[target_col], alpha=0.3, s=10)
        axes[i].set_title(feat)
        axes[i].set_xlabel(feat)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Top {top_n} Features vs {target_col}", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# Uncomment when ready:
# plot_numeric_distributions(df)
# plot_correlation_matrix(df)
# plot_target_distribution(df, TARGET_COL)
# plot_features_vs_target(df, TARGET_COL)


# =============================================================================
# STEP 4: Data Cleaning
# =============================================================================

def handle_missing_values(df):
    """
    Handle missing values. Customize this function for your dataset.

    Returns the cleaned DataFrame and fitted imputers.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Impute numeric columns with median
    num_imputer = SimpleImputer(strategy="median")
    if numeric_cols:
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Impute categorical columns with most frequent value
    cat_imputer = SimpleImputer(strategy="most_frequent")
    if cat_cols:
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df, num_imputer


def handle_outliers(df, columns, multiplier=1.5):
    """
    Clip outliers using IQR method.
    Modify this if you want to remove instead of clip.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            print(f"  {col}: clipping {n_outliers} outliers to [{lower:.2f}, {upper:.2f}]")
            df[col] = df[col].clip(lower, upper)

    return df


def encode_categoricals(df):
    """
    Encode categorical columns. Customize for your dataset.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if cat_cols:
        print(f"One-hot encoding: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


# Uncomment and customize:
# df_clean = df.copy()
# df_clean, num_imputer = handle_missing_values(df_clean)
#
# numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
# numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
# df_clean = handle_outliers(df_clean, numeric_cols)
#
# df_clean = encode_categoricals(df_clean)


# =============================================================================
# STEP 5: Feature Engineering
# =============================================================================

def engineer_features(df):
    """
    Create new features. These are examples — replace with features
    that make sense for YOUR dataset.
    """
    # Example: interaction feature
    # df["age_x_chol"] = df["age"] * df["chol"]

    # Example: ratio feature
    # df["rooms_per_household"] = df["total_rooms"] / df["households"]

    # Example: log transform for skewed features
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    # for col in numeric_cols:
    #     if df[col].skew() > 1.0 and df[col].min() >= 0:
    #         df[f"{col}_log"] = np.log1p(df[col])

    print("Feature engineering complete.")
    return df


# Uncomment:
# df_clean = engineer_features(df_clean)


# =============================================================================
# STEP 6: Feature Scaling
# =============================================================================

def scale_features(df, target_col):
    """Scale all features except the target."""
    feature_cols = [c for c in df.columns if c != target_col]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler


# Uncomment:
# df_clean, scaler = scale_features(df_clean, TARGET_COL)


# =============================================================================
# STEP 7: Statistical Analysis
# =============================================================================

def run_statistical_tests(df, target_col):
    """
    Run basic statistical tests to quantify feature-target relationships.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != target_col]

    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    if df[target_col].nunique() <= 10:
        # Classification: t-tests between classes
        print("\nT-tests (feature differences between classes):")
        classes = df[target_col].unique()
        for feat in numeric_cols:
            groups = [df[df[target_col] == c][feat] for c in classes]
            if len(groups) == 2:
                t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  {feat:30s}: t={t_stat:7.3f}, p={p_val:.4f} {sig}")
    else:
        # Regression: Pearson correlations
        print("\nPearson correlations with target:")
        for feat in numeric_cols:
            r, p = stats.pearsonr(df[feat], df[target_col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat:30s}: r={r:7.3f}, p={p:.4f} {sig}")


# Uncomment:
# run_statistical_tests(df_clean, TARGET_COL)


# =============================================================================
# STEP 8: Train / Validation / Test Split
# =============================================================================

def split_and_save(df, target_col, random_state=42):
    """
    Split into 60/20/20 train/val/test and save to disk.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Determine if stratified split is appropriate
    stratify = y if y.nunique() <= 10 else None

    # 80/20 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )

    # Split the 80% into 75/25 => 60/20 overall
    stratify_temp = y_temp if stratify is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state,
        stratify=stratify_temp
    )

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")

    # Save to CSV
    for name, data in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
                        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        path = f"data/processed/{name}.csv"
        data.to_csv(path, index=False)
        print(f"  Saved: {path}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# Uncomment:
# X_train, X_val, X_test, y_train, y_val, y_test = split_and_save(df_clean, TARGET_COL)


# =============================================================================
# STEP 9: Save Preprocessing Objects
# =============================================================================

def save_preprocessors(scaler, num_imputer):
    """Save fitted preprocessing objects for use in serving."""
    joblib.dump(scaler, "data/processed/scaler.joblib")
    joblib.dump(num_imputer, "data/processed/num_imputer.joblib")
    print("Saved: scaler.joblib, num_imputer.joblib")


# Uncomment:
# save_preprocessors(scaler, num_imputer)


# =============================================================================
# DONE — Review the Deliverables Checklist in the lesson before moving on.
# =============================================================================
print("\nStarter script loaded. Uncomment sections and run step by step.")

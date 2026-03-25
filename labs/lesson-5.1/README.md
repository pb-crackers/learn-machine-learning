# Lab 5.1: Data Pipeline & Exploration

## Objective

Build a complete data pipeline for your capstone project: load a dataset, explore it thoroughly,
clean and preprocess it, engineer features, and split it into train/validation/test sets.

## Getting Started

1. Choose a dataset from the suggested list below (or bring your own).
2. Open `starter.py` and follow the numbered steps.
3. Create a `plots/` directory to save your visualizations.
4. Create a `data/processed/` directory to save your cleaned splits.

## Suggested Datasets

| Dataset | Task | How to Get It |
|---------|------|---------------|
| Heart Disease (UCI) | Binary classification | `wget https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data` or use [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) |
| California Housing | Regression | `from sklearn.datasets import fetch_california_housing` |
| Wine Quality | Classification/Regression | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| Spaceship Titanic | Binary classification | [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic) |
| Online Shoppers Intention | Binary classification | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) |
| Bank Marketing | Binary classification | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) |

## What to Deliver

By the end of this lab you should have:

- Written answers to all 5 inspection questions (in comments or markdown)
- At least 5 saved plots in `plots/` (distributions, correlations, feature vs target, etc.)
- A documented cleaning strategy for each column with missing data
- At least 2 engineered features
- Saved CSV files in `data/processed/`: `X_train.csv`, `X_val.csv`, `X_test.csv`, `y_train.csv`, `y_val.csv`, `y_test.csv`
- Saved preprocessing objects: `scaler.joblib`, `num_imputer.joblib`

## Tips

- Do not rush. Time spent here saves debugging time in Phase 2.
- Write comments explaining *why* you made each cleaning decision, not just what you did.
- If a feature has more than 5% missing values, think carefully before dropping rows.
- Always check your splits for data leakage: no sample should appear in more than one set.

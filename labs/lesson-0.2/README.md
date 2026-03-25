# Lab 0.2: Pandas, Matplotlib & the ML Data Pipeline

## Objectives

By completing this lab you will:

- Load and explore a real-world dataset using Pandas
- Handle missing values and encode categorical features
- Create informative visualizations with Matplotlib and Seaborn
- Prepare clean feature matrices and target vectors for ML modeling

## Setup

```bash
pip install pandas matplotlib seaborn scikit-learn
```

We use scikit-learn solely for its built-in datasets (no modeling in this lab).

## Instructions

Open `starter.py` and complete each function marked with `TODO`. The lab uses the California Housing dataset, which is bundled with scikit-learn so no file downloads are needed.

### Exercise 1: Load and Explore

Use Pandas to load the dataset into a DataFrame and answer basic questions about its shape, types, and summary statistics.

### Exercise 2: Clean the Data

Handle missing values (we will artificially inject some), remove outliers, and create new features.

### Exercise 3: Visualize

Create four plots:
1. Histogram of the target variable (median house value)
2. Correlation heatmap of all numeric features
3. Scatter plot of median income vs. median house value
4. Box plot of house value grouped by an ocean proximity category

### Exercise 4: Prepare for Modeling

Separate features and target, apply standardization using only training set statistics, and output clean NumPy arrays.

## Running

```bash
python starter.py
```

The script will create a `plots/` directory with your saved visualizations and print test results to the console.

## Stretch Goals

- Add a pair plot for the 4 most correlated features
- Create a geographic scatter plot using latitude and longitude
- Experiment with different missing-value strategies and compare the resulting feature distributions

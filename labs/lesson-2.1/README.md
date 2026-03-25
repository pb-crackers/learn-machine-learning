# Lab 2.1: Complete Toy ML Pipeline with Scikit-Learn

## Objective

Build a complete machine learning pipeline from scratch using scikit-learn. You will load a dataset, explore it, preprocess features, train multiple models, evaluate with cross-validation, and select the best model.

## Tasks

### Part 1: Data Loading and Exploration
1. Load the California Housing dataset using `sklearn.datasets.fetch_california_housing()`
2. Print the shape, feature names, and basic statistics
3. Check for missing values
4. Create a correlation heatmap using matplotlib

### Part 2: Data Splitting
1. Split into training (60%), validation (20%), and test (20%) sets
2. Use `stratify` or fixed `random_state` for reproducibility
3. Print the size of each set to verify

### Part 3: Preprocessing Pipeline
1. Build a `sklearn.pipeline.Pipeline` with `StandardScaler` and a model
2. Ensure the scaler is fit only on training data

### Part 4: Model Comparison
1. Train at least 3 models: `LinearRegression`, `Ridge`, `DecisionTreeRegressor`
2. Evaluate each with 5-fold cross-validation on the training set
3. Print mean and std of R-squared scores

### Part 5: Final Evaluation
1. Select the best model based on CV scores
2. Evaluate on the test set (only once!)
3. Print test R-squared and RMSE

## Getting Started

```bash
pip install scikit-learn numpy matplotlib
python starter.py
```

## Expected Output

Your script should print a comparison table of models and their CV scores, followed by the final test set performance of the best model.

## Stretch Goals

- Add `RandomForestRegressor` to the comparison
- Use `GridSearchCV` to tune the best model's hyperparameters
- Plot predicted vs. actual values for the test set

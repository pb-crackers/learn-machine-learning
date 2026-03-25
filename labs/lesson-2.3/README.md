# Lab 2.3: Build a Classifier and Compute Evaluation Metrics

## Objective

Train a logistic regression classifier on a real dataset, compute all major evaluation metrics by hand, and understand the precision-recall tradeoff by varying the classification threshold.

## Tasks

### Part 1: Load and Explore the Data
1. Load the Breast Cancer Wisconsin dataset (`sklearn.datasets.load_breast_cancer`)
2. Explore the class distribution — is it balanced?
3. Split into train/test sets (80/20) with stratification

### Part 2: Train a Logistic Regression Classifier
1. Build a pipeline with `StandardScaler` and `LogisticRegression`
2. Train on the training set
3. Generate predictions and predicted probabilities

### Part 3: Compute Metrics Manually
1. Build the confusion matrix by hand (count TP, FP, TN, FN)
2. Compute accuracy, precision, recall, and F1 from your counts
3. Verify your calculations match `sklearn.metrics` functions

### Part 4: Threshold Analysis
1. Vary the classification threshold from 0.0 to 1.0
2. Plot precision and recall vs. threshold
3. Find the threshold that maximizes F1 score

### Part 5: ROC Curve
1. Plot the ROC curve
2. Compute AUC-ROC
3. Explain what the curve tells you about the model

## Getting Started

```bash
pip install scikit-learn numpy matplotlib
python starter.py
```

## Expected Output

- Confusion matrix with labeled axes
- Manually computed metrics matching sklearn output
- Precision-recall vs. threshold plot
- ROC curve with AUC annotation

## Stretch Goals

- Compare logistic regression with KNN (K=3, K=7, K=15)
- Try the model on an imbalanced dataset (use `make_classification` with `weights=[0.95, 0.05]`)
- Implement log loss from scratch and verify against `sklearn.metrics.log_loss`

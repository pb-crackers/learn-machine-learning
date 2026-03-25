# Lab 2.4: Decision Trees vs. Random Forests

## Objective

Train a decision tree and a random forest on a classification task, visualize the decision tree, compare their performance, and understand the impact of key hyperparameters.

## Tasks

### Part 1: Load and Prepare Data
1. Load the Wine dataset (`sklearn.datasets.load_wine`)
2. Split into train/test sets (80/20, stratified)
3. Explore feature distributions

### Part 2: Train and Visualize a Decision Tree
1. Train a `DecisionTreeClassifier` with `max_depth=3`
2. Visualize the tree using `sklearn.tree.plot_tree`
3. Print feature importances

### Part 3: Overfitting Analysis
1. Train trees with `max_depth` from 1 to 20
2. Record training accuracy and cross-validation accuracy for each
3. Plot both curves to find the sweet spot

### Part 4: Random Forest
1. Train a `RandomForestClassifier` with 100 trees
2. Compare its CV accuracy with the best single tree
3. Plot feature importances for the random forest

### Part 5: Ensemble Comparison
1. Compare: Decision Tree, Random Forest, AdaBoost, GradientBoosting
2. Use 5-fold cross-validation for each
3. Print a summary table of results

## Getting Started

```bash
pip install scikit-learn numpy matplotlib
python starter.py
```

## Expected Output

- A visual decision tree diagram
- Overfitting curve (train vs. CV accuracy vs. max_depth)
- Feature importance bar chart
- Ensemble comparison table

## Stretch Goals

- Try XGBoost (`pip install xgboost`) and add it to the comparison
- Visualize decision boundaries for 2 features using `contourf`
- Use `GridSearchCV` to find optimal Random Forest hyperparameters

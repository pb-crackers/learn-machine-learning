# Lab 2.2: Implement Linear Regression from Scratch

## Objective

Implement linear regression using gradient descent from scratch in NumPy, then compare your implementation against scikit-learn's `LinearRegression`.

## Tasks

### Part 1: Generate Synthetic Data
1. Create a dataset with a known linear relationship: `y = 3*x1 + 5*x2 - 2 + noise`
2. Visualize the data

### Part 2: Implement Cost Function
1. Implement the MSE cost function: `J = (1/2m) * sum((X @ theta - y)^2)`
2. Test it with random parameters

### Part 3: Implement Gradient Descent
1. Compute the gradient: `grad = (1/m) * X.T @ (X @ theta - y)`
2. Implement the update loop
3. Track cost history across iterations

### Part 4: Feature Scaling Experiment
1. Run gradient descent WITHOUT feature scaling
2. Run gradient descent WITH StandardScaler
3. Compare convergence speed by plotting cost history

### Part 5: Compare with Scikit-Learn
1. Train `sklearn.linear_model.LinearRegression`
2. Compare learned parameters (theta) with your implementation
3. Compare predictions on a test set

### Part 6: Normal Equation
1. Implement the closed-form solution: `theta = (X.T @ X)^(-1) @ X.T @ y`
2. Verify it matches the other methods

## Getting Started

```bash
pip install numpy matplotlib scikit-learn
python starter.py
```

## Expected Output

- A convergence plot showing cost decreasing over iterations
- Side-by-side comparison of parameters from scratch vs. sklearn
- Demonstration that feature scaling dramatically speeds up convergence

## Stretch Goals

- Implement mini-batch gradient descent
- Add L2 regularization (Ridge) to your gradient descent
- Experiment with different learning rates and plot the results

"""
Lab 2.2: Implement Linear Regression from Scratch

Build linear regression with gradient descent in pure NumPy,
then compare your results against scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ================================================================
# PART 1: Generate Synthetic Data
# ================================================================

def generate_data(m=200, random_state=42):
    """Generate synthetic linear data: y = 3*x1 + 5*x2 - 2 + noise."""
    rng = np.random.RandomState(random_state)

    x1 = rng.uniform(0, 10, m)
    x2 = rng.uniform(0, 5, m)
    noise = rng.randn(m) * 2.0

    # True relationship
    y = 3 * x1 + 5 * x2 - 2 + noise

    X = np.column_stack([x1, x2])
    return X, y


# ================================================================
# PART 2: Cost Function
# ================================================================

def compute_cost(X, y, theta):
    """
    Compute the MSE cost function.

    Parameters:
        X: feature matrix with bias column, shape (m, n+1)
        y: target vector, shape (m,)
        theta: parameter vector, shape (n+1,)

    Returns:
        cost: scalar value of J(theta)
    """
    m = len(y)
    # TODO: Implement the cost function
    # predictions = X @ theta
    # errors = predictions - y
    # cost = (1 / (2 * m)) * np.sum(errors ** 2)
    # return cost
    return 0.0  # Replace with your implementation


# ================================================================
# PART 3: Gradient Descent
# ================================================================

def compute_gradient(X, y, theta):
    """
    Compute the gradient of the cost function.

    Parameters:
        X: feature matrix with bias column, shape (m, n+1)
        y: target vector, shape (m,)
        theta: parameter vector, shape (n+1,)

    Returns:
        gradient: vector of shape (n+1,)
    """
    m = len(y)
    # TODO: Implement the gradient computation
    # gradient = (1 / m) * (X.T @ (X @ theta - y))
    # return gradient
    return np.zeros_like(theta)  # Replace with your implementation


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Run gradient descent to optimize theta.

    Parameters:
        X: feature matrix with bias column
        y: target vector
        theta: initial parameters
        alpha: learning rate
        num_iters: number of iterations

    Returns:
        theta: optimized parameters
        cost_history: list of cost values at each iteration
    """
    cost_history = []

    for i in range(num_iters):
        # TODO: Compute gradient and update theta
        # gradient = compute_gradient(X, y, theta)
        # theta = theta - alpha * gradient
        # cost_history.append(compute_cost(X, y, theta))
        pass  # Replace with your implementation

    return theta, cost_history


# ================================================================
# PART 4: Feature Scaling Experiment
# ================================================================

def scaling_experiment(X_raw, y):
    """Compare convergence with and without feature scaling."""
    print("\n" + "=" * 50)
    print("PART 4: Feature Scaling Experiment")
    print("=" * 50)

    # Without scaling
    X_no_scale = np.column_stack([np.ones(len(y)), X_raw])
    theta_init = np.zeros(X_no_scale.shape[1])

    # TODO: Run gradient descent without scaling
    # Use a very small learning rate (e.g., 1e-4) since features are not scaled
    # theta_no_scale, history_no_scale = gradient_descent(
    #     X_no_scale, y, theta_init.copy(), alpha=1e-4, num_iters=1000
    # )

    # With scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_with_scale = np.column_stack([np.ones(len(y)), X_scaled])
    theta_init_scaled = np.zeros(X_with_scale.shape[1])

    # TODO: Run gradient descent with scaling
    # Use a larger learning rate (e.g., 0.1) since features are scaled
    # theta_scaled, history_scaled = gradient_descent(
    #     X_with_scale, y, theta_init_scaled.copy(), alpha=0.1, num_iters=1000
    # )

    # TODO: Plot both convergence curves side by side
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # axes[0].plot(history_no_scale)
    # axes[0].set_title("Without Scaling")
    # axes[0].set_xlabel("Iteration")
    # axes[0].set_ylabel("Cost")
    # axes[1].plot(history_scaled)
    # axes[1].set_title("With Scaling")
    # axes[1].set_xlabel("Iteration")
    # axes[1].set_ylabel("Cost")
    # plt.tight_layout()
    # plt.show()

    pass  # Remove after implementing


# ================================================================
# PART 5: Compare with Scikit-Learn
# ================================================================

def compare_with_sklearn(X_raw, y):
    """Compare your implementation against scikit-learn."""
    print("\n" + "=" * 50)
    print("PART 5: Comparison with Scikit-Learn")
    print("=" * 50)

    # Your implementation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_with_bias = np.column_stack([np.ones(len(y)), X_scaled])
    theta_init = np.zeros(X_with_bias.shape[1])

    # TODO: Run your gradient descent
    # theta_custom, _ = gradient_descent(X_with_bias, y, theta_init, alpha=0.1, num_iters=5000)
    # print(f"Your theta:   {theta_custom.round(4)}")

    # Scikit-learn
    # TODO: Train sklearn LinearRegression on the SAME scaled data
    # model = LinearRegression()
    # model.fit(X_scaled, y)
    # theta_sklearn = np.array([model.intercept_, *model.coef_])
    # print(f"Sklearn theta: {theta_sklearn.round(4)}")

    # TODO: Compare predictions
    # y_pred_custom = X_with_bias @ theta_custom
    # y_pred_sklearn = model.predict(X_scaled)
    # print(f"\nMax prediction difference: {np.max(np.abs(y_pred_custom - y_pred_sklearn)):.6f}")

    pass  # Remove after implementing


# ================================================================
# PART 6: Normal Equation
# ================================================================

def normal_equation(X, y):
    """
    Solve for optimal theta using the normal equation.

    theta* = (X^T X)^(-1) X^T y
    """
    # TODO: Implement the normal equation
    # return np.linalg.pinv(X.T @ X) @ (X.T @ y)
    return np.zeros(X.shape[1])  # Replace with your implementation


def verify_normal_equation(X_raw, y):
    """Verify normal equation matches other methods."""
    print("\n" + "=" * 50)
    print("PART 6: Normal Equation Verification")
    print("=" * 50)

    X_with_bias = np.column_stack([np.ones(len(y)), X_raw])

    # TODO: Compute theta using the normal equation
    # theta_ne = normal_equation(X_with_bias, y)
    # print(f"Normal equation theta: {theta_ne.round(4)}")
    # print(f"Expected: intercept ~ -2, coef1 ~ 3, coef2 ~ 5")

    pass  # Remove after implementing


# ================================================================
# MAIN
# ================================================================

def main():
    print("Lab 2.2: Linear Regression from Scratch")
    print("=" * 50)

    # Part 1: Generate data
    X_raw, y = generate_data()
    print(f"Generated {X_raw.shape[0]} samples with {X_raw.shape[1]} features")
    print(f"True relationship: y = 3*x1 + 5*x2 - 2 + noise")

    # Part 2: Test cost function
    print("\n" + "=" * 50)
    print("PART 2: Cost Function Test")
    print("=" * 50)
    X_test = np.column_stack([np.ones(len(y)), X_raw])
    theta_test = np.zeros(3)
    cost = compute_cost(X_test, y, theta_test)
    print(f"Cost with theta=[0,0,0]: {cost:.4f}")

    # Part 3: Gradient descent
    print("\n" + "=" * 50)
    print("PART 3: Gradient Descent")
    print("=" * 50)
    # Uncomment and fill in after implementing gradient_descent
    # theta_gd, history = gradient_descent(X_test, y, theta_test, alpha=1e-4, num_iters=5000)
    # print(f"Learned theta: {theta_gd.round(4)}")
    # plt.plot(history)
    # plt.xlabel("Iteration")
    # plt.ylabel("Cost")
    # plt.title("Gradient Descent Convergence")
    # plt.show()

    # Part 4: Feature scaling
    scaling_experiment(X_raw, y)

    # Part 5: Compare with sklearn
    compare_with_sklearn(X_raw, y)

    # Part 6: Normal equation
    verify_normal_equation(X_raw, y)

    print("\n" + "=" * 50)
    print("Lab complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

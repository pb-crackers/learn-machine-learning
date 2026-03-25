"""
Lab 1.3: Calculus Intuition (Gradients & Optimization)
=======================================================
Complete the exercises below. Each function has instructions in its docstring.
Run this file to execute all exercises: python starter.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Exercise 1: Gradient Descent from Scratch
# ============================================================================

def exercise_1():
    """
    Implement gradient descent for linear regression.

    Tasks:
    1. Generate 100 data points from y = 2.5x + 8 + noise
    2. Initialize w=0, b=0
    3. Implement the MSE loss function
    4. Implement the gradient computation:
       - dL/dw = (-2/n) * sum(x_i * (y_i - y_hat_i))
       - dL/db = (-2/n) * sum(y_i - y_hat_i)
    5. Run gradient descent for 500 iterations with lr=0.01
    6. Plot: (a) the data + learned line, (b) the loss curve over iterations
    """
    np.random.seed(42)

    # TODO: Generate data
    # n = 100
    # X = np.random.uniform(0, 10, n)
    # y = 2.5 * X + 8 + np.random.normal(0, 2, n)

    # TODO: Initialize parameters
    # w = 0.0
    # b = 0.0
    # lr = 0.01
    # n_iterations = 500
    # losses = []

    # TODO: Gradient descent loop
    # for i in range(n_iterations):
    #     # Forward pass: predictions
    #     y_pred = w * X + b
    #
    #     # Compute MSE loss
    #     loss = np.mean((y - y_pred) ** 2)
    #     losses.append(loss)
    #
    #     # Compute gradients
    #     dw = (-2 / n) * np.sum(X * (y - y_pred))
    #     db = (-2 / n) * np.sum(y - y_pred)
    #
    #     # Update parameters
    #     w -= lr * dw
    #     b -= lr * db
    #
    #     if i % 100 == 0:
    #         print(f"Iteration {i:4d}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

    # TODO: Plot results
    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    #
    # # Plot 1: Data + learned line
    # axes[0].scatter(X, y, alpha=0.5, s=20, label='Data')
    # x_line = np.linspace(0, 10, 100)
    # axes[0].plot(x_line, w * x_line + b, 'r-', linewidth=2,
    #              label=f'Learned: y = {w:.2f}x + {b:.2f}')
    # axes[0].plot(x_line, 2.5 * x_line + 8, 'g--', linewidth=1,
    #              label='True: y = 2.5x + 8')
    # axes[0].set_xlabel('x')
    # axes[0].set_ylabel('y')
    # axes[0].set_title('Linear Regression via Gradient Descent')
    # axes[0].legend()
    # axes[0].grid(True, alpha=0.3)
    #
    # # Plot 2: Loss curve
    # axes[1].plot(losses, color='steelblue')
    # axes[1].set_xlabel('Iteration')
    # axes[1].set_ylabel('MSE Loss')
    # axes[1].set_title('Loss Curve')
    # axes[1].grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.show()

    print("Exercise 1: Complete the TODOs above\n")


# ============================================================================
# Exercise 2: Learning Rate Experiment
# ============================================================================

def exercise_2():
    """
    Compare gradient descent with different learning rates.

    Tasks:
    1. Use the same data as Exercise 1
    2. Run gradient descent with lr = [0.0001, 0.001, 0.01, 0.1, 1.0]
    3. For each, record the loss at every step (use 200 steps)
    4. Plot ALL loss curves on the same chart
    5. Note which learning rates converge, which are slow, and which diverge
    """
    np.random.seed(42)

    # TODO: Generate data (same as exercise 1)
    # n = 100
    # X = np.random.uniform(0, 10, n)
    # y = 2.5 * X + 8 + np.random.normal(0, 2, n)

    # TODO: Run GD with different learning rates
    # learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
    # all_losses = {}
    #
    # for lr in learning_rates:
    #     w, b = 0.0, 0.0
    #     losses = []
    #
    #     for i in range(200):
    #         y_pred = w * X + b
    #         loss = np.mean((y - y_pred) ** 2)
    #         losses.append(loss)
    #
    #         dw = (-2 / n) * np.sum(X * (y - y_pred))
    #         db = (-2 / n) * np.sum(y - y_pred)
    #
    #         w -= lr * dw
    #         b -= lr * db
    #
    #         # Clip to prevent overflow for large lr
    #         if loss > 1e10:
    #             losses.extend([float('inf')] * (200 - i - 1))
    #             break
    #
    #     all_losses[lr] = losses

    # TODO: Plot all loss curves
    # plt.figure(figsize=(10, 6))
    # for lr, losses in all_losses.items():
    #     plt.plot(losses, label=f'lr={lr}', linewidth=2)
    # plt.xlabel('Iteration')
    # plt.ylabel('MSE Loss')
    # plt.title('Effect of Learning Rate on Convergence')
    # plt.yscale('log')  # log scale helps see differences
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.ylim(1, 1e6)
    # plt.show()

    print("Exercise 2: Complete the TODOs above\n")


# ============================================================================
# Exercise 3: Visualize the Optimization Path
# ============================================================================

def exercise_3():
    """
    Visualize gradient descent on a 2D contour plot.

    Tasks:
    1. Define a loss function: L(w, b) = mean((y - w*X - b)^2)
    2. Create a grid of (w, b) values and compute L at each point
    3. Plot as a contour plot
    4. Run gradient descent from 3 different starting points
    5. Overlay all 3 paths on the contour plot
    """
    np.random.seed(42)

    # TODO: Generate simple data
    # n = 50
    # X = np.random.uniform(0, 5, n)
    # y = 2.0 * X + 3.0 + np.random.normal(0, 1, n)

    # TODO: Define loss function over (w, b) grid
    # def compute_loss(w_val, b_val):
    #     return np.mean((y - w_val * X - b_val) ** 2)

    # TODO: Create grid
    # w_range = np.linspace(-2, 6, 100)
    # b_range = np.linspace(-2, 8, 100)
    # W_grid, B_grid = np.meshgrid(w_range, b_range)
    # L_grid = np.zeros_like(W_grid)
    # for i in range(100):
    #     for j in range(100):
    #         L_grid[i, j] = compute_loss(W_grid[i, j], B_grid[i, j])

    # TODO: Run gradient descent from different starting points
    # starts = [np.array([-1.0, -1.0]), np.array([5.0, 7.0]), np.array([0.0, 6.0])]
    # colors = ['red', 'blue', 'green']
    # paths = []
    #
    # for start in starts:
    #     w, b = start[0], start[1]
    #     path = [(w, b)]
    #     lr = 0.01
    #
    #     for _ in range(200):
    #         y_pred = w * X + b
    #         dw = (-2 / n) * np.sum(X * (y - y_pred))
    #         db = (-2 / n) * np.sum(y - y_pred)
    #         w -= lr * dw
    #         b -= lr * db
    #         path.append((w, b))
    #
    #     paths.append(np.array(path))

    # TODO: Plot contour + paths
    # plt.figure(figsize=(10, 8))
    # plt.contour(W_grid, B_grid, L_grid, levels=30, cmap='viridis')
    # plt.colorbar(label='MSE Loss')
    #
    # for path, color, start in zip(paths, colors, starts):
    #     plt.plot(path[:, 0], path[:, 1], f'{color[0]}o-', markersize=2,
    #              linewidth=1, alpha=0.7)
    #     plt.plot(path[0, 0], path[0, 1], f'{color[0]}s', markersize=10,
    #              label=f'Start ({start[0]:.0f}, {start[1]:.0f})')
    #     plt.plot(path[-1, 0], path[-1, 1], f'{color[0]}*', markersize=15)
    #
    # plt.xlabel('w (weight)')
    # plt.ylabel('b (bias)')
    # plt.title('Gradient Descent Paths on Loss Surface')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    print("Exercise 3: Complete the TODOs above\n")


# ============================================================================
# Exercise 4: Mini-Batch vs. Batch Gradient Descent
# ============================================================================

def exercise_4():
    """
    Compare full-batch and mini-batch gradient descent.

    Tasks:
    1. Generate a larger dataset (1000 points)
    2. Implement full-batch GD (use all data per step)
    3. Implement mini-batch GD (use batches of 32)
    4. Run both for the same number of EPOCHS (passes through data)
    5. Plot both loss curves and compare:
       - Which has smoother loss?
       - Which converges faster in wall-clock time?
    """
    np.random.seed(42)

    # TODO: Generate larger dataset
    # n = 1000
    # X = np.random.uniform(0, 10, n)
    # y = 2.5 * X + 8 + np.random.normal(0, 3, n)
    # n_epochs = 50

    # TODO: Full-batch gradient descent
    # w_full, b_full = 0.0, 0.0
    # lr = 0.001
    # losses_full = []
    #
    # for epoch in range(n_epochs):
    #     y_pred = w_full * X + b_full
    #     loss = np.mean((y - y_pred) ** 2)
    #     losses_full.append(loss)
    #
    #     dw = (-2 / n) * np.sum(X * (y - y_pred))
    #     db = (-2 / n) * np.sum(y - y_pred)
    #     w_full -= lr * dw
    #     b_full -= lr * db

    # TODO: Mini-batch gradient descent
    # w_mb, b_mb = 0.0, 0.0
    # batch_size = 32
    # losses_mb = []
    #
    # for epoch in range(n_epochs):
    #     indices = np.random.permutation(n)
    #     epoch_loss = 0
    #
    #     for start in range(0, n, batch_size):
    #         batch_idx = indices[start:start + batch_size]
    #         X_batch = X[batch_idx]
    #         y_batch = y[batch_idx]
    #         bs = len(X_batch)
    #
    #         y_pred = w_mb * X_batch + b_mb
    #         epoch_loss += np.sum((y_batch - y_pred) ** 2)
    #
    #         dw = (-2 / bs) * np.sum(X_batch * (y_batch - y_pred))
    #         db = (-2 / bs) * np.sum(y_batch - y_pred)
    #         w_mb -= lr * dw
    #         b_mb -= lr * db
    #
    #     losses_mb.append(epoch_loss / n)

    # TODO: Plot comparison
    # plt.figure(figsize=(10, 5))
    # plt.plot(losses_full, 'b-', linewidth=2, label=f'Full Batch (w={w_full:.2f}, b={b_full:.2f})')
    # plt.plot(losses_mb, 'r-', linewidth=2, alpha=0.7, label=f'Mini-Batch (w={w_mb:.2f}, b={b_mb:.2f})')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE Loss')
    # plt.title('Full-Batch vs Mini-Batch Gradient Descent')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()
    #
    # print(f"Full batch final:  w={w_full:.4f}, b={b_full:.4f}")
    # print(f"Mini-batch final:  w={w_mb:.4f}, b={b_mb:.4f}")
    # print(f"True values:       w=2.5000, b=8.0000")

    print("Exercise 4: Complete the TODOs above\n")


# ============================================================================
# Run all exercises
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lab 1.3: Calculus Intuition (Gradients & Optimization)")
    print("=" * 60)
    print()

    print("--- Exercise 1: Gradient Descent from Scratch ---")
    exercise_1()

    print("--- Exercise 2: Learning Rate Experiment ---")
    exercise_2()

    print("--- Exercise 3: Optimization Path Visualization ---")
    exercise_3()

    print("--- Exercise 4: Mini-Batch vs. Batch ---")
    exercise_4()

    print("Done! Uncomment the TODO sections to complete each exercise.")

"""
Lab 1.1: Linear Algebra for ML
===============================
Complete the exercises below. Each function has instructions in its docstring.
Run this file to execute all exercises: python starter.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time


# ============================================================================
# Exercise 1: Matrix Operations and Batch Predictions
# ============================================================================

def exercise_1():
    """
    Compare loop-based vs matrix-based predictions.

    Tasks:
    1. Create a random data matrix X with shape (1000, 5)
    2. Create a random weight vector w with shape (5,)
    3. Compute predictions using a Python for-loop (dot product per row)
    4. Compute predictions using matrix multiplication (X @ w)
    5. Verify both give the same result (use np.allclose)
    6. Time both approaches and print the speedup factor
    """
    np.random.seed(42)

    # TODO: Create X (1000 samples, 5 features) and w (5 weights)
    X = None  # Replace with your code
    w = None  # Replace with your code

    # TODO: Loop-based predictions
    # Time the loop approach
    start = time.time()
    predictions_loop = np.zeros(1000)
    # for i in range(len(X)):
    #     predictions_loop[i] = ...  # dot product of row i with w
    loop_time = time.time() - start

    # TODO: Matrix multiplication predictions
    start = time.time()
    predictions_matrix = None  # Replace with X @ w
    matrix_time = time.time() - start

    # TODO: Verify they match
    # print(f"Results match: {np.allclose(predictions_loop, predictions_matrix)}")

    # TODO: Print timing comparison
    # print(f"Loop time: {loop_time:.6f}s")
    # print(f"Matrix time: {matrix_time:.6f}s")
    # print(f"Speedup: {loop_time / matrix_time:.1f}x")

    print("Exercise 1: Complete the TODOs above\n")


# ============================================================================
# Exercise 2: 2D Transformations
# ============================================================================

def exercise_2():
    """
    Visualize how matrices transform 2D shapes.

    Tasks:
    1. Define a shape as a set of 2D points (an arrow or house shape)
    2. Create a rotation matrix for 30 degrees
    3. Create a scaling matrix that scales x by 2 and y by 0.5
    4. Create a shear matrix with shear factor 0.5 in x
    5. Apply each transformation and plot original + transformed side by side
    """
    # A simple arrow shape (as columns: each column is a point)
    # Connect points in order to draw the shape
    arrow = np.array([
        [0, 1, 0.7, 1, 0.7, 1, 0],   # x coordinates
        [0, 0.5, 0.3, 0.5, 0.7, 0.5, 1]  # y coordinates
    ])

    # TODO: Create rotation matrix for 30 degrees
    # theta = np.radians(30)
    # R = np.array([[np.cos(theta), -np.sin(theta)],
    #               [np.sin(theta),  np.cos(theta)]])

    # TODO: Create scaling matrix (2x horizontal, 0.5x vertical)
    # S = np.array([[2, 0],
    #               [0, 0.5]])

    # TODO: Create shear matrix (shear x by 0.5)
    # Sh = np.array([[1, 0.5],
    #                [0, 1]])

    # TODO: Apply each transformation: transformed = matrix @ arrow

    # TODO: Plot original and all three transformations in a 2x2 grid
    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # For each subplot:
    #   ax.plot(points[0], points[1], 'b-o')  or 'r-o' for transformed
    #   ax.set_aspect('equal')
    #   ax.grid(True)
    #   ax.set_title(...)

    print("Exercise 2: Complete the TODOs above\n")


# ============================================================================
# Exercise 3: Cosine Similarity
# ============================================================================

def exercise_3():
    """
    Compute pairwise cosine similarity between document vectors.

    Tasks:
    1. Create 5 "document vectors" (e.g., simulated word counts, length 10)
       - Make docs 0 and 1 similar, docs 2 and 3 similar, doc 4 different
    2. Write a function to compute cosine similarity between two vectors
    3. Build a 5x5 similarity matrix
    4. Display it as a heatmap using plt.imshow
    """
    np.random.seed(42)

    # TODO: Create 5 document vectors
    # Hint: Make some similar by starting from the same base + small noise
    # doc0 = np.array([5, 3, 0, 0, 1, 0, 2, 0, 1, 0])
    # doc1 = np.array([4, 3, 0, 0, 2, 0, 1, 0, 1, 0])  # similar to doc0
    # ...

    # TODO: Implement cosine similarity
    def cosine_similarity(a, b):
        """Compute cosine similarity between vectors a and b."""
        # return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        pass

    # TODO: Build the 5x5 similarity matrix
    # docs = [doc0, doc1, doc2, doc3, doc4]
    # sim_matrix = np.zeros((5, 5))
    # for i in range(5):
    #     for j in range(5):
    #         sim_matrix[i, j] = cosine_similarity(docs[i], docs[j])

    # TODO: Plot heatmap
    # plt.figure(figsize=(6, 5))
    # plt.imshow(sim_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    # plt.colorbar(label='Cosine Similarity')
    # plt.xticks(range(5), [f'Doc {i}' for i in range(5)])
    # plt.yticks(range(5), [f'Doc {i}' for i in range(5)])
    # for i in range(5):
    #     for j in range(5):
    #         plt.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center')
    # plt.title('Document Cosine Similarity')
    # plt.tight_layout()
    # plt.show()

    print("Exercise 3: Complete the TODOs above\n")


# ============================================================================
# Exercise 4: Mini-PCA
# ============================================================================

def exercise_4():
    """
    Perform PCA from scratch on a 2D dataset.

    Tasks:
    1. Generate 200 points from a 2D distribution with a clear direction
       (e.g., correlated x and y)
    2. Center the data (subtract the mean)
    3. Compute the covariance matrix
    4. Find eigenvalues and eigenvectors
    5. Plot the data with eigenvectors overlaid (scaled by eigenvalue)
    6. Project data onto the first principal component and plot
    """
    np.random.seed(42)

    # TODO: Generate correlated 2D data
    # Hint: Use a rotation to create correlation
    # n = 200
    # uncorrelated = np.random.randn(2, n) * np.array([[3], [1]])  # stretch x more
    # theta = np.radians(30)
    # R = np.array([[np.cos(theta), -np.sin(theta)],
    #               [np.sin(theta),  np.cos(theta)]])
    # data = R @ uncorrelated  # shape (2, n)

    # TODO: Center the data
    # mean = data.mean(axis=1, keepdims=True)
    # centered = data - mean

    # TODO: Compute covariance matrix
    # cov_matrix = np.cov(centered)  # 2x2

    # TODO: Eigendecomposition
    # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # TODO: Sort by eigenvalue (largest first)
    # idx = np.argsort(eigenvalues)[::-1]
    # eigenvalues = eigenvalues[idx]
    # eigenvectors = eigenvectors[:, idx]

    # TODO: Plot data with eigenvectors
    # plt.figure(figsize=(8, 8))
    # plt.scatter(data[0], data[1], alpha=0.4, s=20)
    # origin = data.mean(axis=1)
    # for i in range(2):
    #     vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
    #     plt.arrow(origin[0], origin[1], vec[0], vec[1],
    #               head_width=0.2, color=['red', 'blue'][i], linewidth=2,
    #               label=f'PC{i+1} (λ={eigenvalues[i]:.2f})')
    # plt.axis('equal')
    # plt.legend()
    # plt.title('PCA: Principal Components')
    # plt.grid(True)
    # plt.show()

    # TODO: Project onto first PC
    # pc1 = eigenvectors[:, 0]  # first principal component direction
    # projections = pc1 @ centered  # scalar projection for each point
    # reconstructed = np.outer(pc1, projections) + mean  # back to 2D
    #
    # plt.figure(figsize=(8, 8))
    # plt.scatter(data[0], data[1], alpha=0.3, s=20, label='Original')
    # plt.scatter(reconstructed[0], reconstructed[1], alpha=0.3, s=20,
    #             color='red', label='Projected onto PC1')
    # plt.axis('equal')
    # plt.legend()
    # plt.title('Projection onto First Principal Component')
    # plt.grid(True)
    # plt.show()

    print("Exercise 4: Complete the TODOs above\n")


# ============================================================================
# Run all exercises
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lab 1.1: Linear Algebra for ML")
    print("=" * 60)
    print()

    print("--- Exercise 1: Batch Predictions ---")
    exercise_1()

    print("--- Exercise 2: 2D Transformations ---")
    exercise_2()

    print("--- Exercise 3: Cosine Similarity ---")
    exercise_3()

    print("--- Exercise 4: Mini-PCA ---")
    exercise_4()

    print("Done! Uncomment the TODO sections to complete each exercise.")

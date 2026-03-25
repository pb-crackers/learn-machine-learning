"""
Lab 0.1 — NumPy & Vectorized Thinking
======================================
Complete each function marked with TODO.
Run this file to check your solutions: python starter.py
"""

import numpy as np
import time

# ---------------------------------------------------------------------------
# Exercise 1: Loop vs. Vectorized Dot Product
# ---------------------------------------------------------------------------

def dot_product_loop(a, b):
    """Compute the dot product of two 1-D arrays using a Python for-loop.

    Args:
        a: 1-D numpy array of length n
        b: 1-D numpy array of length n

    Returns:
        A scalar (float or int) equal to sum(a[i] * b[i] for all i).

    TODO: Implement this using a for-loop (no NumPy operations on whole arrays).
    """
    # TODO: your code here
    pass


def dot_product_vectorized(a, b):
    """Compute the dot product of two 1-D arrays using NumPy (no loops).

    Args:
        a: 1-D numpy array of length n
        b: 1-D numpy array of length n

    Returns:
        A scalar equal to the dot product of a and b.

    TODO: Implement this in one line using the @ operator or np.dot.
    """
    # TODO: your code here
    pass


def benchmark_dot_product():
    """Benchmark loop vs vectorized dot product. No changes needed."""
    rng = np.random.default_rng(42)
    n = 1_000_000
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)

    start = time.perf_counter()
    result_loop = dot_product_loop(a, b)
    time_loop = time.perf_counter() - start

    start = time.perf_counter()
    result_vec = dot_product_vectorized(a, b)
    time_vec = time.perf_counter() - start

    assert result_loop is not None, "dot_product_loop returned None -- did you implement it?"
    assert result_vec is not None, "dot_product_vectorized returned None -- did you implement it?"
    assert abs(result_loop - result_vec) < 1e-6, (
        f"Results differ: loop={result_loop}, vectorized={result_vec}"
    )

    print(f"  Loop:       {time_loop:.4f}s")
    print(f"  Vectorized: {time_vec:.4f}s")
    print(f"  Speedup:    {time_loop / time_vec:.1f}x")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: Broadcasting — Normalize a Dataset
# ---------------------------------------------------------------------------

def normalize_dataset(X):
    """Normalize (standardize) each feature column to zero mean and unit variance.

    Given X of shape (n_samples, n_features), return X_normalized where each
    column has mean ≈ 0 and std ≈ 1.

    Formula: X_normalized[:, j] = (X[:, j] - mean_j) / std_j

    Args:
        X: 2-D numpy array of shape (n_samples, n_features)

    Returns:
        X_normalized: same shape as X, with each column standardized.

    TODO: Use broadcasting. Your solution should be 1-3 lines with NO loops.
    Hint: X.mean(axis=0) gives the mean of each column.
    """
    # TODO: your code here
    pass


def test_normalize():
    """Test the normalize function. No changes needed."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 4)) * np.array([10, 100, 0.5, 1000]) + np.array([5, -20, 3, 500])

    X_norm = normalize_dataset(X)

    assert X_norm is not None, "normalize_dataset returned None -- did you implement it?"
    assert X_norm.shape == X.shape, f"Shape mismatch: expected {X.shape}, got {X_norm.shape}"

    col_means = X_norm.mean(axis=0)
    col_stds = X_norm.std(axis=0)

    assert np.allclose(col_means, 0, atol=1e-10), (
        f"Column means should be ~0, got {col_means}"
    )
    assert np.allclose(col_stds, 1, atol=1e-10), (
        f"Column stds should be ~1, got {col_stds}"
    )
    print("  PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: Vectorized Loss Functions
# ---------------------------------------------------------------------------

def mean_squared_error(y_true, y_pred):
    """Compute Mean Squared Error between true and predicted values.

    MSE = (1/n) * sum((y_pred_i - y_true_i)^2)

    Args:
        y_true: 1-D numpy array of true values
        y_pred: 1-D numpy array of predicted values

    Returns:
        A scalar: the MSE.

    TODO: Implement this in one line, no loops.
    """
    # TODO: your code here
    pass


def mean_absolute_error(y_true, y_pred):
    """Compute Mean Absolute Error between true and predicted values.

    MAE = (1/n) * sum(|y_pred_i - y_true_i|)

    Args:
        y_true: 1-D numpy array of true values
        y_pred: 1-D numpy array of predicted values

    Returns:
        A scalar: the MAE.

    TODO: Implement this in one line, no loops. Hint: np.abs()
    """
    # TODO: your code here
    pass


def test_loss_functions():
    """Test MSE and MAE. No changes needed."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.3, 4.7])

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    assert mse is not None, "mean_squared_error returned None -- did you implement it?"
    assert mae is not None, "mean_absolute_error returned None -- did you implement it?"

    expected_mse = np.mean((y_pred - y_true) ** 2)
    expected_mae = np.mean(np.abs(y_pred - y_true))

    assert abs(mse - expected_mse) < 1e-10, f"MSE: expected {expected_mse}, got {mse}"
    assert abs(mae - expected_mae) < 1e-10, f"MAE: expected {expected_mae}, got {mae}"

    print(f"  MSE = {mse:.6f}  (expected {expected_mse:.6f})")
    print(f"  MAE = {mae:.6f}  (expected {expected_mae:.6f})")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Exercise 4: Reshape Challenge — Image Tensor
# ---------------------------------------------------------------------------

def reshape_to_image_tensor(flat_pixels, batch, height, width, channels):
    """Reshape a flat 1-D array of pixel values into an image tensor.

    In ML frameworks, images are typically stored as 4-D tensors with shape
    (batch, height, width, channels). For example, a batch of 2 RGB images
    of size 4x4 would have shape (2, 4, 4, 3).

    Args:
        flat_pixels: 1-D numpy array with batch*height*width*channels elements
        batch: number of images
        height: pixel height of each image
        width: pixel width of each image
        channels: number of color channels (e.g., 3 for RGB)

    Returns:
        A 4-D numpy array of shape (batch, height, width, channels)

    TODO: One line -- use reshape.
    """
    # TODO: your code here
    pass


def extract_red_channel(image_tensor):
    """Extract the red (first) channel from an image tensor.

    Args:
        image_tensor: 4-D numpy array of shape (batch, height, width, channels)

    Returns:
        3-D numpy array of shape (batch, height, width) containing only channel 0.

    TODO: One line -- use slicing/indexing.
    """
    # TODO: your code here
    pass


def test_reshape():
    """Test image tensor reshaping. No changes needed."""
    rng = np.random.default_rng(42)

    batch, height, width, channels = 2, 4, 4, 3
    n_pixels = batch * height * width * channels
    flat = rng.integers(0, 256, size=n_pixels)

    tensor = reshape_to_image_tensor(flat, batch, height, width, channels)
    assert tensor is not None, "reshape_to_image_tensor returned None -- did you implement it?"
    assert tensor.shape == (2, 4, 4, 3), f"Expected shape (2,4,4,3), got {tensor.shape}"
    print(f"  Image tensor shape: {tensor.shape}")

    red = extract_red_channel(tensor)
    assert red is not None, "extract_red_channel returned None -- did you implement it?"
    assert red.shape == (2, 4, 4), f"Expected shape (2,4,4), got {red.shape}"

    # Verify the red channel values match the first channel of the tensor
    assert np.array_equal(red, tensor[:, :, :, 0]), "Red channel values do not match"
    print(f"  Red channel shape:  {red.shape}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Run all exercises
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Lab 0.1: NumPy & Vectorized Thinking")
    print("=" * 60)

    print("\nExercise 1: Loop vs. Vectorized Dot Product")
    benchmark_dot_product()

    print("\nExercise 2: Normalize a Dataset with Broadcasting")
    test_normalize()

    print("\nExercise 3: Vectorized Loss Functions (MSE & MAE)")
    test_loss_functions()

    print("\nExercise 4: Reshape Challenge -- Image Tensor")
    test_reshape()

    print("\n" + "=" * 60)
    print("All exercises passed! Great work.")
    print("=" * 60)

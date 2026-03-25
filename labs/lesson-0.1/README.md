# Lab 0.1: NumPy & Vectorized Thinking

## Objectives

By completing this lab you will:

- Measure the performance difference between Python loops and NumPy vectorized operations
- Practice creating, indexing, slicing, and reshaping arrays
- Use broadcasting to normalize a dataset without loops
- Implement common ML building blocks (MSE, MAE) in vectorized form

## Setup

Make sure you have NumPy installed:

```bash
pip install numpy
```

## Instructions

Open `starter.py` and complete each function marked with `TODO`. Every function has:

1. A docstring explaining what it should do
2. Hints in the comments
3. An assertion or test call at the bottom of the file to verify correctness

### Exercise 1: Loop vs. Vectorized Dot Product

Implement dot product using a Python loop and using NumPy. Run the benchmark to see the speed difference.

### Exercise 2: Broadcasting -- Normalize a Dataset

Given a matrix of shape `(n_samples, n_features)`, subtract the column mean and divide by the column standard deviation using broadcasting (no loops).

### Exercise 3: Vectorized Loss Functions

Implement Mean Squared Error (MSE) and Mean Absolute Error (MAE) without loops.

### Exercise 4: Reshape Challenge -- Image Tensor

Given a flat array of pixel values, reshape it into an image tensor of shape `(batch, height, width, channels)` and extract a single color channel.

## Running

```bash
python starter.py
```

All exercises include assertions that print "PASSED" on success. If an assertion fails, read the error message and fix your implementation.

## Stretch Goals

- Implement a vectorized softmax function: `softmax(x) = exp(x) / sum(exp(x))`
- Use `np.einsum` to compute a batch of dot products
- Benchmark `float32` vs `float64` operations

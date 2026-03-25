# Lab 1.1: Linear Algebra for ML

## Overview

In this lab, you will practice the core linear algebra operations used throughout machine learning. You will work with vectors and matrices in NumPy, compute predictions using matrix multiplication, visualize 2D transformations, and perform a mini principal component analysis.

## Prerequisites

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## Exercises

### Exercise 1: Matrix Operations and Batch Predictions

Compare computing predictions with a Python loop versus matrix multiplication. You will:
- Create a data matrix of 1000 samples with 5 features
- Create a weight vector
- Compute predictions both ways and time them

### Exercise 2: 2D Transformations

Build rotation, scaling, and shear matrices. Apply them to a set of 2D points (a shape like a square or arrow) and plot the before/after.

### Exercise 3: Cosine Similarity

Given a set of "document vectors" (simulated word frequency vectors), compute the pairwise cosine similarity matrix and visualize it as a heatmap.

### Exercise 4: Mini-PCA

Generate a 2D dataset with a clear principal direction, compute its covariance matrix, find eigenvectors, and project the data onto the top principal component.

## Running

```bash
python starter.py
```

## Expected Output

Each exercise prints results to the console and generates matplotlib plots. You should see:
1. Timing comparison showing matrix multiplication is much faster
2. Plots of original and transformed shapes
3. A cosine similarity heatmap
4. A scatter plot with principal component directions overlaid

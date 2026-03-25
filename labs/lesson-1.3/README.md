# Lab 1.3: Calculus Intuition (Gradients & Optimization)

## Overview

In this lab, you will implement gradient descent from scratch, experiment with learning rates, and visualize optimization paths on loss surfaces. This is the most important algorithm in ML — by implementing it yourself, you will build deep intuition for how models learn.

## Prerequisites

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## Exercises

### Exercise 1: Gradient Descent from Scratch

Implement gradient descent for linear regression using only NumPy. Generate synthetic data from a known line, train the model, and plot the learned fit alongside the loss curve.

### Exercise 2: Learning Rate Experiment

Run gradient descent with 5 different learning rates on the same problem. Plot all loss curves on one chart and observe the effect of too-small, just-right, and too-large learning rates.

### Exercise 3: Visualize the Optimization Path

Create a 2D loss surface (contour plot) for a two-parameter model. Overlay the gradient descent trajectory from different starting points. Observe how the path navigates the surface.

### Exercise 4: Mini-Batch vs. Batch

Implement both full-batch and mini-batch gradient descent. Compare convergence speed and loss curve smoothness on the same dataset.

## Running

```bash
python starter.py
```

## Expected Output

Each exercise produces console output showing learned parameters and matplotlib visualizations of loss curves, fit lines, contour plots with trajectories, and convergence comparisons.

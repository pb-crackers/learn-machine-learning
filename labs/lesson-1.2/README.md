# Lab 1.2: Statistics & Probability for ML

## Overview

In this lab, you will explore probability distributions, compute descriptive statistics, apply Bayes' theorem, and investigate correlations. These are the statistical tools you will use every time you analyze a dataset or evaluate a model.

## Prerequisites

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## Exercises

### Exercise 1: Distribution Explorer

Generate samples from normal, uniform, and binomial distributions with different parameters. Plot histograms side by side and observe how the parameters change the shape.

### Exercise 2: Statistics Calculator

Compute descriptive statistics on a realistic dataset. Identify outliers using the z-score method (|z| > 3) and visualize them.

### Exercise 3: Bayes' Theorem in Practice

Implement the medical test scenario from the lesson. Vary the disease prevalence from 1% to 50% and plot how the posterior probability changes.

### Exercise 4: Correlation Investigation

Generate datasets with known correlations (positive, negative, zero). Compute correlation matrices and create scatter plot grids.

## Running

```bash
python starter.py
```

## Expected Output

Each exercise produces console output and matplotlib visualizations showing distributions, statistics, Bayesian updating curves, and correlation scatter plots.

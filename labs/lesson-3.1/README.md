# Lab 3.1: Build a Single-Layer Perceptron from Scratch

## Objective

Build a perceptron (single artificial neuron) from scratch using only NumPy. Train it to learn logic gates and visualize decision boundaries.

## Learning Goals

- Implement the forward pass of a single neuron (weighted sum + activation)
- Implement the perceptron learning rule for weight updates
- Understand why a single neuron can only learn linearly separable patterns
- Visualize decision boundaries to build geometric intuition

## Tasks

### Task 1: Implement the Perceptron Class
Complete the `Perceptron` class in `starter.py`:
- `forward()`: Compute the weighted sum and apply the activation function
- `predict()`: Classify inputs as 0 or 1
- `train()`: Update weights using the perceptron learning rule

### Task 2: Train on Logic Gates
- Train the perceptron on AND, OR, and NAND gates
- Verify it achieves 100% accuracy on all three
- Try training on XOR and observe that it fails

### Task 3: Visualize Decision Boundaries
- Plot the decision boundary for each learned gate
- Show the data points colored by class
- Explain geometrically why XOR is impossible for a single neuron

### Task 4: Experiment with Activation Functions
- Replace the step function with sigmoid activation
- Compare convergence speed and decision boundary smoothness
- Plot the loss curve during training

## Expected Output

```
AND Gate: Converged in 8 epochs, Accuracy: 100%
OR Gate:  Converged in 5 epochs, Accuracy: 100%
NAND Gate: Converged in 10 epochs, Accuracy: 100%
XOR Gate: Did not converge after 100 epochs, Accuracy: 50%
```

## Stretch Goals

- Implement a 2-input, 2-neuron layer that *can* solve XOR by combining outputs
- Experiment with different learning rates and plot convergence curves
- Add L2 regularization to the weight update rule

## Prerequisites

- Python 3.8+
- NumPy
- Matplotlib

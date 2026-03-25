"""
Lab 3.1: Build a Single-Layer Perceptron from Scratch
=====================================================

In this lab, you will implement a perceptron (single artificial neuron)
using only NumPy and train it to learn simple logic gates.
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """A single artificial neuron (perceptron).

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    activation : str
        Activation function to use: 'step' or 'sigmoid'.
    learning_rate : float
        Step size for weight updates.
    """

    def __init__(self, n_inputs, activation='step', learning_rate=0.1):
        # TODO: Initialize weights to small random values (shape: n_inputs,)
        self.weights = None
        # TODO: Initialize bias to 0.0
        self.bias = None
        self.activation = activation
        self.learning_rate = learning_rate

    def _activate(self, z):
        """Apply the activation function.

        Parameters
        ----------
        z : float
            The weighted sum (pre-activation value).

        Returns
        -------
        float
            The activated output.
        """
        if self.activation == 'step':
            # TODO: Return 1.0 if z >= 0, else 0.0
            pass
        elif self.activation == 'sigmoid':
            # TODO: Return the sigmoid of z: 1 / (1 + exp(-z))
            pass
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x):
        """Compute the neuron's output for a single input.

        Parameters
        ----------
        x : np.ndarray, shape (n_inputs,)
            A single input sample.

        Returns
        -------
        float
            The neuron's output after activation.
        """
        # TODO: Compute the weighted sum: z = dot(weights, x) + bias
        z = None
        # TODO: Apply the activation function
        output = None
        return output

    def predict(self, X):
        """Predict class labels for multiple inputs.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_inputs)
            Input data.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted labels (0 or 1).
        """
        # TODO: Apply forward() to each row of X and threshold at 0.5
        pass

    def train(self, X, y, n_epochs=100):
        """Train the perceptron using the perceptron learning rule.

        The perceptron learning rule:
            error = y_true - y_pred
            weights += learning_rate * error * x
            bias += learning_rate * error

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_inputs)
            Training data.
        y : np.ndarray, shape (n_samples,)
            Target labels (0 or 1).
        n_epochs : int
            Maximum number of training epochs.

        Returns
        -------
        list
            Number of misclassifications at each epoch.
        """
        errors_per_epoch = []

        for epoch in range(n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                # TODO: Compute the prediction
                prediction = None

                # TODO: Compute the error (yi - prediction)
                error = None

                # TODO: Update weights using the perceptron learning rule
                # self.weights += ...
                # self.bias += ...

                if abs(error) > 0.5:  # Misclassification
                    errors += 1

            errors_per_epoch.append(errors)

            # Early stopping: if no errors, we've converged
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
        else:
            print(f"Did not converge after {n_epochs} epochs")

        return errors_per_epoch


def plot_decision_boundary(perceptron, X, y, title="Decision Boundary"):
    """Plot the decision boundary of a trained perceptron.

    Parameters
    ----------
    perceptron : Perceptron
        A trained perceptron.
    X : np.ndarray, shape (n_samples, 2)
        Input data (must be 2D for visualization).
    y : np.ndarray, shape (n_samples,)
        Target labels.
    title : str
        Plot title.
    """
    # TODO: Create a mesh grid covering the input space
    # TODO: Compute predictions for every point in the grid
    # TODO: Plot the decision regions using plt.contourf()
    # TODO: Plot the data points colored by class
    # TODO: Add title, labels, and legend

    # Hint: Use np.meshgrid and reshape the grid for prediction
    pass


# ============================================================
# Task 1: Test your perceptron on logic gates
# ============================================================

# Data for logic gates
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# AND gate
y_and = np.array([0, 0, 0, 1])

# OR gate
y_or = np.array([0, 1, 1, 1])

# NAND gate
y_nand = np.array([1, 1, 1, 0])

# XOR gate (should fail!)
y_xor = np.array([0, 1, 1, 0])

# TODO: Create a perceptron and train it on each gate
# TODO: Print accuracy for each gate

print("=" * 50)
print("Task 1: Logic Gate Learning")
print("=" * 50)

# Example for AND gate:
# p_and = Perceptron(n_inputs=2, activation='step')
# errors = p_and.train(X, y_and)
# predictions = p_and.predict(X)
# accuracy = np.mean(predictions == y_and) * 100
# print(f"AND Gate - Accuracy: {accuracy:.0f}%")


# ============================================================
# Task 2: Visualize decision boundaries
# ============================================================

print("\n" + "=" * 50)
print("Task 2: Decision Boundaries")
print("=" * 50)

# TODO: Plot decision boundaries for AND, OR, NAND, and XOR
# Use a 2x2 subplot grid


# ============================================================
# Task 3: Sigmoid activation comparison
# ============================================================

print("\n" + "=" * 50)
print("Task 3: Sigmoid Activation")
print("=" * 50)

# TODO: Train a perceptron with sigmoid activation on AND gate
# TODO: Compare convergence speed with step activation
# TODO: Plot the error curves for both activations on the same plot


# ============================================================
# Task 4 (Stretch): Solve XOR with two layers
# ============================================================

print("\n" + "=" * 50)
print("Task 4 (Stretch): XOR with Two Perceptrons")
print("=" * 50)

# Hint: XOR(a, b) = AND(OR(a, b), NAND(a, b))
# You can combine two trained perceptrons to solve XOR!

# TODO: Use the trained OR and NAND perceptrons as a hidden layer
# TODO: Feed their outputs into an AND perceptron
# TODO: Verify this solves XOR


if __name__ == '__main__':
    print("\nLab 3.1 Complete!")
    print("Review your results and make sure you understand:")
    print("  - Why the perceptron can learn AND, OR, NAND but not XOR")
    print("  - How weights and bias define a linear decision boundary")
    print("  - The role of the activation function in a neuron")

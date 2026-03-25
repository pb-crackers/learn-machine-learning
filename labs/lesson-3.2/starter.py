"""
Lab 3.2: Two-Layer Neural Network from Scratch with Backpropagation
===================================================================

Build a neural network from scratch using only NumPy and train it
to classify MNIST handwritten digits.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Helper: Load MNIST data
# ============================================================
def load_mnist():
    """Load and preprocess MNIST dataset.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
        Preprocessed data splits.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data.astype(np.float32), mnist.target.astype(np.int64)

    # Normalize to [0, 1]
    X = X / 255.0

    # Split: 60k train, 5k val, 5k test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=5000, random_state=42, stratify=y_temp
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoded vectors.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Integer class labels.
    num_classes : int
        Number of classes.

    Returns
    -------
    np.ndarray, shape (n_samples, num_classes)
        One-hot encoded labels.
    """
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot


# ============================================================
# Activation functions
# ============================================================

def relu(z):
    """ReLU activation function."""
    # TODO: Implement ReLU: max(0, z)
    pass


def relu_derivative(z):
    """Derivative of ReLU."""
    # TODO: Return 1 where z > 0, else 0
    pass


def softmax(z):
    """Numerically stable softmax.

    Parameters
    ----------
    z : np.ndarray, shape (batch_size, num_classes)
        Raw logits.

    Returns
    -------
    np.ndarray, shape (batch_size, num_classes)
        Probabilities summing to 1 along axis=1.
    """
    # TODO: Implement numerically stable softmax
    # Hint: Subtract the max of each row for numerical stability
    # exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    # return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    pass


# ============================================================
# The Neural Network
# ============================================================

class TwoLayerNet:
    """A 2-layer fully connected neural network for classification.

    Architecture: Input -> Hidden (ReLU) -> Output (Softmax)

    Parameters
    ----------
    input_size : int
        Number of input features (784 for MNIST).
    hidden_size : int
        Number of neurons in the hidden layer.
    output_size : int
        Number of output classes (10 for MNIST).
    """

    def __init__(self, input_size, hidden_size, output_size):
        # He initialization for weights
        # TODO: Initialize W1 with shape (input_size, hidden_size)
        #       using np.random.randn * sqrt(2 / input_size)
        self.W1 = None
        self.b1 = np.zeros(hidden_size)

        # TODO: Initialize W2 with shape (hidden_size, output_size)
        self.W2 = None
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : np.ndarray, shape (batch_size, input_size)
            Input data.

        Returns
        -------
        np.ndarray, shape (batch_size, output_size)
            Predicted probabilities for each class.
        """
        # TODO: Layer 1 - Linear + ReLU
        # self.z1 = X @ self.W1 + self.b1
        # self.a1 = relu(self.z1)

        # TODO: Layer 2 - Linear + Softmax
        # self.z2 = self.a1 @ self.W2 + self.b2
        # self.a2 = softmax(self.z2)

        # Save input for backward pass
        # self.X = X

        # return self.a2
        pass

    def compute_loss(self, y_pred, y_true_onehot, epsilon=1e-15):
        """Compute categorical cross-entropy loss.

        Parameters
        ----------
        y_pred : np.ndarray, shape (batch_size, num_classes)
            Predicted probabilities.
        y_true_onehot : np.ndarray, shape (batch_size, num_classes)
            One-hot encoded true labels.

        Returns
        -------
        float
            Average cross-entropy loss.
        """
        # TODO: Implement cross-entropy loss
        # L = -mean(sum(y_true * log(y_pred)))
        pass

    def backward(self, y_true_onehot):
        """Backward pass: compute gradients for all parameters.

        Parameters
        ----------
        y_true_onehot : np.ndarray, shape (batch_size, num_classes)
            One-hot encoded true labels.

        Returns
        -------
        dict
            Gradients for W1, b1, W2, b2.
        """
        batch_size = y_true_onehot.shape[0]

        # TODO: Output layer gradient
        # For softmax + cross-entropy, the gradient simplifies to:
        # dz2 = (self.a2 - y_true_onehot) / batch_size

        # TODO: Compute dW2 and db2
        # dW2 = self.a1.T @ dz2
        # db2 = np.sum(dz2, axis=0)

        # TODO: Propagate gradient to hidden layer
        # da1 = dz2 @ self.W2.T
        # dz1 = da1 * relu_derivative(self.z1)

        # TODO: Compute dW1 and db1
        # dW1 = self.X.T @ dz1
        # db1 = np.sum(dz1, axis=0)

        # return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        pass

    def update_weights(self, grads, learning_rate):
        """Update weights using gradient descent.

        Parameters
        ----------
        grads : dict
            Gradients from backward().
        learning_rate : float
            Step size for updates.
        """
        # TODO: Update all four parameters
        # self.W1 -= learning_rate * grads['dW1']
        # ...
        pass

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_size)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class indices.
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


# ============================================================
# Training function
# ============================================================

def train(model, X_train, y_train, X_val, y_val,
          epochs=50, batch_size=128, learning_rate=0.01):
    """Train the neural network.

    Parameters
    ----------
    model : TwoLayerNet
        The network to train.
    X_train, y_train : np.ndarray
        Training data and integer labels.
    X_val, y_val : np.ndarray
        Validation data and integer labels.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate for gradient descent.

    Returns
    -------
    dict
        Training history (losses and accuracies).
    """
    y_train_onehot = one_hot_encode(y_train)
    n_samples = X_train.shape[0]

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]

        epoch_loss = 0.0
        n_batches = 0

        # TODO: Mini-batch training loop
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # TODO: Forward pass
            # TODO: Compute loss
            # TODO: Backward pass
            # TODO: Update weights

            pass

        # TODO: Compute training metrics
        # train_pred = model.predict(X_train)
        # train_acc = np.mean(train_pred == y_train)

        # TODO: Compute validation metrics
        # val_pred = model.predict(X_val)
        # val_acc = np.mean(val_pred == y_val)

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            pass
            # print(f"Epoch {epoch+1}/{epochs} | "
            #       f"Train Loss: {avg_loss:.4f} | "
            #       f"Train Acc: {train_acc:.1%} | "
            #       f"Val Acc: {val_acc:.1%}")

    return history


# ============================================================
# Gradient checking
# ============================================================

def gradient_check(model, X_sample, y_sample_onehot, epsilon=1e-5):
    """Verify backprop gradients against numerical gradients.

    Parameters
    ----------
    model : TwoLayerNet
    X_sample : np.ndarray, shape (1, input_size)
    y_sample_onehot : np.ndarray, shape (1, num_classes)
    epsilon : float
        Perturbation size for numerical gradient.
    """
    # Get analytical gradients
    model.forward(X_sample)
    grads = model.backward(y_sample_onehot)

    # TODO: Check a few elements of dW1
    # For each element W1[i, j]:
    #   1. Add epsilon to W1[i,j], compute loss (loss_plus)
    #   2. Subtract epsilon from W1[i,j], compute loss (loss_minus)
    #   3. Numerical gradient = (loss_plus - loss_minus) / (2 * epsilon)
    #   4. Compare with grads['dW1'][i, j]
    #   5. Compute relative error

    print("Gradient check: (implement me!)")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

    # Task 1: Create and train the network
    print("=" * 60)
    print("Task 1: Train a 2-layer network on MNIST")
    print("=" * 60)

    # TODO: Create the network
    # model = TwoLayerNet(input_size=784, hidden_size=128, output_size=10)

    # TODO: Train it
    # history = train(model, X_train, y_train, X_val, y_val,
    #                 epochs=50, batch_size=128, learning_rate=0.01)

    # TODO: Evaluate on test set
    # test_pred = model.predict(X_test)
    # test_acc = np.mean(test_pred == y_test)
    # print(f"\nFinal Test Accuracy: {test_acc:.1%}")

    # Task 2: Plot training curves
    print("\n" + "=" * 60)
    print("Task 2: Plot Training Curves")
    print("=" * 60)
    # TODO: Plot training and validation loss/accuracy

    # Task 3: Hyperparameter experiments
    print("\n" + "=" * 60)
    print("Task 3: Hyperparameter Experiments")
    print("=" * 60)
    # TODO: Try different learning rates and hidden sizes

    # Task 4: Gradient checking
    print("\n" + "=" * 60)
    print("Task 4: Gradient Checking")
    print("=" * 60)
    # TODO: Run gradient check on a small sample

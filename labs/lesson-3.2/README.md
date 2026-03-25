# Lab 3.2: Two-Layer Neural Network from Scratch with Backpropagation

## Objective

Build a 2-layer neural network entirely from scratch using NumPy, implement backpropagation, and train it to classify handwritten digits from the MNIST dataset.

## Learning Goals

- Implement the forward pass for a multi-layer network
- Implement backpropagation (the chain rule applied layer by layer)
- Train on real data (MNIST) and achieve reasonable accuracy
- Understand the effect of hyperparameters on training

## Tasks

### Task 1: Implement the Network Class
Complete the `TwoLayerNet` class:
- `forward()`: Compute predictions through both layers
- `compute_loss()`: Compute cross-entropy loss
- `backward()`: Compute gradients using backpropagation
- `update_weights()`: Apply gradient descent updates

### Task 2: Train on MNIST
- Load MNIST data (provided helper function)
- Train for 50 epochs with mini-batch gradient descent
- Plot the training and validation loss curves
- Achieve at least 95% test accuracy

### Task 3: Hyperparameter Experiments
- Try different learning rates: 0.001, 0.01, 0.1, 1.0
- Try different hidden layer sizes: 32, 64, 128, 256
- Plot accuracy vs. hyperparameter for each experiment

### Task 4: Gradient Checking
- Implement numerical gradient checking
- Verify your backprop gradients match numerical approximations
- Report the relative error for a sample of weights

## Expected Output

```
Epoch 10/50 | Train Loss: 0.312 | Train Acc: 91.2% | Val Acc: 92.1%
Epoch 20/50 | Train Loss: 0.198 | Train Acc: 94.5% | Val Acc: 94.8%
Epoch 50/50 | Train Loss: 0.089 | Train Acc: 97.6% | Val Acc: 96.2%

Final Test Accuracy: 96.1%
```

## Stretch Goals

- Add a third hidden layer and observe how training changes
- Implement learning rate decay
- Visualize the learned weight matrices as images

## Prerequisites

- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn (for loading MNIST)

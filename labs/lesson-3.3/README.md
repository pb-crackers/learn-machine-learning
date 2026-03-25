# Lab 3.3: Rebuild the MNIST Classifier in PyTorch

## Objective

Rebuild the MNIST digit classifier from Lab 3.2 using PyTorch, learning the framework's core abstractions: tensors, autograd, nn.Module, DataLoader, and the standard training loop.

## Learning Goals

- Convert between NumPy arrays and PyTorch tensors
- Build a neural network using `nn.Module`
- Use `DataLoader` for efficient batched data loading
- Implement the standard PyTorch training loop
- Save and load model checkpoints

## Tasks

### Task 1: Data Loading with PyTorch
- Use `torchvision.datasets.MNIST` to load the data
- Create `DataLoader` objects for train, validation, and test sets
- Apply appropriate transforms (ToTensor, Normalize)

### Task 2: Build the Model
- Create an `MNISTNet` class extending `nn.Module`
- Architecture: 784 -> 256 (ReLU) -> 128 (ReLU) -> 10
- Print the model summary and parameter count

### Task 3: Train the Model
- Use `nn.CrossEntropyLoss` and `optim.Adam`
- Implement the training loop with train/validation split
- Track loss and accuracy per epoch
- Achieve at least 97% test accuracy

### Task 4: Experiment and Improve
- Try different architectures (wider, deeper)
- Compare SGD vs. Adam optimizers
- Add dropout and observe the effect on overfitting
- Save the best model checkpoint

### Task 5: Visualize Results
- Plot training/validation curves
- Show example predictions (correct and incorrect)
- Create a confusion matrix

## Expected Output

```
Epoch 1/15  | Train Loss: 0.258, Acc: 92.4% | Val Acc: 95.8%
Epoch 5/15  | Train Loss: 0.054, Acc: 98.3% | Val Acc: 97.5%
Epoch 15/15 | Train Loss: 0.012, Acc: 99.6% | Val Acc: 98.0%

Test Accuracy: 97.8%
```

## Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- Matplotlib

# Lab 3.4: CNN in PyTorch for CIFAR-10

## Objective

Build and train a Convolutional Neural Network (CNN) in PyTorch to classify CIFAR-10 images (10 classes of 32x32 color images). Apply convolutions, pooling, and batch normalization to achieve strong performance.

## Learning Goals

- Build a CNN using `nn.Conv2d`, `nn.MaxPool2d`, and `nn.BatchNorm2d`
- Understand how spatial dimensions change through conv and pool layers
- Apply data augmentation to improve generalization
- Visualize learned convolutional filters and feature maps

## Tasks

### Task 1: Load CIFAR-10 with Data Augmentation
- Apply training augmentations: random crop, horizontal flip, normalization
- Apply test transforms: just normalization (no augmentation)
- Create DataLoaders with appropriate batch sizes

### Task 2: Build the CNN
Complete the `CIFAR10CNN` class:
- 3 convolutional blocks, each with: Conv -> BatchNorm -> ReLU -> MaxPool
- A fully connected classifier head
- Print the model architecture and parameter count

### Task 3: Train and Evaluate
- Train for 30 epochs with Adam optimizer
- Track training and validation accuracy per epoch
- Target: 80%+ test accuracy

### Task 4: Visualize
- Plot training curves (loss and accuracy)
- Visualize the first layer's learned filters
- Show feature maps for a sample image at different layers
- Display a grid of example predictions

### Task 5: Architecture Experiments
- Compare your CNN against a fully connected network (same parameter count)
- Try adding more layers or changing filter counts
- Experiment with different pooling strategies

## Expected Output

```
Epoch 5/30  | Train Acc: 72.3% | Test Acc: 71.8%
Epoch 15/30 | Train Acc: 89.1% | Test Acc: 82.4%
Epoch 30/30 | Train Acc: 96.2% | Test Acc: 85.1%

Final Test Accuracy: 85.1%
```

## CIFAR-10 Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- Matplotlib

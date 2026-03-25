"""
Lab 3.4: CNN in PyTorch for CIFAR-10
=====================================

Build and train a Convolutional Neural Network to classify
CIFAR-10 images (32x32 color images, 10 classes).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-10 normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ============================================================
# Task 1: Data Loading with Augmentation
# ============================================================

def get_data_loaders(batch_size=128):
    """Create CIFAR-10 data loaders with augmentation.

    Returns
    -------
    train_loader, test_loader : DataLoader
    """
    # TODO: Define training transforms with augmentation
    # - RandomCrop(32, padding=4)
    # - RandomHorizontalFlip()
    # - ToTensor()
    # - Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_transform = None

    # TODO: Define test transforms (no augmentation!)
    # - ToTensor()
    # - Normalize(CIFAR10_MEAN, CIFAR10_STD)
    test_transform = None

    # TODO: Load datasets
    # train_dataset = datasets.CIFAR10(root='./data', train=True,
    #                                  download=True, transform=train_transform)
    # test_dataset = datasets.CIFAR10(root='./data', train=False,
    #                                 download=True, transform=test_transform)

    # TODO: Create DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                           shuffle=True, num_workers=2)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size,
    #                          shuffle=False, num_workers=2)
    # return train_loader, test_loader
    pass


# ============================================================
# Task 2: Build the CNN
# ============================================================

class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10 classification.

    Architecture:
        Block 1: Conv(3->32, 3x3) -> BN -> ReLU -> Conv(32->32, 3x3) -> BN -> ReLU -> MaxPool
        Block 2: Conv(32->64, 3x3) -> BN -> ReLU -> Conv(64->64, 3x3) -> BN -> ReLU -> MaxPool
        Block 3: Conv(64->128, 3x3) -> BN -> ReLU -> Conv(128->128, 3x3) -> BN -> ReLU -> MaxPool
        Classifier: Flatten -> Linear(128*4*4, 256) -> ReLU -> Dropout -> Linear(256, 10)
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # TODO: Define convolutional blocks
        # self.block1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),       # 32x32 -> 16x16
        # )

        # TODO: Block 2 (32->64 channels, 16x16 -> 8x8)

        # TODO: Block 3 (64->128 channels, 8x8 -> 4x4)

        # TODO: Define classifier head
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128 * 4 * 4, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, num_classes),
        # )
        pass

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, 3, 32, 32)
            Batch of CIFAR-10 images.

        Returns
        -------
        torch.Tensor, shape (batch_size, 10)
            Raw logits.
        """
        # TODO: Pass through blocks and classifier
        # x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)
        # x = self.classifier(x)
        # return x
        pass


# ============================================================
# Task 3: Training and Evaluation
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # TODO: Standard training step
        # optimizer.zero_grad()
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # TODO: Track metrics
        pass

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            # TODO: Forward pass and track metrics
            pass

    return total_loss / total, correct / total


# ============================================================
# Task 4: Visualization
# ============================================================

def visualize_filters(model):
    """Visualize the learned filters of the first convolutional layer."""
    # TODO: Get weights from the first conv layer
    # TODO: Normalize and display as a grid of images
    pass


def visualize_feature_maps(model, image, device):
    """Show feature maps at different layers for a sample image.

    Parameters
    ----------
    model : CIFAR10CNN
    image : torch.Tensor, shape (1, 3, 32, 32)
    device : torch.device
    """
    # TODO: Register forward hooks to capture intermediate activations
    # TODO: Run a forward pass
    # TODO: Display feature maps as image grids
    pass


def show_predictions(model, test_loader, device, n=16):
    """Show a grid of predictions on test images."""
    # TODO: Get a batch of test images
    # TODO: Make predictions
    # TODO: Display in a grid with predicted class labels
    # TODO: Color-code correct (green) vs incorrect (red)
    pass


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Task 1: Load data
    print("=" * 60)
    print("Task 1: Loading CIFAR-10")
    print("=" * 60)
    # TODO: train_loader, test_loader = get_data_loaders()

    # Task 2: Build model
    print("\n" + "=" * 60)
    print("Task 2: Building CNN")
    print("=" * 60)
    # TODO: model = CIFAR10CNN().to(device)
    # TODO: Print model summary and parameter count
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")

    # Task 3: Train
    print("\n" + "=" * 60)
    print("Task 3: Training")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    # TODO: optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    # TODO: Training loop for 30 epochs
    # for epoch in range(30):
    #     train_loss, train_acc = train_one_epoch(...)
    #     test_loss, test_acc = evaluate(...)
    #     # Store in history
    #     # Print progress

    # Task 4: Visualize
    print("\n" + "=" * 60)
    print("Task 4: Visualizations")
    print("=" * 60)
    # TODO: Plot training curves
    # TODO: Visualize filters
    # TODO: Show predictions

    # Task 5: Architecture comparison
    print("\n" + "=" * 60)
    print("Task 5: CNN vs FC Network Comparison")
    print("=" * 60)
    # TODO: Create an FC network with similar parameter count
    # TODO: Train it on CIFAR-10
    # TODO: Compare test accuracy

    print("\nLab 3.4 Complete!")

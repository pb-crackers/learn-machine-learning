"""
Lab 3.3: Rebuild the MNIST Classifier in PyTorch
=================================================

Rebuild the MNIST classifier using PyTorch, learning the standard
framework patterns: nn.Module, DataLoader, and the training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Task 1: Data Loading
# ============================================================

def get_data_loaders(batch_size=64):
    """Create train, validation, and test data loaders.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
        PyTorch data loaders for each split.
    """
    # TODO: Define transforms
    # - Convert to tensor (transforms.ToTensor())
    # - Normalize with MNIST mean=0.1307, std=0.3081
    transform = None

    # TODO: Load MNIST dataset
    # train_full = datasets.MNIST(root='./data', train=True,
    #                             download=True, transform=transform)
    # test_dataset = datasets.MNIST(root='./data', train=False,
    #                               download=True, transform=transform)

    # TODO: Split training into train (55000) and validation (5000)
    # train_dataset, val_dataset = random_split(train_full, [55000, 5000])

    # TODO: Create DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                           shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # return train_loader, val_loader, test_loader
    pass


# ============================================================
# Task 2: Build the Model
# ============================================================

class MNISTNet(nn.Module):
    """Feedforward neural network for MNIST classification.

    Architecture: 784 -> 256 (ReLU) -> 128 (ReLU) -> 10
    """

    def __init__(self):
        super().__init__()
        # TODO: Define the layers
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(784, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 10)
        # self.relu = nn.ReLU()
        pass

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, 1, 28, 28)
            Batch of MNIST images.

        Returns
        -------
        torch.Tensor, shape (batch_size, 10)
            Raw logits for each class.
        """
        # TODO: Implement forward pass
        # x = self.flatten(x)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
        pass


# ============================================================
# Task 3: Training Loop
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # TODO: Implement the training step
        # 1. optimizer.zero_grad()
        # 2. Forward pass: outputs = model(images)
        # 3. Compute loss: loss = criterion(outputs, labels)
        # 4. Backward pass: loss.backward()
        # 5. Update weights: optimizer.step()

        # TODO: Track metrics
        # total_loss += loss.item() * images.size(0)
        # _, predicted = outputs.max(1)
        # correct += predicted.eq(labels).sum().item()
        # total += labels.size(0)
        pass

    return total_loss / total, correct / total


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # TODO: Forward pass and compute metrics
            pass

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, device,
                epochs=15, lr=0.001):
    """Full training loop with validation monitoring.

    Returns
    -------
    history : dict
        Training metrics per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1%} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1%}")

        # TODO: Save best model
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), 'best_mnist_model.pth')

    return history


# ============================================================
# Task 5: Visualization
# ============================================================

def plot_training_curves(history):
    """Plot training and validation loss/accuracy curves."""
    # TODO: Create a 1x2 subplot with loss and accuracy curves
    pass


def show_predictions(model, test_loader, device, n_examples=10):
    """Show example predictions with the actual images."""
    # TODO: Get a batch of test images
    # TODO: Make predictions
    # TODO: Display images with predicted and true labels
    # TODO: Highlight incorrect predictions in red
    pass


def plot_confusion_matrix(model, test_loader, device):
    """Create and plot a confusion matrix."""
    # TODO: Collect all predictions and true labels
    # TODO: Compute confusion matrix
    # TODO: Plot as a heatmap
    pass


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Task 1: Load data
    print("=" * 60)
    print("Task 1: Loading Data")
    print("=" * 60)
    # TODO: train_loader, val_loader, test_loader = get_data_loaders()

    # Task 2: Build model
    print("\n" + "=" * 60)
    print("Task 2: Building Model")
    print("=" * 60)
    # TODO: model = MNISTNet().to(device)
    # TODO: Print model and parameter count

    # Task 3: Train
    print("\n" + "=" * 60)
    print("Task 3: Training")
    print("=" * 60)
    # TODO: history = train_model(model, train_loader, val_loader, device)

    # Task 4: Evaluate on test set
    print("\n" + "=" * 60)
    print("Task 4: Final Evaluation")
    print("=" * 60)
    # TODO: test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    # print(f"Test Accuracy: {test_acc:.1%}")

    # Task 5: Visualize
    print("\n" + "=" * 60)
    print("Task 5: Visualizations")
    print("=" * 60)
    # TODO: plot_training_curves(history)
    # TODO: show_predictions(model, test_loader, device)
    # TODO: plot_confusion_matrix(model, test_loader, device)

    print("\nLab 3.3 Complete!")

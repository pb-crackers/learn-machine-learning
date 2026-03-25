"""
Lab 3.5: Fine-Tune a Pretrained Model on a Custom Dataset
==========================================================

Use transfer learning to adapt a pretrained ResNet-18 for a
custom image classification task. Apply data augmentation,
learning rate scheduling, and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
import copy


# ImageNet normalization (required for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Task 1: Data Preparation
# ============================================================

def get_data_loaders(data_dir='./data', batch_size=32):
    """Create data loaders with augmentation for transfer learning.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.
    batch_size : int
        Batch size for data loaders.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    num_classes : int
    """
    # TODO: Define training transforms
    # - Resize to 256
    # - Random crop to 224
    # - Random horizontal flip
    # - Random rotation (up to 15 degrees)
    # - Color jitter (brightness=0.2, contrast=0.2)
    # - ToTensor()
    # - Normalize with ImageNet statistics
    train_transform = transforms.Compose([
        # TODO: Add transforms
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # TODO: Define validation/test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load dataset
    # Option 1: Use a built-in dataset
    # full_dataset = datasets.Flowers102(root=data_dir, split='train',
    #                                    download=True, transform=train_transform)

    # Option 2: Use ImageFolder for custom data
    # full_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
    #                                     transform=train_transform)

    # For this lab, we'll use CIFAR-10 as a stand-in
    # (replace with your dataset of choice)
    full_train = datasets.CIFAR10(root=data_dir, train=True,
                                  download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=eval_transform)

    # TODO: Split train into train/val (e.g., 90%/10%)
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    num_classes = 10  # Adjust for your dataset
    return train_loader, val_loader, test_loader, num_classes


# ============================================================
# Task 2: Set Up Transfer Learning
# ============================================================

def create_transfer_model(num_classes, freeze_backbone=True):
    """Create a pretrained ResNet-18 for transfer learning.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    freeze_backbone : bool
        If True, freeze all layers except the final classifier.

    Returns
    -------
    model : nn.Module
    """
    # TODO: Load pretrained ResNet-18
    # model = models.resnet18(weights='IMAGENET1K_V1')
    model = models.resnet18(weights='IMAGENET1K_V1')

    # TODO: Freeze backbone parameters
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # TODO: Replace the final FC layer
    # The original fc layer is nn.Linear(512, 1000) for ImageNet
    # Replace it for your number of classes
    # model.fc = nn.Sequential(
    #     nn.Linear(512, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.3),
    #     nn.Linear(256, num_classes)
    # )
    pass

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")

    return model


def create_scratch_model(num_classes):
    """Create a ResNet-18 without pretrained weights (for comparison).

    Parameters
    ----------
    num_classes : int

    Returns
    -------
    model : nn.Module
    """
    # TODO: Load ResNet-18 WITHOUT pretrained weights
    # model = models.resnet18(weights=None)
    # model.fc = nn.Linear(512, num_classes)
    # return model
    pass


# ============================================================
# Task 3 & 4: Training with Early Stopping
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_with_early_stopping(model, train_loader, val_loader, device,
                              optimizer, scheduler=None,
                              epochs=20, patience=5):
    """Train model with early stopping and optional LR scheduling.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    device : torch.device
    optimizer : torch.optim.Optimizer
    scheduler : optional learning rate scheduler
    epochs : int
    patience : int
        Number of epochs without improvement before stopping.

    Returns
    -------
    history : dict
    best_model_state : dict
    """
    criterion = nn.CrossEntropyLoss()
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        # Step scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"  Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
              f"Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")

        # TODO: Early stopping logic
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_model_state = copy.deepcopy(model.state_dict())
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"  Early stopping at epoch {epoch+1}")
        #         break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"  Best validation accuracy: {best_val_acc:.1%}")
    return history, best_model_state


# ============================================================
# Task 4: Fine-Tuning (Unfreeze Later Layers)
# ============================================================

def unfreeze_later_layers(model):
    """Unfreeze the last residual block (layer4) for fine-tuning.

    Parameters
    ----------
    model : nn.Module
        A ResNet model with frozen backbone.
    """
    # TODO: Unfreeze layer4
    # for param in model.layer4.parameters():
    #     param.requires_grad = True

    # Print updated parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"After unfreezing layer4: {trainable:,} / {total:,} trainable "
          f"({100*trainable/total:.1f}%)")
    pass


def create_fine_tune_optimizer(model, lr_pretrained=1e-4, lr_new=1e-3):
    """Create optimizer with different learning rates for pretrained
    and new layers.

    Parameters
    ----------
    model : nn.Module
    lr_pretrained : float
        Learning rate for pretrained (unfrozen) layers.
    lr_new : float
        Learning rate for newly added layers.

    Returns
    -------
    optimizer : torch.optim.Optimizer
    """
    # TODO: Create parameter groups with different learning rates
    # pretrained_params = []
    # new_params = []
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if 'fc' in name:
    #             new_params.append(param)
    #         else:
    #             pretrained_params.append(param)
    #
    # optimizer = optim.Adam([
    #     {'params': pretrained_params, 'lr': lr_pretrained},
    #     {'params': new_params, 'lr': lr_new},
    # ], weight_decay=1e-4)
    # return optimizer
    pass


# ============================================================
# Visualization
# ============================================================

def plot_training_comparison(histories, labels):
    """Plot training curves for multiple experiments.

    Parameters
    ----------
    histories : list of dicts
    labels : list of str
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for history, label in zip(histories, labels):
        axes[0].plot(history['train_loss'], label=f'{label} (train)')
        axes[0].plot(history['val_loss'], '--', label=f'{label} (val)')

        axes[1].plot(history['train_acc'], label=f'{label} (train)')
        axes[1].plot(history['val_acc'], '--', label=f'{label} (val)')

    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Task 1: Load data
    print("=" * 60)
    print("Task 1: Loading Data")
    print("=" * 60)
    train_loader, val_loader, test_loader, num_classes = get_data_loaders()

    # Task 2: Create transfer learning model
    print("\n" + "=" * 60)
    print("Task 2: Feature Extraction (Frozen Backbone)")
    print("=" * 60)
    # TODO: model = create_transfer_model(num_classes, freeze_backbone=True)
    # TODO: model = model.to(device)

    # Task 3: Train with frozen backbone
    print("\n" + "=" * 60)
    print("Task 3: Training (Feature Extraction)")
    print("=" * 60)
    # TODO: optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    # TODO: history_fe, _ = train_with_early_stopping(
    #     model, train_loader, val_loader, device,
    #     optimizer=optimizer, epochs=10, patience=5
    # )

    # Task 4: Fine-tune with unfrozen layers
    print("\n" + "=" * 60)
    print("Task 4: Fine-Tuning (Unfrozen layer4)")
    print("=" * 60)
    # TODO: unfreeze_later_layers(model)
    # TODO: optimizer = create_fine_tune_optimizer(model)
    # TODO: scheduler = CosineAnnealingLR(optimizer, T_max=20)
    # TODO: history_ft, _ = train_with_early_stopping(
    #     model, train_loader, val_loader, device,
    #     optimizer=optimizer, scheduler=scheduler,
    #     epochs=20, patience=5
    # )

    # Evaluate on test set
    # criterion = nn.CrossEntropyLoss()
    # test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    # print(f"\nTransfer Learning Test Accuracy: {test_acc:.1%}")

    # Task 5: Compare with training from scratch
    print("\n" + "=" * 60)
    print("Task 5: Training from Scratch (Comparison)")
    print("=" * 60)
    # TODO: scratch_model = create_scratch_model(num_classes).to(device)
    # TODO: optimizer = optim.Adam(scratch_model.parameters(), lr=0.001)
    # TODO: scheduler = CosineAnnealingLR(optimizer, T_max=30)
    # TODO: history_scratch, _ = train_with_early_stopping(
    #     scratch_model, train_loader, val_loader, device,
    #     optimizer=optimizer, scheduler=scheduler,
    #     epochs=30, patience=10
    # )
    # TODO: scratch_test_loss, scratch_test_acc = evaluate(
    #     scratch_model, test_loader, criterion, device
    # )
    # print(f"From Scratch Test Accuracy: {scratch_test_acc:.1%}")

    # Visualization
    # TODO: plot_training_comparison(
    #     [history_fe, history_ft, history_scratch],
    #     ['Feature Extraction', 'Fine-Tuned', 'From Scratch']
    # )

    print("\nLab 3.5 Complete!")
    print("Key takeaway: Transfer learning achieves better results")
    print("with less data and less training time.")

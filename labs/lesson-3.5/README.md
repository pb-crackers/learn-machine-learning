# Lab 3.5: Fine-Tune a Pretrained Model on a Custom Dataset

## Objective

Use transfer learning to fine-tune a pretrained ResNet-18 model on a custom image classification dataset. Apply the practical deep learning techniques from Lesson 3.5: data augmentation, learning rate scheduling, and early stopping.

## Learning Goals

- Load and use pretrained models from `torchvision.models`
- Freeze and unfreeze layers for transfer learning
- Apply data augmentation appropriate for the target domain
- Implement learning rate scheduling and early stopping
- Compare transfer learning vs. training from scratch

## Dataset

This lab uses the Oxford Flowers 102 dataset (or any custom dataset in ImageFolder format). You can also substitute your own dataset.

## Tasks

### Task 1: Prepare the Data
- Load a dataset using `torchvision.datasets` (Flowers102 or ImageFolder)
- Apply ImageNet-standard transforms: resize to 256, center crop to 224, normalize
- Add training augmentations: random horizontal flip, color jitter, random rotation
- Split into train/validation/test sets

### Task 2: Set Up Transfer Learning
- Load pretrained ResNet-18 with ImageNet weights
- Freeze all layers except the final classifier
- Replace the final fully connected layer for your number of classes
- Print how many parameters are trainable vs. frozen

### Task 3: Train (Feature Extraction)
- Train only the new classifier head (frozen backbone)
- Use Adam optimizer with lr=0.001
- Train for 10 epochs and track validation accuracy

### Task 4: Fine-Tune
- Unfreeze the last residual block (layer4)
- Use different learning rates: 1e-4 for pretrained layers, 1e-3 for new layers
- Apply cosine annealing learning rate schedule
- Train for 20 more epochs with early stopping (patience=5)

### Task 5: Compare with Training from Scratch
- Train an identical architecture from random initialization
- Use the same hyperparameters and training duration
- Compare final accuracy to show the benefit of transfer learning

## Expected Output

```
Feature Extraction (10 epochs):
  Trainable params: 51,400 / 11,228,100 (0.5%)
  Best Val Accuracy: 78.2%

Fine-Tuning (20 epochs):
  Trainable params: 2,408,500 / 11,228,100 (21.4%)
  Best Val Accuracy: 91.5%

From Scratch (30 epochs):
  Best Val Accuracy: 62.3%
```

## Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- Matplotlib

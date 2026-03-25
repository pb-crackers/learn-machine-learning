"""
Lab 5.2 Starter — Model Building & Training
==============================================
Follow the numbered steps. This script assumes you completed Lab 5.1
and have processed data saved in data/processed/.

Usage:
    python starter.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, mean_absolute_error, r2_score,
)

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# STEP 1: Load Processed Data
# =============================================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_val = pd.read_csv("data/processed/X_val.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_val = pd.read_csv("data/processed/y_val.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Target classes: {y_train.nunique()}")

# Set task type: "classification" or "regression"
TASK = "classification"  # Change to "regression" if appropriate


# =============================================================================
# STEP 2: Results Tracker
# =============================================================================

results = {"model": [], "val_metric": [], "notes": []}


def log_result(model_name, metric_value, notes=""):
    """Log a model result for later comparison."""
    results["model"].append(model_name)
    results["val_metric"].append(round(metric_value, 4))
    results["notes"].append(notes)
    print(f"  Logged: {model_name} -> {metric_value:.4f}")


# =============================================================================
# STEP 3: Classical ML Baselines
# =============================================================================

print("\n" + "=" * 60)
print("BASELINE MODELS")
print("=" * 60)

if TASK == "classification":
    # --- Logistic Regression ---
    print("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_val_pred_lr = lr_model.predict(X_val)

    print(classification_report(y_val, y_val_pred_lr))
    lr_f1 = f1_score(y_val, y_val_pred_lr, average="weighted")
    log_result("LogisticRegression", lr_f1)

    # --- Random Forest ---
    print("\n--- Random Forest ---")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_val_pred_rf = rf_model.predict(X_val)

    print(classification_report(y_val, y_val_pred_rf))
    rf_f1 = f1_score(y_val, y_val_pred_rf, average="weighted")
    log_result("RandomForest", rf_f1)

else:
    # --- Linear Regression ---
    print("\n--- Linear Regression ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_val_pred_lr = lr_model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_lr))
    print(f"RMSE: {rmse:.4f}, MAE: {mean_absolute_error(y_val, y_val_pred_lr):.4f}, "
          f"R2: {r2_score(y_val, y_val_pred_lr):.4f}")
    log_result("LinearRegression", rmse, "lower is better")

    # --- Random Forest ---
    print("\n--- Random Forest ---")
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_val_pred_rf = rf_model.predict(X_val)

    rmse_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))
    print(f"RMSE: {rmse_rf:.4f}, MAE: {mean_absolute_error(y_val, y_val_pred_rf):.4f}, "
          f"R2: {r2_score(y_val, y_val_pred_rf):.4f}")
    log_result("RandomForest", rmse_rf, "lower is better")

# Save baseline model for later use in Phase 3
joblib.dump(rf_model, "models/rf_baseline.joblib")


# =============================================================================
# STEP 4: Neural Network Definition
# =============================================================================

class TabularNet(nn.Module):
    """Feedforward neural network for tabular data."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# STEP 5: Prepare Data for PyTorch
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Convert to tensors
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32)

if TASK == "classification":
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
else:
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


# =============================================================================
# STEP 6: Training and Evaluation Functions
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, loader, criterion, device):
    """Evaluate model and return loss, predictions, and targets."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            n_batches += 1

            if TASK == "classification":
                preds = torch.sigmoid(outputs).cpu().numpy()
            else:
                preds = outputs.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / n_batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return avg_loss, all_preds, all_targets


# =============================================================================
# STEP 7: Train the Neural Network
# =============================================================================

print("\n" + "=" * 60)
print("NEURAL NETWORK TRAINING")
print("=" * 60)

input_dim = X_train.shape[1]
hidden_dims = [128, 64, 32]
output_dim = 1
dropout_rate = 0.3
learning_rate = 1e-3

model = TabularNet(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
print(f"\nArchitecture: {hidden_dims}, dropout={dropout_rate}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

if TASK == "classification":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)

# Training loop with early stopping
NUM_EPOCHS = 100
PATIENCE = 10
best_val_loss = float("inf")
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "models/best_model.pt")
    else:
        patience_counter += 1

    if (epoch + 1) % 5 == 0 or patience_counter == 0:
        marker = "*" if patience_counter == 0 else ""
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} {marker}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Load best model
model.load_state_dict(torch.load("models/best_model.pt"))


# =============================================================================
# STEP 8: Plot Learning Curves
# =============================================================================

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plots/learning_curves.png", dpi=150)
plt.close()
print("\nSaved: plots/learning_curves.png")


# =============================================================================
# STEP 9: Evaluate Neural Network on Validation Set
# =============================================================================

val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)

if TASK == "classification":
    val_pred_labels = (val_preds > 0.5).astype(int).flatten()
    val_true_labels = val_targets.flatten().astype(int)

    print("\n=== Neural Network (Best Checkpoint) ===")
    print(classification_report(val_true_labels, val_pred_labels))

    nn_f1 = f1_score(val_true_labels, val_pred_labels, average="weighted")
    log_result("NeuralNetwork_v1", nn_f1,
               f"hidden={hidden_dims}, dropout={dropout_rate}, lr={learning_rate}")
else:
    val_preds_flat = val_preds.flatten()
    val_targets_flat = val_targets.flatten()
    nn_rmse = np.sqrt(mean_squared_error(val_targets_flat, val_preds_flat))
    print(f"\n=== Neural Network (Best Checkpoint) ===")
    print(f"RMSE: {nn_rmse:.4f}")
    log_result("NeuralNetwork_v1", nn_rmse,
               f"hidden={hidden_dims}, dropout={dropout_rate}, lr={learning_rate}")


# =============================================================================
# STEP 10: Experimentation
# =============================================================================
# Run at least 3 experiments, changing ONE variable at a time.
# Uncomment and modify the configs below.

print("\n" + "=" * 60)
print("EXPERIMENTS")
print("=" * 60)

experiments = [
    {"hidden_dims": [64, 32], "dropout": 0.2, "lr": 1e-3, "name": "smaller_net"},
    {"hidden_dims": [256, 128, 64], "dropout": 0.3, "lr": 1e-3, "name": "wider_net"},
    {"hidden_dims": [128, 64, 32], "dropout": 0.5, "lr": 1e-3, "name": "more_dropout"},
    # Add your own experiments here:
    # {"hidden_dims": [128, 64, 32], "dropout": 0.3, "lr": 1e-4, "name": "lower_lr"},
]

for config in experiments:
    print(f"\n--- {config['name']} ---")

    exp_model = TabularNet(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        output_dim=output_dim,
        dropout_rate=config["dropout"],
    ).to(device)

    exp_optimizer = torch.optim.Adam(
        exp_model.parameters(), lr=config["lr"], weight_decay=1e-4
    )

    # Quick train (30 epochs, no early stopping)
    for ep in range(30):
        train_one_epoch(exp_model, train_loader, criterion, exp_optimizer, device)

    _, exp_preds, exp_targets = evaluate(exp_model, val_loader, criterion, device)

    if TASK == "classification":
        exp_labels = (exp_preds > 0.5).astype(int).flatten()
        exp_true = exp_targets.flatten().astype(int)
        metric = f1_score(exp_true, exp_labels, average="weighted")
    else:
        metric = np.sqrt(mean_squared_error(exp_targets.flatten(), exp_preds.flatten()))

    log_result(config["name"], metric, str(config))


# =============================================================================
# STEP 11: Model Comparison
# =============================================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)
ascending = TASK == "regression"  # lower is better for RMSE
results_df = results_df.sort_values("val_metric", ascending=ascending)
print(results_df.to_string(index=False))
results_df.to_csv("results/model_comparison.csv", index=False)
print("\nSaved: results/model_comparison.csv")


# =============================================================================
# STEP 12: Final Test Set Evaluation
# =============================================================================
# Only run this ONCE after choosing your best model.

print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)

X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load best model
model.load_state_dict(torch.load("models/best_model.pt"))
test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

if TASK == "classification":
    test_pred_labels = (test_preds > 0.5).astype(int).flatten()
    test_true_labels = test_targets.flatten().astype(int)

    print(classification_report(test_true_labels, test_pred_labels))
    test_f1 = f1_score(test_true_labels, test_pred_labels, average="weighted")
    print(f"Validation F1: {nn_f1:.4f}")
    print(f"Test F1:       {test_f1:.4f}")
    print(f"Difference:    {abs(nn_f1 - test_f1):.4f}")
else:
    test_rmse = np.sqrt(mean_squared_error(test_targets.flatten(), test_preds.flatten()))
    print(f"Validation RMSE: {nn_rmse:.4f}")
    print(f"Test RMSE:       {test_rmse:.4f}")
    print(f"Difference:      {abs(nn_rmse - test_rmse):.4f}")


# =============================================================================
# STEP 13: Save Final Model
# =============================================================================

torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": input_dim,
    "hidden_dims": hidden_dims,
    "output_dim": output_dim,
    "dropout_rate": dropout_rate,
}, "models/final_model.pt")

print("\nSaved: models/final_model.pt")
print("Done. Review the deliverables checklist in the lesson before moving to Phase 3.")

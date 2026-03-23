"""
MamaGuard — Training Loop
Trains the Mamba3 model with class-weighted loss and LR scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from src.model import MamaGuardMamba3
from src.data_pipeline import get_dataloaders

MODEL_SAVE_PATH  = "models/mamaguard_mamba3.pt"
SCALER_SAVE_PATH = "models/scaler.pkl"
os.makedirs("models", exist_ok=True)


def compute_class_weights(train_loader):
    """Compute inverse-frequency class weights with 3× boost for high-risk."""
    all_labels = []
    for _, y, _ in train_loader:
        all_labels.extend(y.numpy())

    counts = np.bincount(all_labels, minlength=3)
    total  = counts.sum()

    weights = total / (3 * counts + 1e-6)
    weights[2] *= 3.0   # high-risk class boost
    print(f"Class weights: LOW={weights[0]:.2f}, MID={weights[1]:.2f}, HIGH={weights[2]:.2f}")
    return torch.tensor(weights, dtype=torch.float32)


def train(
    csv_path:   str   = "data/maternal_health.csv",
    epochs:     int   = 50,
    batch_size: int   = 32,
    lr:         float = 1e-3,
    device:     str   = None
):
    """Full training loop with validation and best-model checkpointing."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Load data
    train_loader, val_loader, scaler = get_dataloaders(csv_path, batch_size)

    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

    # Build model
    model = MamaGuardMamba3(
        input_dim=6, d_model=64, n_layers=4, n_classes=3, d_state=32
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss function with class weights
    class_weights = compute_class_weights(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # Training loop
    best_val_loss = float("inf")
    best_val_acc  = 0.0

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += len(y_batch)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits  = model(X_batch)
                loss    = criterion(logits, y_batch)
                val_loss   += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y_batch).sum().item()
                val_total   += len(y_batch)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        train_acc      = train_correct / train_total
        val_acc        = val_correct   / val_total

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.3f}"
        )

        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  * Best model saved (val_acc={val_acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    return model


if __name__ == "__main__":
    train()
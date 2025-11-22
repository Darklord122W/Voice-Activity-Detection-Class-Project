# ============================================================
# experiments.py
# ============================================================
# Assignment: Implement the full training, validation, and evaluation
# loops for Voice Activity Detection (VAD) models.
#
# You must implement all parts except:
#   ✅ The alignment snippet (provided below)
#   ✅ The run_experiment() function (provided and complete)
#
# Use the alignment snippet exactly where specified to ensure
# model outputs and labels have matching time dimensions.
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
from dataloader import create_vad_dataloaders
from models import build_model,init_weights_xavier
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Device Setup
# ------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"))
print(f"✅ Using device: {device}")


# ============================================================
# Alignment Snippet (Use This As-Is)
# ============================================================
"""
Use the following snippet whenever you compare model outputs and labels
to compute the loss or predictions. It ensures both tensors have the
same time dimension length (T).

Place it right after obtaining `logits = model(xb)` and before computing the loss.

Example:
    logits = model(xb)
    # Align lengths (insert here)
    B, Tm, C = logits.shape
    Ty = yb.size(1)
    min_T = min(Tm, Ty)
    logits = logits[:, :min_T, :]
    yb = yb[:, :min_T]
"""

# ============================================================
# Training and Validation
# ============================================================
def train_model(model, train_dl, val_dl, epochs=20, lr=1e-3, patience=3):
    """
    Implement the full training loop with validation and early stopping.
    """
    # TODO: define loss function
    # TODO: define optimizer
    # TODO: define learning rate scheduler
    # TODO: initialize early stopping variables and tracking lists
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )        
    best_loss = float("inf")
    wait = 0  # patience counter

    train_losses = []
    val_losses = []
    # Apply Xavier init to all layers
    model.apply(init_weights_xavier)
    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        # TODO: iterate over batches
        # TODO: move data to device
        # TODO: forward pass (use the alignment snippet before computing loss)
        # TODO: compute loss and backpropagate
        # TODO: apply gradient clipping
        # TODO: optimizer step and track loss
        model.train()
        running_train_loss = 0.0

        # -------------------
        # Training batches
        # -------------------
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            # ----------------------------
            # Alignment snippet
            # ----------------------------
            B, Tm, C = logits.shape
            Ty = yb.size(1)
            min_T = min(Tm, Ty)
            logits = logits[:, :min_T, :]
            yb = yb[:, :min_T]

            # Fuse time into batch for CE loss
            loss = criterion(logits.reshape(-1, C), yb.reshape(-1))

            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_dl)
        train_losses.append(avg_train_loss)



        # ---- Validation ----
        model.eval()
        # TODO: evaluate on validation set using same alignment snippet
        # TODO: collect predictions and compute validation loss
        # TODO: compute accuracy, precision, recall
        val_running_loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)

                logits = model(xb)

                # Alignment snippet
                B, Tm, C = logits.shape
                Ty = yb.size(1)
                min_T = min(Tm, Ty)
                logits = logits[:, :min_T, :]
                yb = yb[:, :min_T]

                loss = criterion(logits.reshape(-1, C), yb.reshape(-1))
                val_running_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_labels.append(yb.cpu().numpy().reshape(-1))

        avg_val_loss = val_running_loss / len(val_dl)
        val_losses.append(avg_val_loss)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        val_acc = (all_preds == all_labels).mean()
        val_prec = precision_score(all_labels, all_preds, zero_division=0)
        val_rec = recall_score(all_labels, all_preds, zero_division=0)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f}"
        )

        print(f"Epoch {epoch:02d} | Train Loss: ... | Val Loss: ... | Val Acc: ...")

        # TODO: scheduler step
        scheduler.step(avg_val_loss)

        # TODO: implement early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print("⛔ Early stopping triggered!")
                break    
    # TODO: load best model before returning
    model.load_state_dict(best_state)
    # TODO: return model and training history (train_loss, val_loss)

    return model, {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
    # raise NotImplementedError


# ============================================================
# Evaluation
# ============================================================
def evaluate_model(model, test_dl):
    """
    Evaluate the model on the test set.

    Steps:
      - Set model to evaluation mode
      - Iterate through test batches
      - Use alignment snippet before comparing predictions and labels
      - Compute accuracy, precision, and recall
    """
    # TODO: implement evaluation loop
    # TODO: collect predictions and labels
    # TODO: compute metrics
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)

            # Alignment snippet
            B, Tm, C = logits.shape
            Ty = yb.size(1)
            min_T = min(Tm, Ty)
            logits = logits[:, :min_T, :]
            yb = yb[:, :min_T]

            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_labels.append(yb.cpu().numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = (all_preds == all_labels).mean()
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    return acc, prec, rec

    # raise NotImplementedError


# ============================================================
# Experiment Runner (Provided)
# ============================================================
def run_experiment(model_name, n_features=40, hidden_size=128, layers=2, seq_len=25, snr_db=-10):
    print(f"\n🚀 Running {model_name.upper()} model (seq_len={seq_len})...")

    train_dl, val_dl, test_dl = create_vad_dataloaders(
        data_root="SpeechCommands/speech",
        noise_train="SpeechCommands/noise",
        noise_val="SpeechCommands/noise",
        noise_test="SpeechCommands/noise",
        batch_size=64,
        num_workers=2,
        seq_len=seq_len,
    )

    model = build_model(model_name, n_features=n_features, hidden_size=hidden_size, num_layers=layers)
    model = model.to(device)

    model, hist = train_model(model, train_dl, val_dl, lr=1e-3)

    plt.figure(figsize=(6, 4))
    plt.plot(hist["train_loss"], label="Train")
    plt.plot(hist["val_loss"], label="Val")
    plt.title(f"{model_name.upper()} Learning Curve (seq={seq_len})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    acc, prec, rec = evaluate_model(model, test_dl)
    print(f"✅ Test → Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}")
    return acc, prec, rec

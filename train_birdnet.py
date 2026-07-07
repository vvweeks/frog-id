"""
train_birdnet.py - Training loop for the BirdNET-embedding classifier
head. Requires build_embedding_cache() to have been run first.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    SPECIES_MAP, NUM_CLASSES, BATCH_SIZE, WEIGHT_DECAY, CHECKPOINT_DIR,
)
from features.birdnet_embeddings import BirdNETEmbeddingDataset
from models.birdnet_head import get_birdnet_classifier_head
from train_utils import compute_class_weights


def train_birdnet_classifier(epochs=50, lr=1e-3, run_id=None, early_stop_patience=5, head_type="mlp"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Splits come from the manifest (recordist-disjoint); no runtime split.
    train_dataset = BirdNETEmbeddingDataset("train", SPECIES_MAP)
    val_dataset = BirdNETEmbeddingDataset("val", SPECIES_MAP)
    test_dataset = BirdNETEmbeddingDataset("test", SPECIES_MAP)

    if len(train_dataset) == 0:
        raise RuntimeError("No cached train embeddings found - run build_embeddings "
                           "and make_split first.")

    # Standardize embeddings using TRAIN stats only (fit on train, apply to
    # all - no val/test leakage). Helps a linear/MLP head on frozen features.
    train_embs = np.stack([np.load(p) for p in train_dataset.embedding_paths]).astype(np.float32)
    emb_mean = train_embs.mean(axis=0)
    emb_std = train_embs.std(axis=0) + 1e-6
    for ds in (train_dataset, val_dataset, test_dataset):
        ds.set_normalization(emb_mean, emb_std)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    embedding_dim = train_embs.shape[1]
    model = get_birdnet_classifier_head(embedding_dim=embedding_dim, head_type=head_type).to(device)
    class_weights = compute_class_weights(train_dataset.labels, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_acc, best_state_dict = 0.0, None
    best_val_loss, epochs_no_improve, stopped_early = float('inf'), 0, False

    print("--- Starting Training (BirdNET-embedding classifier) ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_losses.append(running_loss / len(train_dataset))

        model.eval()
        val_running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_losses.append(val_running_loss / len(val_dataset))
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        scheduler.step(val_losses[-1])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | "
                  f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping: val loss hasn't improved on best "
                      f"({best_val_loss:.4f}) for {early_stop_patience} "
                      f"consecutive epochs (stopped at epoch {epoch+1}).")
                stopped_early = True
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\nRestored best checkpoint (Val Acc: {best_val_acc:.2f}%)")

    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    final_test_acc = 100 * test_correct / test_total
    print(f"Final Test Accuracy (BirdNET-embedding classifier): {final_test_acc:.2f}%")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_name = f"{run_id}.pth" if run_id else "birdnet_head.pth"
    save_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint: {save_path}")

    return train_losses, val_losses, val_accuracies, final_test_acc, save_path, stopped_early

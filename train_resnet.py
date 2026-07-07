"""
train_resnet.py - Training loop for the ResNet18-on-spectrograms
pipeline. No module-level side effects - call train_model() explicitly.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    SPECIES_MAP, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    BACKBONE_LR_MULT, WEIGHT_DECAY, CHECKPOINT_DIR,
)
from features.spectrogram_dataset import FrogCallDataset
from models.resnet_transfer import get_frog_model, freeze_bn_stats


def train_model(run_id=None, early_stop_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Splits come from the manifest (recordist-disjoint), so val is
    # leakage-free from train - no runtime random split needed.
    train_dataset = FrogCallDataset("train", SPECIES_MAP)
    val_dataset = FrogCallDataset("val", SPECIES_MAP)
    test_dataset = FrogCallDataset("test", SPECIES_MAP)

    if len(train_dataset) == 0:
        raise RuntimeError("Training split is empty - run scripts.make_split first.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = get_frog_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': LEARNING_RATE * BACKBONE_LR_MULT},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_acc, best_state_dict = 0.0, None
    best_val_loss, epochs_no_improve, stopped_early = float('inf'), 0, False

    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        freeze_bn_stats(model)

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

        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {[f'{lr:.2e}' for lr in current_lrs]}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

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
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_name = f"{run_id}.pth" if run_id else "resnet18.pth"
    save_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint: {save_path}")

    return train_losses, val_losses, val_accuracies, final_test_acc, save_path, stopped_early

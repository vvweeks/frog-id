"""
train_resnet.py - Training loop for the ResNet18-on-spectrograms
pipeline. No module-level side effects - call train_model() explicitly.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from config import (
    TRAIN_DIR, TEST_DIR, SPECIES_MAP, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    VAL_SPLIT, BACKBONE_LR_MULT, WEIGHT_DECAY, RANDOM_SEED, DRIVE_SAVE_DIR,
)
from features.spectrogram_dataset import FrogCallDataset
from models.resnet_transfer import get_frog_model, freeze_bn_stats


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    train_transform_dataset = FrogCallDataset(TRAIN_DIR, SPECIES_MAP, is_train=True)
    val_transform_dataset = FrogCallDataset(TRAIN_DIR, SPECIES_MAP, is_train=False)
    # Both scan TRAIN_DIR independently via os.listdir(); force identical
    # ordering so train_idx/val_idx (computed once, below) select the same
    # files from each rather than relying on directory-listing order to
    # coincidentally match between the two separate scans.
    val_transform_dataset.file_paths = train_transform_dataset.file_paths
    val_transform_dataset.labels = train_transform_dataset.labels

    if len(train_transform_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    all_indices = list(range(len(train_transform_dataset)))
    all_labels = train_transform_dataset.labels
    train_idx, val_idx = train_test_split(
        all_indices, test_size=VAL_SPLIT, stratify=all_labels, random_state=RANDOM_SEED,
    )
    train_subset = Subset(train_transform_dataset, train_idx)
    val_subset = Subset(val_transform_dataset, val_idx)
    test_dataset = FrogCallDataset(TEST_DIR, SPECIES_MAP, is_train=False)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_dataset)}")

    model = get_frog_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': LEARNING_RATE * BACKBONE_LR_MULT},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_acc, best_state_dict = 0.0, None

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
        train_losses.append(running_loss / len(train_subset))

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

        val_losses.append(val_running_loss / len(val_subset))
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

    save_path = os.path.join(DRIVE_SAVE_DIR, "ct_frog_resnet18_v5a.pth")
    torch.save(model.state_dict(), save_path)

    return train_losses, val_losses, val_accuracies, final_test_acc

"""
train_utils.py - Small shared helpers for the training loops.
"""
import numpy as np
import torch


def compute_class_weights(labels, num_classes):
    """Inverse-frequency ('balanced') class weights for CrossEntropyLoss,
    so under-represented species aren't drowned out by common ones.

    Classes with no training samples get weight 1.0 (harmless - they
    contribute nothing to the loss anyway). Returns a float tensor of
    length num_classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float32)
    present = counts > 0
    n_present = int(present.sum())
    total = counts.sum()
    if n_present > 0:
        weights[present] = (total / (n_present * counts[present])).astype(np.float32)
    return torch.from_numpy(weights)

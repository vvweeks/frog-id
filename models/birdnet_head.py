"""
models/birdnet_head.py - Lightweight classifier head trained on top of
frozen BirdNET embeddings.
"""
import torch.nn as nn

from config import NUM_CLASSES


def get_birdnet_classifier_head(embedding_dim=1024, hidden_dim=256, dropout_p=0.4):
    """embedding_dim should match features.birdnet_embeddings.EMBEDDING_DIM
    from your sanity check (1024 for BirdNET v2.4)."""
    return nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(embedding_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_p),
        nn.Linear(hidden_dim, NUM_CLASSES),
    )

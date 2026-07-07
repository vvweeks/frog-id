"""
models/birdnet_head.py - Lightweight classifier head trained on top of
frozen BirdNET embeddings.
"""
import torch.nn as nn

from config import NUM_CLASSES


def get_birdnet_classifier_head(embedding_dim=1024, hidden_dim=256, dropout_p=0.4, head_type="mlp"):
    """Classifier head on frozen BirdNET embeddings. embedding_dim should
    match features.birdnet_embeddings.EMBEDDING_DIM (1024 for BirdNET v2.4).

    head_type:
      "mlp"    - 2-layer MLP with dropout (more capacity).
      "linear" - a plain linear probe (dropout + one Linear). Often wins on
                 small data, as in the Ribbit frog project - less to overfit.
    """
    if head_type == "linear":
        return nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(embedding_dim, NUM_CLASSES),
        )
    return nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(embedding_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_p),
        nn.Linear(hidden_dim, NUM_CLASSES),
    )

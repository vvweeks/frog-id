"""
scripts/build_embeddings.py - Entry point for BirdNET embedding
extraction/caching. Run explicitly: python -m scripts.build_embeddings
Loads the BirdNET model and runs the sanity check ONLY when invoked
directly - never on import.
"""
import os
from config import TRAIN_DIR, TEST_DIR, SPECIES_MAP
from features.birdnet_embeddings import (
    sanity_check_encode, build_embedding_cache,
)

if __name__ == "__main__":
    sample_file = None
    for class_name in SPECIES_MAP:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.isdir(class_dir) and os.listdir(class_dir):
            sample_file = os.path.join(class_dir, os.listdir(class_dir)[0])
            break

    if sample_file:
        sanity_check_encode(sample_file)
    else:
        raise RuntimeError("No training audio found - run download_data.py first.")

    build_embedding_cache(TRAIN_DIR, SPECIES_MAP)
    build_embedding_cache(TEST_DIR, SPECIES_MAP)

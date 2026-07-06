"""
data/blocklist.py - Cleanup for files already on disk that match
CORRUPT_FILE_IDS (the downloaders themselves already skip these going
forward; this is for files downloaded before the blocklist existed).
"""
import os
from config import TRAIN_DIR, TEST_DIR, SPECIES_MAP, EMBEDDING_CACHE_DIR, CORRUPT_FILE_IDS


def remove_known_corrupt_files():
    """Deletes any files matching CORRUPT_FILE_IDS from TRAIN_DIR/TEST_DIR,
    plus their cached BirdNET embeddings if present."""
    removed = 0
    for base_dir in [TRAIN_DIR, TEST_DIR]:
        for class_name in SPECIES_MAP:
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                stem = os.path.splitext(fname)[0]
                if stem in CORRUPT_FILE_IDS:
                    file_path = os.path.join(class_dir, fname)
                    os.remove(file_path)
                    removed += 1
                    print(f"  Removed: {file_path}")

                    cache_path = os.path.join(EMBEDDING_CACHE_DIR, f"{class_name}__{fname}.npy")
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                        print(f"    Removed cached embedding: {cache_path}")

    print(f"\nRemoved {removed} known-corrupt file(s).")

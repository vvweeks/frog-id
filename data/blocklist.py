"""
data/blocklist.py - Cleanup for files already on disk that match
CORRUPT_FILE_IDS (the downloaders skip these going forward; this removes
any that were fetched before the blocklist existed), along with their
manifest rows and any cached BirdNET segment embeddings.
"""
import os

from config import DATA_DIR, SPECIES_MAP, EMBEDDING_CACHE_DIR, CORRUPT_FILE_IDS, species_dir
from data.manifest import Manifest


def remove_known_corrupt_files(manifest=None):
    """Deletes audio matching CORRUPT_FILE_IDS from the flat per-species
    dirs, drops their manifest rows, and removes cached segment embeddings."""
    own_manifest = manifest is None
    if manifest is None:
        manifest = Manifest.load()

    removed = 0
    for class_name in SPECIES_MAP:
        class_dir = species_dir(class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in list(os.listdir(class_dir)):
            stem = os.path.splitext(fname)[0]
            if stem not in CORRUPT_FILE_IDS:
                continue

            os.remove(os.path.join(class_dir, fname))
            removed += 1
            print(f"  Removed: {os.path.join(class_dir, fname)}")

            manifest.drop(class_name, fname)

            # Remove any cached segment embeddings for this file.
            prefix = f"{class_name}__{fname}__seg"
            if os.path.isdir(EMBEDDING_CACHE_DIR):
                for cached in os.listdir(EMBEDDING_CACHE_DIR):
                    if cached.startswith(prefix):
                        os.remove(os.path.join(EMBEDDING_CACHE_DIR, cached))
                        print(f"    Removed cached embedding: {cached}")

    if own_manifest:
        manifest.save()
    print(f"\nRemoved {removed} known-corrupt file(s).")

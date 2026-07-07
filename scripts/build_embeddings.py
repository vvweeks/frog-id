"""
scripts/build_embeddings.py - Entry point for BirdNET embedding
extraction/caching. Run explicitly: python -m scripts.build_embeddings
Loads the BirdNET model and runs the sanity check ONLY when invoked
directly - never on import. Caches embeddings for every file in the
manifest (split-agnostic - the split is applied later at train time).
"""
import os
from config import SPECIES_MAP, species_dir
from data.manifest import Manifest
from features.birdnet_embeddings import sanity_check_encode, build_embedding_cache

if __name__ == "__main__":
    manifest = Manifest.load()
    rows = manifest.rows()
    if not rows:
        raise RuntimeError("Manifest is empty - run `python -m scripts.download_data` first.")

    sample_file = None
    for r in rows:
        candidate = os.path.join(species_dir(r["species"]), r["filename"])
        if os.path.exists(candidate):
            sample_file = candidate
            break
    if sample_file is None:
        raise RuntimeError("No audio files found on disk for manifest entries.")

    sanity_check_encode(sample_file)
    build_embedding_cache(SPECIES_MAP, manifest=manifest)

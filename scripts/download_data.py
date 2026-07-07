"""
scripts/download_data.py - Entry point for dataset acquisition.
Run explicitly: python -m scripts.download_data
This is the ONLY place download side effects happen.

Both downloaders share a single manifest (so counts and recordist/quality
metadata stay consistent), which is saved once at the end. Assigning
train/val/test is a separate step - run `python -m scripts.make_split`
afterward.
"""
from data.manifest import Manifest
from data.xeno_canto import download_xeno_canto_data
from data.inaturalist import fill_gaps_with_inaturalist
from data.blocklist import remove_known_corrupt_files
from data.inventory import print_dataset_inventory

if __name__ == "__main__":
    manifest = Manifest.load()
    remove_known_corrupt_files(manifest=manifest)
    download_xeno_canto_data(manifest=manifest)
    fill_gaps_with_inaturalist(manifest=manifest)
    manifest.save()
    print_dataset_inventory(manifest=manifest)
    print("\nNext: run `python -m scripts.make_split` to assign train/val/test.")

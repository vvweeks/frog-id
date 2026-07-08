"""
scripts/download_data.py - Entry point for dataset acquisition.
Run explicitly:
  python -m scripts.download_data                # both sources (default)
  python -m scripts.download_data --source xc    # Xeno-canto only
  python -m scripts.download_data --source inat  # iNaturalist only
This is the ONLY place download side effects happen.

Both downloaders share a single manifest (so counts and recordist/quality
metadata stay consistent), which is saved once at the end. Assigning
train/val/test is a separate step - run `python -m scripts.make_split`
afterward.
"""
import argparse

from data.manifest import Manifest
from data.xeno_canto import download_xeno_canto_data
from data.inaturalist import fill_gaps_with_inaturalist
from data.blocklist import remove_known_corrupt_files
from data.inventory import print_dataset_inventory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["both", "xc", "inat"], default="both",
                        help="Which source(s) to pull from (default both). "
                             "Use 'xc' to add Xeno-canto only, without iNaturalist gap-fill.")
    args = parser.parse_args()

    manifest = Manifest.load()
    remove_known_corrupt_files(manifest=manifest)
    if args.source in ("both", "xc"):
        download_xeno_canto_data(manifest=manifest)
    if args.source in ("both", "inat"):
        fill_gaps_with_inaturalist(manifest=manifest)
    manifest.save()
    print_dataset_inventory(manifest=manifest)
    print("\nNext: run `python -m scripts.make_split` to assign train/val/test.")

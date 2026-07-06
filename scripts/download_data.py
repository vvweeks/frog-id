"""
scripts/download_data.py - Entry point for dataset acquisition.
Run explicitly: python -m scripts.download_data
This is the ONLY place download side effects happen.
"""
from data.xeno_canto import download_xeno_canto_data
from data.inaturalist import fill_gaps_with_inaturalist
from data.blocklist import remove_known_corrupt_files
from data.inventory import print_dataset_inventory

if __name__ == "__main__":
    remove_known_corrupt_files()
    download_xeno_canto_data()
    fill_gaps_with_inaturalist()
    print_dataset_inventory()

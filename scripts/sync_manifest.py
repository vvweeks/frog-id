"""
scripts/sync_manifest.py - Register on-disk audio files that aren't in the
manifest yet, so nothing on disk gets ignored by training.

This covers Xeno-canto files downloaded before the manifest existed (or when
the XC API was unavailable to back-fill their recordist/quality). Source is
inferred from the filename (inat_* -> iNaturalist, else Xeno-canto).
Recordist and quality are left blank - we have no API listing to fill them -
so make_split treats each such file as its own group (freely splittable).

Once the XC key works again, re-running scripts.download_data will back-fill
real recordist metadata for these files and enable a recordist-disjoint split.

Run:  python -m scripts.sync_manifest
Then: python -m scripts.make_split   (to assign the newly-added files a split)
"""
import os

from config import SPECIES_MAP, species_dir
from data.manifest import Manifest


def sync_manifest():
    manifest = Manifest.load()
    added = 0
    by_source = {"xeno_canto": 0, "inat": 0}
    for species in SPECIES_MAP:
        d = species_dir(species)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith((".mp3", ".wav")):
                continue
            if manifest.has(species, fname):
                continue
            source = "inat" if fname.startswith("inat_") else "xeno_canto"
            manifest.upsert(species, fname, source, "", "")  # unknown recordist/quality
            by_source[source] += 1
            added += 1

    manifest.save()
    print(f"Registered {added} previously-unmanifested files "
          f"(Xeno-canto: {by_source['xeno_canto']}, iNaturalist: {by_source['inat']}).")
    if by_source["xeno_canto"]:
        print("Note: these have no recordist yet, so they aren't recordist-disjoint in "
              "the split. Re-run scripts.download_data once the XC key works to back-fill it.")
    print("Next: run `python -m scripts.make_split` to assign train/val/test.")


if __name__ == "__main__":
    sync_manifest()

"""
data/manifest.py - Read/write helpers for the dataset manifest: the
single source of truth mapping each audio file to its species, source,
recordist, quality, and (once assigned) train/val/test split.

Audio lives flat at DATA_DIR/<species>/<filename>. The manifest is keyed
by (species, filename). Recordist is captured so the split can be made
recordist-disjoint - no recordist appears in more than one split - which
stops the model from "recognizing the recording" (same mic, site, or
individual animal) instead of learning the species.
"""
import csv
import os

from config import MANIFEST_PATH

FIELDNAMES = ["species", "filename", "source", "recordist", "quality", "split"]


class Manifest:
    def __init__(self):
        # (species, filename) -> row dict
        self._rows = {}

    @classmethod
    def load(cls):
        """Loads the manifest if it exists, else returns an empty one."""
        m = cls()
        if os.path.exists(MANIFEST_PATH):
            with open(MANIFEST_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    m._rows[(row["species"], row["filename"])] = row
        return m

    def upsert(self, species, filename, source, recordist, quality):
        """Add or update a file's metadata. Preserves any existing split
        assignment, so re-running downloads never wipes an assigned split."""
        key = (species, filename)
        existing = self._rows.get(key, {})
        self._rows[key] = {
            "species": species,
            "filename": filename,
            "source": source,
            "recordist": (recordist or "").strip(),
            "quality": (quality or "").strip(),
            "split": existing.get("split", ""),
        }

    def has(self, species, filename):
        return (species, filename) in self._rows

    def set_split(self, species, filename, split):
        self._rows[(species, filename)]["split"] = split

    def drop(self, species, filename):
        self._rows.pop((species, filename), None)

    def count_for_species(self, species):
        return sum(1 for sp, _ in self._rows if sp == species)

    def for_split(self, split):
        """All rows assigned to a given split ('train'/'val'/'test')."""
        return [r for r in self.rows() if r.get("split") == split]

    def rows(self):
        """All rows, sorted by (species, filename) for stable output."""
        return [self._rows[k] for k in sorted(self._rows)]

    def save(self):
        os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
        with open(MANIFEST_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in self.rows():
                writer.writerow(row)
        print(f"Saved manifest ({len(self._rows)} files): {MANIFEST_PATH}")

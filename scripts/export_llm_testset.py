"""
scripts/export_llm_testset.py - Package the held-out TEST recordings into a
self-contained folder you can hand to an LLM for a head-to-head comparison
against the trained model.

Exports the WHOLE test recordings (same files the model's test split uses),
with anonymized names so neither the filename nor a folder path leaks the
species. Produces:

  llm_testset/
    recordings/        test_0001.mp3, test_0002.mp3, ...  (upload these)
    prompt.txt         open-ended instructions for the LLM (give this)
    answer_key.csv     clip_id, true_common_name, true_scientific_name (DO NOT give the LLM)
    mapping.csv        clip_id -> original species/filename (your reference)

Run (after make_split):  python -m scripts.export_llm_testset
"""
import csv
import os
import shutil

from config import DATA_DIR, PROJECT_DIR, SPECIES_MAP, species_dir
from data.manifest import Manifest

EXPORT_DIR = os.path.join(PROJECT_DIR, "llm_testset")
RECORDINGS_DIR = os.path.join(EXPORT_DIR, "recordings")

PROMPT = """\
You are an expert field herpetologist identifying North American frog and toad
species from audio recordings.

You will receive a set of audio files named test_0001, test_0002, and so on.
For EACH recording, listen and identify the single most likely frog or toad
species that is calling.

Respond with one line per recording, in CSV format, nothing else:

    clip_id,species

- clip_id is the file's name without extension (e.g. test_0001).
- species is your single best guess, as a common name (e.g. "Spring Peeper")
  or a scientific name (e.g. "Pseudacris crucifer"). Give exactly one species
  per clip. If unsure, still give your single best guess.

Example response:
    test_0001,American Bullfrog
    test_0002,Spring Peeper
"""


def export():
    manifest = Manifest.load()
    test_rows = manifest.for_split("test")
    if not test_rows:
        raise RuntimeError("No test files in the manifest - run scripts.make_split first.")

    # Deterministic order so clip ids are stable across re-runs.
    test_rows.sort(key=lambda r: (r["species"], r["filename"]))

    if os.path.isdir(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    answer_key, mapping = [], []
    exported = 0
    for i, r in enumerate(test_rows, start=1):
        species = r["species"]
        src = os.path.join(species_dir(species), r["filename"])
        if not os.path.exists(src):
            print(f"  [skip - missing] {src}")
            continue
        clip_id = f"test_{i:04d}"
        ext = os.path.splitext(r["filename"])[1] or ".mp3"
        shutil.copy(src, os.path.join(RECORDINGS_DIR, f"{clip_id}{ext}"))

        common = species.replace("_", " ")
        answer_key.append({"clip_id": clip_id, "true_common_name": common,
                           "true_scientific_name": SPECIES_MAP[species]})
        mapping.append({"clip_id": clip_id, "species": species, "original_filename": r["filename"]})
        exported += 1

    with open(os.path.join(EXPORT_DIR, "answer_key.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "true_common_name", "true_scientific_name"])
        w.writeheader()
        w.writerows(answer_key)
    with open(os.path.join(EXPORT_DIR, "mapping.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "species", "original_filename"])
        w.writeheader()
        w.writerows(mapping)
    with open(os.path.join(EXPORT_DIR, "prompt.txt"), "w") as f:
        f.write(PROMPT)

    total_mb = sum(
        os.path.getsize(os.path.join(RECORDINGS_DIR, f)) for f in os.listdir(RECORDINGS_DIR)
    ) / (1024 * 1024)
    print(f"Exported {exported} test recordings to {RECORDINGS_DIR} ({total_mb:.0f} MB total).")
    print(f"  Upload:  recordings/ + prompt.txt")
    print(f"  Keep:    answer_key.csv (ground truth), mapping.csv (reference)")
    print(f"  Score:   python -m scripts.score_llm <llm_responses.csv>")


if __name__ == "__main__":
    export()

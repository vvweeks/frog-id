"""
scripts/make_split.py - Assigns each file in the manifest to train/val/test.

The split is recordist-DISJOINT: every recording by a given recordist lands
entirely in one split, so the model can't "recognize the recording" (same
mic, site, or individual animal) across train and test and thereby inflate
its scores. Within that hard constraint it's stratified by species as well
as possible; with few recordists per species, perfect balance isn't always
achievable, so the per-class breakdown is printed for inspection.

Files with no known recordist each become their own singleton group, so
missing metadata never forces two unrelated files onto the same side.

Run:  python -m scripts.make_split [--folds 7] [--seed 42]
  --folds N  ->  roughly 1/N to test, 1/N to val, the rest to train
                 (default 7 ~= 14% / 14% / 72%).
"""
import argparse
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from config import RANDOM_SEED
from data.manifest import Manifest


def _groups_for(rows):
    """One group id per recordist; recordist-less files get a unique
    singleton id so they can be placed freely."""
    groups = []
    for i, r in enumerate(rows):
        rec = (r.get("recordist") or "").strip()
        groups.append(rec if rec else f"__nogroup_{i}")
    return groups


def assign_splits(rows, n_folds, seed):
    """Returns a list of 'train'/'val'/'test', one per row.

    Tries StratifiedGroupKFold (grouped + stratified); if the data is too
    small/imbalanced for it, falls back to an unstratified grouped split so
    the run still succeeds (recordist-disjointness is preserved either way)."""
    species = [r["species"] for r in rows]
    groups = _groups_for(rows)
    n_groups = len(set(groups))

    if n_groups < 3:
        raise RuntimeError(
            f"Only {n_groups} distinct recordist group(s) in the manifest - "
            f"not enough to form a 3-way recordist-disjoint split. Download "
            f"more data (from more recordists) first."
        )
    folds = min(n_folds, n_groups)

    # --- Preferred path: stratified + grouped ---
    try:
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_of = [None] * len(rows)
        for fold_idx, (_, test_idx) in enumerate(sgkf.split(np.arange(len(rows)), species, groups)):
            for i in test_idx:
                fold_of[i] = fold_idx
        # fold 0 -> test, fold 1 -> val, the remaining folds -> train
        return ["test" if f == 0 else "val" if f == 1 else "train" for f in fold_of]
    except ValueError as e:
        print(f"  [warn] StratifiedGroupKFold couldn't stratify ({e}); "
              f"falling back to an unstratified grouped split.")

    # --- Fallback: grouped but not stratified (far weaker requirements) ---
    frac = 1.0 / folds
    idx = np.arange(len(rows))
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    trainval_idx, test_idx = next(gss.split(idx, species, groups))

    tv_groups = [groups[i] for i in trainval_idx]
    val_frac = frac / (1.0 - frac)  # take val as ~frac of the whole from what's left
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    rel_train, rel_val = next(gss2.split(trainval_idx, [species[i] for i in trainval_idx], tv_groups))

    split_of = ["train"] * len(rows)
    for i in test_idx:
        split_of[i] = "test"
    for i in trainval_idx[rel_val]:
        split_of[i] = "val"
    return split_of


def print_breakdown(rows, split_of):
    per = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for r, s in zip(rows, split_of):
        per[r["species"]][s] += 1

    print(f"\n{'Species':<24}{'Train':>7}{'Val':>7}{'Test':>7}")
    print("-" * 45)
    totals = {"train": 0, "val": 0, "test": 0}
    empty_test = []
    for species in sorted(per):
        c = per[species]
        print(f"{species:<24}{c['train']:>7}{c['val']:>7}{c['test']:>7}")
        for k in totals:
            totals[k] += c[k]
        if c["test"] == 0:
            empty_test.append(species)
    print("-" * 45)
    print(f"{'TOTAL':<24}{totals['train']:>7}{totals['val']:>7}{totals['test']:>7}")

    if empty_test:
        print(f"\n⚠️  No test samples for: {', '.join(empty_test)}")
        print("   (too few recordists to spare one for test - download more data)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=7,
                        help="~1/folds to test and to val, rest to train (default 7).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    manifest = Manifest.load()
    rows = manifest.rows()
    if not rows:
        raise RuntimeError("Manifest is empty - run scripts.download_data first.")

    split_of = assign_splits(rows, args.folds, args.seed)
    for r, s in zip(rows, split_of):
        manifest.set_split(r["species"], r["filename"], s)
    manifest.save()
    print_breakdown(rows, split_of)

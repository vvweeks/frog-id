"""
data/inventory.py - Reports dataset composition from the manifest:
per-species counts broken out by split (train/val/test/unassigned) and
by source (Xeno-canto/iNaturalist).
"""
from collections import defaultdict

from config import SPECIES_MAP
from data.manifest import Manifest


def print_dataset_inventory(manifest=None):
    manifest = manifest or Manifest.load()
    rows = manifest.rows()

    # species -> counts
    by_split = defaultdict(lambda: defaultdict(int))   # species -> split -> n
    by_source = defaultdict(lambda: defaultdict(int))  # species -> source -> n
    for r in rows:
        sp = r["species"]
        by_split[sp][r.get("split") or "unassigned"] += 1
        by_source[sp][r.get("source") or "unknown"] += 1

    print("\n=================== DATASET INVENTORY (from manifest) ===================")
    header = f"{'Species':<24}{'XC':>5}{'iNat':>6}{'Train':>7}{'Val':>6}{'Test':>6}{'Unasgn':>8}"
    print(header)
    print("-" * len(header))

    totals = defaultdict(int)
    empty_test, unassigned_any = [], False
    for species in SPECIES_MAP:
        s, src = by_split[species], by_source[species]
        xc, inat = src.get("xeno_canto", 0), src.get("inat", 0)
        tr, va, te = s.get("train", 0), s.get("val", 0), s.get("test", 0)
        un = s.get("unassigned", 0)
        print(f"{species:<24}{xc:>5}{inat:>6}{tr:>7}{va:>6}{te:>6}{un:>8}")
        for k, v in [("xc", xc), ("inat", inat), ("train", tr), ("val", va), ("test", te), ("unassigned", un)]:
            totals[k] += v
        total_for_sp = xc + inat
        if total_for_sp > 0 and te == 0:
            empty_test.append(species)
        if un > 0:
            unassigned_any = True

    print("-" * len(header))
    print(f"{'TOTAL':<24}{totals['xc']:>5}{totals['inat']:>6}"
          f"{totals['train']:>7}{totals['val']:>6}{totals['test']:>6}{totals['unassigned']:>8}")
    print(f"\nGrand total: {len(rows)} files "
          f"[XC: {totals['xc']}, iNat: {totals['inat']}]")

    if unassigned_any:
        print("\nℹ️  Some files are unassigned - run `python -m scripts.make_split` to assign train/val/test.")
    if empty_test:
        print(f"⚠️  Classes with 0 test samples: {', '.join(empty_test)}")
    print("========================================================================")
    return rows

"""
data/inventory.py - Reports current dataset composition per class,
split by Train/Test and by source (Xeno-canto/iNaturalist).
"""
import os
import pandas as pd

from config import TRAIN_DIR, TEST_DIR, SPECIES_MAP, VAL_SPLIT


def _count_by_source(folder_path):
    xc_count = 0
    inat_count = 0
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.startswith("inat_"):
                inat_count += 1
            elif file_name.endswith(".mp3") or file_name.endswith(".wav"):
                xc_count += 1
    return xc_count, inat_count


def print_dataset_inventory(val_split=None):
    """
    Prints a per-class inventory broken out by physical split (Train/Test)
    and by source. Note: there is no physical Validation folder - Val is
    carved out of TRAIN_DIR at runtime by train_test_split() during
    training, so the Val/Effective-Train columns here are an *estimate*.
    """
    print("\n=================== DATASET INVENTORY REPORT ===================")
    effective_val_split = val_split if val_split is not None else VAL_SPLIT

    table_rows = []
    totals = {k: 0 for k in [
        "xc_train", "inat_train", "total_train",
        "xc_test", "inat_test", "total_test",
        "est_val", "est_effective_train",
    ]}

    for class_name, sci_name in SPECIES_MAP.items():
        train_path = os.path.join(TRAIN_DIR, class_name)
        test_path = os.path.join(TEST_DIR, class_name)

        xc_train, inat_train = _count_by_source(train_path)
        xc_test, inat_test = _count_by_source(test_path)

        total_train = xc_train + inat_train
        total_test = xc_test + inat_test
        est_val = round(total_train * effective_val_split)
        est_effective_train = total_train - est_val

        table_rows.append({
            "Common Name": class_name.replace("_", " "),
            "Scientific Name": sci_name,
            "XC (Train)": xc_train, "iNat (Train)": inat_train,
            "Total Train": total_train,
            "Est. Val": est_val, "Est. Effective Train": est_effective_train,
            "XC (Test)": xc_test, "iNat (Test)": inat_test,
            "Total Test": total_test,
            "Grand Total": total_train + total_test,
        })

        totals["xc_train"] += xc_train
        totals["inat_train"] += inat_train
        totals["total_train"] += total_train
        totals["xc_test"] += xc_test
        totals["inat_test"] += inat_test
        totals["total_test"] += total_test
        totals["est_val"] += est_val
        totals["est_effective_train"] += est_effective_train

    df = pd.DataFrame(table_rows)
    try:
        from IPython.display import display
        display(df)
    except Exception:
        # IPython being importable doesn't guarantee a live kernel to
        # display through (e.g. when run via `!python -m ...`, which
        # executes as a subprocess outside the notebook's kernel).
        print(df.to_string(index=False))

    print(f"\n--- Totals (Est. Val based on val_split={effective_val_split:.0%}) ---")
    print(f"Train (raw files, before Val split): {totals['total_train']}  "
          f"[XC: {totals['xc_train']}, iNat: {totals['inat_train']}]")
    print(f"  -> Est. Effective Train: {totals['est_effective_train']}  |  Est. Val: {totals['est_val']}")
    print(f"Test (held-out, untouched during training): {totals['total_test']}  "
          f"[XC: {totals['xc_test']}, iNat: {totals['inat_test']}]")
    print(f"Grand Total (all files): {totals['total_train'] + totals['total_test']}")

    thin_test_classes = df[df["Total Test"] < 5]["Common Name"].tolist()
    if thin_test_classes:
        print(f"\n⚠️  Classes with fewer than 5 Test samples: {', '.join(thin_test_classes)}")

    print("================================================================")
    return df

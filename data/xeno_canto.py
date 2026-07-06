"""
data/xeno_canto.py - Xeno-canto API v3 downloader with synonym fallback
and corrupt-file blocklist support. Pure data acquisition - no model or
training code lives here.
"""
import os
import time
import random
import requests

from config import (
    TRAIN_DIR, TEST_DIR, SPECIES_MAP, XC_API_KEY,
    TARGET_TRAIN, TARGET_TEST, CORRUPT_FILE_IDS, get_candidate_names,
)


def _query_xeno_canto(sci_name):
    """Runs a single Xeno-canto v3 query for one scientific name.
    Returns the list of recordings (empty list on any failure)."""
    name_parts = sci_name.strip().split()
    genus, species = name_parts[0], name_parts[1] if len(name_parts) > 1 else ""
    query_str = f'gen:{genus}' + (f' sp:{species}' if species else '') + ' cnt:"United States"'

    response = requests.get(
        "https://xeno-canto.org/api/3/recordings",
        params={'query': query_str, 'key': XC_API_KEY}
    )
    if response.status_code != 200:
        return []
    return response.json().get('recordings', [])


def _download_xc_recordings(recordings, out_dir, limit):
    """Downloads up to `limit` new recordings into out_dir.
    Returns count actually downloaded."""
    downloaded = 0
    random.shuffle(recordings)
    for rec in recordings:
        if downloaded >= limit:
            break
        rec_id = str(rec.get('id'))
        if rec_id in CORRUPT_FILE_IDS:
            continue  # known-corrupt, skip permanently
        file_url = rec.get('file')
        if not file_url:
            continue
        if file_url.startswith("//"):
            file_url = "https:" + file_url
        file_name = os.path.join(out_dir, f"{rec_id}.mp3")
        if os.path.exists(file_name):
            continue
        try:
            res = requests.get(file_url, headers={'User-Agent': 'Mozilla/5.0'})
            content_type = res.headers.get('Content-Type', '')
            if res.status_code == 200 and (content_type.startswith('audio') or content_type == 'application/octet-stream'):
                with open(file_name, 'wb') as f:
                    f.write(res.content)
                downloaded += 1
        except requests.RequestException as e:
            print(f"  [Xeno-canto] download failed for {rec_id}: {e}")
        time.sleep(0.5)
    return downloaded


def download_xeno_canto_data(target_train=TARGET_TRAIN, target_test=TARGET_TEST):
    if not XC_API_KEY:
        raise RuntimeError("XC_API_KEY is not set - see config.py for how to set it.")

    print("--- Xeno-Canto Scraping (API v3, synonym-aware) ---")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    for class_name in SPECIES_MAP:
        train_path = os.path.join(TRAIN_DIR, class_name)
        test_path = os.path.join(TEST_DIR, class_name)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for sci_name in get_candidate_names(class_name):
            needed_train = max(0, target_train - len(os.listdir(train_path)))
            needed_test = max(0, target_test - len(os.listdir(test_path)))

            if needed_train == 0 and needed_test == 0:
                break  # target already met, no need to try more synonyms

            try:
                recordings = _query_xeno_canto(sci_name)
                if not recordings:
                    print(f"  -> '{sci_name}' returned 0 recordings for {class_name}.")
                    continue

                good_quality = [r for r in recordings if r.get('q') in ['A', 'B']]
                messy_quality = [r for r in recordings if r.get('q') in ['C', 'D']]

                got_train = _download_xc_recordings(good_quality, train_path, needed_train) if needed_train else 0
                got_test = _download_xc_recordings(messy_quality, test_path, needed_test) if needed_test else 0

                print(f"  -> '{sci_name}': +{got_train} Train, +{got_test} Test for {class_name}.")

            except Exception as e:
                print(f"  -> Xeno-Canto error on {class_name} ({sci_name}): {e}")

        final_train = len(os.listdir(train_path))
        final_test = len(os.listdir(test_path))
        print(f"{class_name}: {final_train}/{target_train} Train, {final_test}/{target_test} Test.\n")

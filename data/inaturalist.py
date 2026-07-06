"""
data/inaturalist.py - iNaturalist gap-filling downloader with synonym
fallback and corrupt-file blocklist support.
"""
import os
import time
import requests

from config import (
    TRAIN_DIR, TEST_DIR, SPECIES_MAP, INAT_HEADERS,
    TARGET_TRAIN, TARGET_TEST, CORRUPT_FILE_IDS, get_candidate_names,
)

INAT_BASE_URL = "https://api.inaturalist.org/v1/observations"


def _fetch_and_download_inat(sci_name, quality_grade, out_dir, limit):
    downloaded = 0
    page = 1
    while downloaded < limit:
        params = {
            'taxon_name': sci_name, 'sounds': 'true',
            'quality_grade': quality_grade, 'per_page': 30, 'page': page,
        }
        try:
            resp = requests.get(INAT_BASE_URL, params=params, headers=INAT_HEADERS)
        except requests.RequestException as e:
            print(f"  [iNaturalist] request failed for '{sci_name}' page {page}: {e}")
            break
        if resp.status_code != 200:
            break
        results = resp.json().get('results', [])
        if not results:
            break
        for obs in results:
            if downloaded >= limit:
                break
            for sound in obs.get('sounds', []):
                if downloaded >= limit:
                    break
                stem = f"inat_{sound.get('id')}"
                if stem in CORRUPT_FILE_IDS:
                    continue  # known-corrupt, skip permanently
                file_url = sound.get('file_url')
                if not file_url:
                    continue
                file_name = os.path.join(out_dir, f"{stem}.mp3")
                if os.path.exists(file_name):
                    continue
                try:
                    audio_resp = requests.get(file_url, headers=INAT_HEADERS)
                    content_type = audio_resp.headers.get('Content-Type', '')
                    if audio_resp.status_code == 200 and (content_type.startswith('audio') or content_type == 'application/octet-stream'):
                        with open(file_name, 'wb') as f:
                            f.write(audio_resp.content)
                        downloaded += 1
                except requests.RequestException as e:
                    print(f"  [iNaturalist] download failed for {stem}: {e}")
                time.sleep(1.0)
        page += 1
        time.sleep(1.0)
    return downloaded


def fill_gaps_with_inaturalist(species_map=None, target_train=TARGET_TRAIN, target_test=TARGET_TEST):
    print("--- iNaturalist Gap-Filling (synonym-aware) ---")
    target_species = species_map if species_map is not None else SPECIES_MAP

    for class_name in target_species:
        train_path = os.path.join(TRAIN_DIR, class_name)
        test_path = os.path.join(TEST_DIR, class_name)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for sci_name in get_candidate_names(class_name):
            needed_train = max(0, target_train - len(os.listdir(train_path)))
            needed_test = max(0, target_test - len(os.listdir(test_path)))

            if needed_train == 0 and needed_test == 0:
                break

            print(f"Topping up {class_name} via '{sci_name}': "
                  f"{needed_train} Train, {needed_test} Test needed...")

            got_train = _fetch_and_download_inat(sci_name, "research", train_path, needed_train) if needed_train else 0
            got_test = _fetch_and_download_inat(sci_name, "needs_id", test_path, needed_test) if needed_test else 0

            print(f"  -> '{sci_name}': +{got_train} Train, +{got_test} Test for {class_name}.")

        final_train = len(os.listdir(train_path))
        final_test = len(os.listdir(test_path))
        print(f"{class_name}: {final_train}/{target_train} Train, {final_test}/{target_test} Test.\n")

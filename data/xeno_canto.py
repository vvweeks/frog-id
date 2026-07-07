"""
data/xeno_canto.py - Xeno-canto API v3 downloader with synonym fallback
and corrupt-file blocklist support. Pure data acquisition - no model or
training code lives here.

Downloads into the flat DATA_DIR/<species>/ layout and records each
file's recordist + quality in the manifest. Quality no longer decides
train vs test (that's the recordist-disjoint split's job) - it's just
logged, and used to prefer better-sounding recordings when we can't
grab them all.
"""
import os
import time
import requests

from config import (
    DATA_DIR, SPECIES_MAP, XC_API_KEY,
    TARGET_PER_SPECIES, CORRUPT_FILE_IDS, get_candidate_names, species_dir,
)
from data.manifest import Manifest

# Lower sorts first -> we download the best-quality audio we can before
# falling back to messier recordings. Unrated ('', 'no score') sorts last.
_QUALITY_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _count_audio(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith((".mp3", ".wav"))])


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


def _download_xc_recordings(recordings, class_name, out_dir, limit, manifest):
    """Downloads up to `limit` NEW recordings into out_dir (best quality
    first), and records recordist/quality for every recording already on
    disk too - so existing files get back-filled into the manifest
    without re-downloading. Returns count actually downloaded."""
    downloaded = 0
    recordings = sorted(recordings, key=lambda r: _QUALITY_ORDER.get(r.get('q', ''), 9))
    for rec in recordings:
        rec_id = str(rec.get('id'))
        if rec_id in CORRUPT_FILE_IDS:
            continue  # known-corrupt, skip permanently
        filename = f"{rec_id}.mp3"
        file_path = os.path.join(out_dir, filename)
        recordist = rec.get('rec', '')
        quality = rec.get('q', '')

        if os.path.exists(file_path):
            # Already have it - just make sure its metadata is recorded.
            manifest.upsert(class_name, filename, "xeno_canto", recordist, quality)
            continue
        if downloaded >= limit:
            continue  # keep scanning to back-fill existing files, but stop downloading

        file_url = rec.get('file')
        if not file_url:
            continue
        if file_url.startswith("//"):
            file_url = "https:" + file_url
        try:
            res = requests.get(file_url, headers={'User-Agent': 'Mozilla/5.0'})
            content_type = res.headers.get('Content-Type', '')
            if res.status_code == 200 and (content_type.startswith('audio') or content_type == 'application/octet-stream'):
                with open(file_path, 'wb') as f:
                    f.write(res.content)
                manifest.upsert(class_name, filename, "xeno_canto", recordist, quality)
                downloaded += 1
        except requests.RequestException as e:
            print(f"  [Xeno-canto] download failed for {rec_id}: {e}")
        time.sleep(0.5)
    return downloaded


def download_xeno_canto_data(target_per_species=TARGET_PER_SPECIES, manifest=None):
    if not XC_API_KEY:
        raise RuntimeError("XC_API_KEY is not set - see config.py for how to set it.")

    print("--- Xeno-Canto Scraping (API v3, synonym-aware, flat layout) ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    own_manifest = manifest is None
    if manifest is None:
        manifest = Manifest.load()

    for class_name in SPECIES_MAP:
        out_dir = species_dir(class_name)
        os.makedirs(out_dir, exist_ok=True)

        for sci_name in get_candidate_names(class_name):
            needed = max(0, target_per_species - _count_audio(out_dir))
            if needed == 0:
                break  # target already met, no need to try more synonyms

            try:
                recordings = _query_xeno_canto(sci_name)
                if not recordings:
                    print(f"  -> '{sci_name}' returned 0 recordings for {class_name}.")
                    continue
                got = _download_xc_recordings(recordings, class_name, out_dir, needed, manifest)
                print(f"  -> '{sci_name}': +{got} for {class_name}.")
            except Exception as e:
                print(f"  -> Xeno-Canto error on {class_name} ({sci_name}): {e}")

        print(f"{class_name}: {_count_audio(out_dir)}/{target_per_species} recordings.\n")

    if own_manifest:
        manifest.save()

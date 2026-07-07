"""
data/inaturalist.py - iNaturalist gap-filling downloader with synonym
fallback and corrupt-file blocklist support.

Research-grade only: research-grade observations have a community-verified
ID (2+ agreeing identifiers), so their species labels are trustworthy.
"needs_id" observations are unverified and would inject label noise into
the pool, so we don't pull them. Downloads into the flat
DATA_DIR/<species>/ layout and records the observer as the recordist.
"""
import os
import time
import requests

from config import (
    SPECIES_MAP, INAT_HEADERS,
    TARGET_PER_SPECIES, CORRUPT_FILE_IDS, get_candidate_names, species_dir,
)
from data.manifest import Manifest

INAT_BASE_URL = "https://api.inaturalist.org/v1/observations"


def _count_audio(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith((".mp3", ".wav"))])


def _observer(obs):
    """Stable per-recordist id for grouping: the user's login, falling
    back to their numeric id."""
    user = obs.get('user') or {}
    return user.get('login') or str(user.get('id', '')) or ''


def _fetch_and_download_inat(sci_name, class_name, out_dir, limit, manifest):
    """Downloads up to `limit` NEW research-grade sounds, and back-fills
    manifest metadata for any already on disk. Returns count downloaded."""
    downloaded = 0
    page = 1
    while downloaded < limit:
        params = {
            'taxon_name': sci_name, 'sounds': 'true',
            'quality_grade': 'research', 'per_page': 30, 'page': page,
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
            recordist = _observer(obs)
            for sound in obs.get('sounds', []):
                if downloaded >= limit:
                    break
                stem = f"inat_{sound.get('id')}"
                if stem in CORRUPT_FILE_IDS:
                    continue  # known-corrupt, skip permanently
                filename = f"{stem}.mp3"
                file_path = os.path.join(out_dir, filename)
                if os.path.exists(file_path):
                    manifest.upsert(class_name, filename, "inat", recordist, "research")
                    continue
                file_url = sound.get('file_url')
                if not file_url:
                    continue
                try:
                    audio_resp = requests.get(file_url, headers=INAT_HEADERS)
                    content_type = audio_resp.headers.get('Content-Type', '')
                    if audio_resp.status_code == 200 and (content_type.startswith('audio') or content_type == 'application/octet-stream'):
                        with open(file_path, 'wb') as f:
                            f.write(audio_resp.content)
                        manifest.upsert(class_name, filename, "inat", recordist, "research")
                        downloaded += 1
                except requests.RequestException as e:
                    print(f"  [iNaturalist] download failed for {stem}: {e}")
                time.sleep(1.0)
        page += 1
        time.sleep(1.0)
    return downloaded


def fill_gaps_with_inaturalist(target_per_species=TARGET_PER_SPECIES, manifest=None):
    print("--- iNaturalist Gap-Filling (research-grade only, synonym-aware) ---")
    own_manifest = manifest is None
    if manifest is None:
        manifest = Manifest.load()

    for class_name in SPECIES_MAP:
        out_dir = species_dir(class_name)
        os.makedirs(out_dir, exist_ok=True)

        for sci_name in get_candidate_names(class_name):
            needed = max(0, target_per_species - _count_audio(out_dir))
            if needed == 0:
                break

            print(f"Topping up {class_name} via '{sci_name}': {needed} needed...")
            got = _fetch_and_download_inat(sci_name, class_name, out_dir, needed, manifest)
            print(f"  -> '{sci_name}': +{got} for {class_name}.")

        print(f"{class_name}: {_count_audio(out_dir)}/{target_per_species} recordings.\n")

    if own_manifest:
        manifest.save()

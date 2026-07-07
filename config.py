"""
config.py - All shared constants live here. Every other module imports
from this file rather than relying on notebook-style implicit globals.
"""
import os
import random
import numpy as np
import torch
from dotenv import load_dotenv

# --- Environment / storage ---
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if not IN_COLAB:
    load_dotenv()  # reads ./.env if present; no-op otherwise

if IN_COLAB:
    # Scripts run via `!python -m ...` execute as subprocesses with no
    # live kernel, so they can't complete Drive's interactive auth flow -
    # skip calling mount() if it's already mounted (done from an actual
    # notebook cell), and fail with a clear message if it isn't.
    if not os.path.ismount('/content/drive'):
        try:
            drive.mount('/content/drive')
        except Exception as e:
            raise RuntimeError(
                "Google Drive isn't mounted. Run `from google.colab import "
                "drive; drive.mount('/content/drive')` in a notebook cell "
                "first (this needs a live kernel to complete), then re-run "
                "this script."
            ) from e
    DRIVE_SAVE_DIR = '/content/drive/MyDrive/6575_Deep_Learning'
else:
    DRIVE_SAVE_DIR = os.environ.get('FROG_ID_SAVE_DIR', './6575_Deep_Learning')

os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

DATA_DIR = os.path.join(DRIVE_SAVE_DIR, "ct_frog_dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
EMBEDDING_CACHE_DIR = os.path.join(DATA_DIR, "birdnet_embeddings")
CHECKPOINT_DIR = os.path.join(DRIVE_SAVE_DIR, "checkpoints")

# --- Audio / preprocessing ---
SAMPLE_RATE = 22050
DURATION_SEC = 3.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION_SEC)

# --- Training hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
BACKBONE_LR_MULT = 0.1
WEIGHT_DECAY = 1e-4

# --- Reproducibility (set once, here, nowhere else) ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# --- Dataset download targets ---
TARGET_TRAIN = 60
TARGET_TEST = 15

# --- Species ---
# To add a species for a future capstone extension: add the Common Name
# as the key and the current Latin name as the value. NUM_CLASSES and the
# model's output layer both resize automatically.
SPECIES_MAP = {
    "American_Toad": "Anaxyrus americanus",
    "Fowlers_Toad": "Anaxyrus fowleri",
    "Gray_Treefrog": "Dryophytes versicolor",
    "Spring_Peeper": "Pseudacris crucifer",
    "American_Bullfrog": "Aquarana catesbeiana",
    "Green_Frog": "Aquarana clamitans",
    "Atlantic_Leopard_Frog": "Lithobates kauffeldi",
    "Northern_Leopard_Frog": "Lithobates pipiens",
    "Pickerel_Frog": "Lithobates palustris",
    "Wood_Frog": "Boreorana sylvatica",
    "Eastern_Spadefoot": "Scaphiopus holbrookii",
}
NUM_CLASSES = len(SPECIES_MAP)

# Alternate scientific names to try if a platform's own taxonomy hasn't
# adopted the newer genus split used in SPECIES_MAP above (e.g. Lithobates
# vs Aquarana/Boreorana is contested and unevenly adopted across sources).
SPECIES_SYNONYMS = {
    "American_Toad": [],
    "Fowlers_Toad": [],
    "Gray_Treefrog": ["Hyla versicolor"],
    "Spring_Peeper": [],
    "American_Bullfrog": ["Lithobates catesbeianus", "Rana catesbeiana"],
    "Green_Frog": ["Lithobates clamitans", "Rana clamitans"],
    "Atlantic_Leopard_Frog": ["Rana kauffeldi"],
    "Northern_Leopard_Frog": ["Rana pipiens"],
    "Pickerel_Frog": ["Rana palustris"],
    "Wood_Frog": ["Lithobates sylvaticus", "Rana sylvatica"],
    "Eastern_Spadefoot": [],
}


def get_candidate_names(class_name):
    """Ordered, de-duplicated list of scientific names to try for a
    class: primary name from SPECIES_MAP first, then synonyms."""
    primary = SPECIES_MAP[class_name]
    synonyms = SPECIES_SYNONYMS.get(class_name, [])
    seen = set()
    names = []
    for name in [primary] + synonyms:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


# --- API credentials ---
# NEVER hardcode these. Set as real environment variables (locally) or
# Colab secrets (in Colab: google.colab.userdata.get(...)) before use.
XC_API_KEY = os.environ.get("XC_API_KEY", "")
INAT_CONTACT_EMAIL = os.environ.get("INAT_CONTACT_EMAIL", "your_email@example.com")
INAT_HEADERS = {
    'User-Agent': f'FrogToadClassifier/1.0 (research project; contact: {INAT_CONTACT_EMAIL})'
}

if IN_COLAB and not XC_API_KEY:
    try:
        from google.colab import userdata
        XC_API_KEY = userdata.get('XC_API_KEY')
    except Exception:
        pass

if not XC_API_KEY:
    print("WARNING: XC_API_KEY is not set. Set it as an environment variable "
          "or Colab secret before calling the Xeno-canto downloader.")

# --- Known-corrupt file IDs (never re-download these) ---
# Filename stem exactly as the downloaders name files:
#   Xeno-canto  -> "{rec['id']}"        e.g. "884331"
#   iNaturalist -> "inat_{sound['id']}" e.g. "inat_2009423"
CORRUPT_FILE_IDS = {
    "884331",           # Green_Frog
    "inat_2009423",     # Fowlers_Toad
    "inat_1918816",     # Pickerel_Frog
    "884330",           # Gray_Treefrog
    "inat_1977906",     # Fowlers_Toad
    "883842",           # American_Toad
}

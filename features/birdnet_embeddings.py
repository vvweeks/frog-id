"""
features/birdnet_embeddings.py - Extracts and caches BirdNET embeddings
per audio file, and a Dataset that loads the cached embeddings for
training a lightweight classifier head.

IMPORTANT: BirdNET's encode() returns an AcousticFileEncodingResult
object, not a plain array or DataFrame. It exposes public properties:
  - .embeddings         shape (n_inputs, n_segments, emb_dim)
  - .embeddings_masked  same shape, boolean, True = invalid/padded segment
  - .emb_dim            embedding dimensionality (e.g. 1024 for v2.4)
We mask out invalid segments before pooling across the segment axis.
"""
import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from config import SAMPLE_RATE, TARGET_LENGTH, TRAIN_DIR, TEST_DIR, SPECIES_MAP, EMBEDDING_CACHE_DIR
from features.spectrogram_dataset import crop_to_target_length

_BIRDNET_MODEL = None
EMBEDDING_DIM = None  # populated on first load_birdnet_model() call


def load_birdnet_model():
    """Lazily loads the BirdNET model exactly once. Call this explicitly
    from a script - it is NOT triggered on import."""
    global _BIRDNET_MODEL, EMBEDDING_DIM
    if _BIRDNET_MODEL is None:
        import birdnet
        _BIRDNET_MODEL = birdnet.load("acoustic", "2.4", "tf")
    return _BIRDNET_MODEL


def sanity_check_encode(sample_file_path):
    """Runs encode() on one file and reports shape/emb_dim - useful to
    confirm the API hasn't changed before a big batch run. Sets the
    module-level EMBEDDING_DIM as a side effect."""
    global EMBEDDING_DIM
    model = load_birdnet_model()
    result = model.encode(sample_file_path)
    EMBEDDING_DIM = int(result.emb_dim)
    print(f"encode() type: {type(result)}")
    print(f"embeddings shape: {result.embeddings.shape}  (n_inputs, n_segments, emb_dim)")
    print(f"emb_dim: {EMBEDDING_DIM}")
    return result


def _pool_embedding(result):
    """Masked mean-pool across the segment axis for a single-file
    encode() result. Returns a single (emb_dim,) vector."""
    embeddings = result.embeddings           # (n_inputs, n_segments, emb_dim)
    mask = result.embeddings_masked          # same shape, True = invalid

    valid = ~mask
    filled = np.where(valid, embeddings, np.nan)
    pooled = np.nanmean(filled, axis=1)      # -> (n_inputs, emb_dim)

    if np.isnan(pooled).any():
        # Edge case: every segment was masked for some input - fall back
        # to an unmasked mean rather than returning NaNs.
        pooled = np.nanmean(embeddings, axis=1)

    return pooled[0]  # single input file per call -> (emb_dim,)


def extract_embedding_for_file(file_path, tmp_wav_path):
    """Crops audio to TARGET_LENGTH, writes a temp wav, runs it through
    BirdNET's encode(), and returns a single pooled embedding vector.
    Returns None if the file is corrupt/unreadable."""
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        print(f"  [Skip - corrupt] {file_path}: {e}")
        return None

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = crop_to_target_length(waveform, SAMPLE_RATE, TARGET_LENGTH, is_train=False)
    sf.write(tmp_wav_path, waveform.squeeze(0).numpy(), SAMPLE_RATE)

    model = load_birdnet_model()
    result = model.encode(tmp_wav_path)
    return _pool_embedding(result)


def build_embedding_cache(root_dir, class_map):
    """Walks root_dir, extracts + caches a BirdNET embedding .npy for
    every audio file not already cached. Safe to re-run - skips
    anything already cached."""
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    tmp_wav_path = os.path.join("/tmp", "birdnet_tmp_clip.wav")
    new_count = 0

    for class_name in class_map:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not (fname.endswith(".mp3") or fname.endswith(".wav")):
                continue
            file_path = os.path.join(class_dir, fname)
            cache_path = os.path.join(EMBEDDING_CACHE_DIR, f"{class_name}__{fname}.npy")
            if os.path.exists(cache_path):
                continue
            embedding = extract_embedding_for_file(file_path, tmp_wav_path)
            if embedding is not None:
                np.save(cache_path, embedding)
                new_count += 1

    print(f"Cached {new_count} new embeddings from {root_dir}.")


class BirdNETEmbeddingDataset(Dataset):
    """Loads pre-cached BirdNET embeddings (fast - no audio decoding or
    model inference at train time, since BirdNET is frozen)."""

    def __init__(self, root_dir, class_map):
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_map.keys())}
        self.embedding_paths = []
        self.labels = []

        for class_name in os.listdir(root_dir):
            if class_name not in self.class_to_idx:
                continue
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                cache_path = os.path.join(EMBEDDING_CACHE_DIR, f"{class_name}__{fname}.npy")
                if os.path.exists(cache_path):
                    self.embedding_paths.append(cache_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.embedding_paths)

    def __getitem__(self, idx):
        embedding = np.load(self.embedding_paths[idx]).astype(np.float32)
        return torch.from_numpy(embedding), self.labels[idx]

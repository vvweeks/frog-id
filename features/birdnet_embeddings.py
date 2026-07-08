"""
features/birdnet_embeddings.py - Extracts and caches BirdNET embeddings
per audio file, and a Dataset that loads the cached embeddings for
training a lightweight classifier head.

BirdNET already segments audio into 3-second windows internally, so we
lean on that for multi-clip: instead of mean-pooling a recording down to
one vector, we keep each valid segment as its own cached sample (capped at
MAX_CLIPS_PER_RECORDING). Audio is fed at BirdNET's native 48 kHz.

encode() returns an AcousticFileEncodingResult exposing:
  - .embeddings         shape (n_inputs, n_segments, emb_dim)
  - .embeddings_masked  same shape, boolean, True = invalid/padded
  - .emb_dim            embedding dimensionality (1024 for v2.4)
"""
import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from config import (
    BIRDNET_SAMPLE_RATE, MAX_CLIPS_PER_RECORDING, SPECIES_MAP,
    EMBEDDING_CACHE_DIR, species_dir,
)
from data.manifest import Manifest

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
    confirm the API hasn't changed before a big batch run."""
    global EMBEDDING_DIM
    model = load_birdnet_model()
    result = model.encode(sample_file_path)
    EMBEDDING_DIM = int(result.emb_dim)
    print(f"encode() type: {type(result)}")
    print(f"embeddings shape: {result.embeddings.shape}  (n_inputs, n_segments, emb_dim)")
    print(f"emb_dim: {EMBEDDING_DIM}")
    return result


def _cache_prefix(species, filename):
    return f"{species}__{filename}__seg"


def extract_segment_embeddings(file_path, tmp_wav_path, max_clips):
    """Returns a list of per-segment embedding vectors (one per valid
    BirdNET 3-second window, capped at max_clips). Empty list if the file
    is corrupt/unreadable."""
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        print(f"  [Skip - corrupt] {file_path}: {e}")
        return []

    if sr != BIRDNET_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, BIRDNET_SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Bound work: never encode more audio than max_clips 3-second windows.
    max_samples = max_clips * int(3 * BIRDNET_SAMPLE_RATE)
    waveform = waveform[:, :max_samples]
    sf.write(tmp_wav_path, waveform.squeeze(0).numpy(), BIRDNET_SAMPLE_RATE)

    model = load_birdnet_model()
    result = model.encode(tmp_wav_path)
    embeddings = np.asarray(result.embeddings)[0]          # (n_segments, emb_dim)
    masked = np.asarray(result.embeddings_masked)[0]        # (n_segments, emb_dim)
    seg_valid = ~masked.all(axis=-1)                        # (n_segments,)

    out = []
    for emb, valid in zip(embeddings, seg_valid):
        if valid:
            out.append(emb.astype(np.float32))
        if len(out) >= max_clips:
            break
    return out


def build_embedding_cache(class_map=None, manifest=None):
    """Walks the manifest and caches one BirdNET embedding .npy per valid
    segment of every audio file not already cached. Safe to re-run."""
    class_map = class_map or SPECIES_MAP
    manifest = manifest or Manifest.load()
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    tmp_wav_path = os.path.join("/tmp", "birdnet_tmp_clip.wav")
    new_count = 0

    for r in manifest.rows():
        species, filename = r["species"], r["filename"]
        if species not in class_map:
            continue
        file_path = os.path.join(species_dir(species), filename)
        if not os.path.exists(file_path):
            continue
        prefix = _cache_prefix(species, filename)
        if os.path.exists(os.path.join(EMBEDDING_CACHE_DIR, f"{prefix}0.npy")):
            continue  # already cached (seg0 present)

        for i, emb in enumerate(extract_segment_embeddings(file_path, tmp_wav_path, MAX_CLIPS_PER_RECORDING)):
            np.save(os.path.join(EMBEDDING_CACHE_DIR, f"{prefix}{i}.npy"), emb)
            new_count += 1

    print(f"Cached {new_count} new segment embeddings.")


class BirdNETEmbeddingDataset(Dataset):
    """Loads pre-cached BirdNET segment embeddings for one manifest split.

    Multi-clip is a training augmentation: on the TRAIN split each segment is
    its own sample. On val/test the task is one prediction per recording, so a
    recording's segments are mean-pooled into a single embedding - one sample
    per recording (matches the LLM per-recording comparison)."""

    def __init__(self, split, class_map):
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_map.keys())}
        self.mean = None   # set via set_normalization() with TRAIN stats
        self.std = None

        multiclip = (split == "train")
        # Each sample: (list_of_segment_paths, label). Train samples have one
        # path each; val/test samples hold all of a recording's segments.
        self.samples = []
        self.labels = []
        for r in Manifest.load().for_split(split):
            species = r["species"]
            if species not in self.class_to_idx:
                continue
            prefix = _cache_prefix(species, r["filename"])
            seg_paths = [
                os.path.join(EMBEDDING_CACHE_DIR, f"{prefix}{i}.npy")
                for i in range(MAX_CLIPS_PER_RECORDING)
                if os.path.exists(os.path.join(EMBEDDING_CACHE_DIR, f"{prefix}{i}.npy"))
            ]
            if not seg_paths:
                continue
            label = self.class_to_idx[species]
            if multiclip:
                for p in seg_paths:
                    self.samples.append(([p], label))
                    self.labels.append(label)
            else:
                self.samples.append((seg_paths, label))
                self.labels.append(label)

    def all_train_vectors(self):
        """Every individual embedding vector across samples - for fitting
        normalization stats and reading the embedding dimension."""
        return np.stack([np.load(p) for paths, _ in self.samples for p in paths]).astype(np.float32)

    def set_normalization(self, mean, std):
        """Standardize embeddings with stats fit on the TRAIN split (fit on
        train, apply to all - no val/test leakage)."""
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        vecs = [np.load(p).astype(np.float32) for p in paths]
        embedding = vecs[0] if len(vecs) == 1 else np.mean(vecs, axis=0)
        if self.mean is not None:
            embedding = (embedding - self.mean) / self.std
        return torch.from_numpy(embedding), self.labels[idx]

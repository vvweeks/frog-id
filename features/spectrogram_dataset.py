"""
features/spectrogram_dataset.py - FrogCallDataset: loads raw audio,
crops to the acoustic peak, converts to a mel-spectrogram "image" for
the ResNet18 transfer-learning pipeline.

Manifest-driven: which files belong to a split is read from the manifest's
`split` column (set by scripts/make_split.py), NOT from folder structure.
Each recording yields up to MAX_CLIPS_PER_RECORDING non-overlapping 3-second
clips - free extra data, and leakage-safe since every clip from one file
shares its recordist and therefore its split.
"""
import os
import random
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from config import (
    SAMPLE_RATE, TARGET_LENGTH, DURATION_SEC, MAX_CLIPS_PER_RECORDING, species_dir,
)
from data.manifest import Manifest


def crop_to_target_length(waveform, sr, target_length, is_train, jitter_range=0.4):
    """Band-weighted peak crop: finds the time frame where energy in the
    frog/toad-relevant frequency band peaks, and crops around it. Adds
    jitter during training for augmentation variety; deterministic
    (centered exactly on the peak) at eval/inference time."""
    if waveform.shape[1] <= target_length:
        pad_amount = target_length - waveform.shape[1]
        return torch.nn.functional.pad(waveform, (0, pad_amount))

    coarse_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=64, n_fft=512, hop_length=512
    )(waveform)
    frog_band_energy = coarse_spec[:, 10:45, :].sum(dim=1)
    frames_in_target = target_length // 512
    energy_smooth = torch.nn.functional.avg_pool1d(
        frog_band_energy, kernel_size=frames_in_target, stride=1
    )
    best_frame = torch.argmax(energy_smooth).item()
    start_sample = best_frame * 512

    if is_train:
        jitter = int(random.uniform(-jitter_range, jitter_range) * sr)
        start_sample = max(0, min(start_sample + jitter, waveform.shape[1] - target_length))
    else:
        start_sample = max(0, min(start_sample, waveform.shape[1] - target_length))

    return waveform[:, start_sample:start_sample + target_length]


def extract_clip(waveform, sr, target_length, clip_index, n_clips, is_train):
    """Splits the recording into n_clips contiguous segments and returns a
    peak-cropped target_length window from the clip_index-th segment. With
    n_clips==1 this is just a peak crop over the whole recording."""
    if n_clips <= 1:
        return crop_to_target_length(waveform, sr, target_length, is_train)
    seg_len = waveform.shape[1] // n_clips
    segment = waveform[:, clip_index * seg_len:(clip_index + 1) * seg_len]
    return crop_to_target_length(segment, sr, target_length, is_train)


def _n_clips_for(path):
    """Cheap clip count from file metadata (no full decode). Uses
    soundfile.info - torchaudio.info isn't available with the torchcodec
    backend."""
    try:
        info = sf.info(path)
    except Exception:
        return None  # unreadable - caller skips it
    duration = info.frames / info.samplerate if info.samplerate else 0
    return max(1, min(MAX_CLIPS_PER_RECORDING, int(duration // DURATION_SEC)))


class FrogCallDataset(Dataset):
    """Mel-spectrogram dataset for the ResNet18 pipeline, one item per
    (recording, clip) drawn from the given manifest split."""

    def __init__(self, split, class_map, is_train=None):
        # is_train controls augmentation; defaults to (split == 'train').
        self.is_train = (split == "train") if is_train is None else is_train
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_map.keys())}

        # Each sample: (path, clip_index, n_clips, label)
        self.samples = []
        self.labels = []
        for r in Manifest.load().for_split(split):
            species = r["species"]
            if species not in self.class_to_idx:
                continue
            path = os.path.join(species_dir(species), r["filename"])
            if not os.path.exists(path):
                continue
            n_clips = _n_clips_for(path)
            if n_clips is None:
                print(f"[skip - unreadable] {path}")
                continue
            label = self.class_to_idx[species]
            for ci in range(n_clips):
                self.samples.append((path, ci, n_clips, label))
                self.labels.append(label)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024,
            hop_length=512, f_min=300, f_max=8000,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, clip_index, n_clips, label = self.samples[idx]

        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            print(f"\n[Data Quality Warning] Could not decode: {path}")
            print(f"  -> Decoding Error: {e}")
            fallback_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(fallback_idx)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = extract_clip(waveform, SAMPLE_RATE, TARGET_LENGTH, clip_index, n_clips, self.is_train)

        if self.is_train:
            shift = int(random.uniform(-0.3, 0.3) * SAMPLE_RATE)
            waveform = torch.roll(waveform, shifts=shift, dims=-1)
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise

        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)

        if self.is_train:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)

        mel_spec = mel_spec.expand(3, -1, -1)
        return mel_spec, label

"""
features/spectrogram_dataset.py - FrogCallDataset: loads raw audio,
crops to the acoustic peak, converts to a mel-spectrogram "image" for
the ResNet18 transfer-learning pipeline.
"""
import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

from config import SAMPLE_RATE, TARGET_LENGTH


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


class FrogCallDataset(Dataset):
    """v3 Error-Tolerant Decoder + Smart Peak Crop dataset for the
    ResNet18-on-spectrograms pipeline."""

    def __init__(self, root_dir, class_map, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.file_paths = []
        self.labels = []

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_map.keys())}

        for class_name in os.listdir(root_dir):
            if class_name not in self.class_to_idx:
                continue
            class_dir = os.path.join(root_dir, class_name)
            for f in os.listdir(class_dir):
                if f.endswith('.mp3') or f.endswith('.wav'):
                    self.file_paths.append(os.path.join(class_dir, f))
                    self.labels.append(self.class_to_idx[class_name])

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024,
            hop_length=512, f_min=300, f_max=8000,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            print(f"\n[Data Quality Warning] Removing corrupt file: {path}")
            print(f"  -> Decoding Error: {e}")
            try:
                os.remove(path)
            except Exception:
                pass
            fallback_idx = random.randint(0, len(self.file_paths) - 1)
            return self.__getitem__(fallback_idx)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = crop_to_target_length(waveform, SAMPLE_RATE, TARGET_LENGTH, self.is_train)

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

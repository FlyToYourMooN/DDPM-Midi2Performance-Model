import logging

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as transforms
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import (
    bessel,
    butter,
    cheby1,
    cheby2,
    ellip,
    resample_poly,
    sosfiltfilt,
)
from torchaudio.transforms import MelScale

logger = logging.getLogger(__name__)


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


# fix params, from FFGan vocoder
class LinearSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        mode="pow2_sqrt",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode

        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, y):
        if y.ndim == 3:
            y = y.squeeze(1)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.view_as_real(spec)

        if self.mode == "pow2_sqrt":
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        return spec


class LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sample_rate=44032,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)
        self.mel_scale = MelScale(
            self.n_mels,
            self.sample_rate,
            self.f_min,
            self.f_max,
            self.n_fft // 2 + 1,
            "slaney",
            "slaney",
        )

    def compress(self, x):
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x):
        return torch.exp(x)

    def forward(self, x):
        x = self.spectrogram(x)
        x = self.mel_scale(x)
        x = self.compress(x)
        return x


def configure_device(device):
    if device.startswith("gpu"):
        if not torch.cuda.is_available():
            raise Exception(
                "CUDA support is not available on your platform. Re-run using CPU or TPU mode"
            )
        gpu_id = device.split(":")[-1]
        if gpu_id == "":
            # Use all GPU's
            gpu_id = "-1"
        return f"cuda:{gpu_id}", gpu_id
    return device


def normalize(S, min_level_db):
    return (S - min_level_db) / -min_level_db


def denormalize(S, min_level_db):
    return (S * -min_level_db) + min_level_db


def padding_spec(spec, max_length):
    lens = len(spec[0])
    padding_size = max_length - lens % max_length
    padding_mask = torch.zeros([len(spec), padding_size])
    spec = torch.cat([spec, padding_mask], axis=1)
    return spec


def cross_fade(a, b, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx : a.shape[0]] = (1 - k) * a[idx:] + k * b[:fade_len]
    np.copyto(dst=result[a.shape[0] :], src=b[fade_len:])
    return result

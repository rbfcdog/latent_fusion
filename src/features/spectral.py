from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy


def _to_array(series: pd.Series) -> np.ndarray:
    values = np.asarray(series.dropna(), dtype=float)
    if values.size == 0:
        return values
    values = values - np.mean(values)
    return values


def welch_psd(
    series: pd.Series,
    fs: float = 1.0,
    nperseg: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    values = _to_array(series)
    nperseg = min(nperseg, values.size)
    if values.size == 0 or nperseg < 2:
        return np.array([]), np.array([])
    freqs, psd = welch(values, fs=fs, nperseg=nperseg)
    return freqs, psd


def fft_spectrum(
    series: pd.Series,
    fs: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    values = _to_array(series)
    n = values.size
    if n == 0:
        return np.array([]), np.array([])
    spectrum = np.fft.fft(values)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    half = n // 2
    magnitude = np.abs(spectrum[:half]) * 2.0 / n
    freqs = freqs[:half]
    return freqs, magnitude


def spectral_entropy(
    series: pd.Series,
    fs: float = 1.0,
    nperseg: int = 256,
) -> float:
    _, psd = welch_psd(series, fs=fs, nperseg=nperseg)
    if psd.size == 0 or np.all(psd <= 0):
        return float("nan")
    psd = psd[psd > 0]
    norm = psd / np.sum(psd)
    if norm.size <= 1:
        return 0.0
    return float(entropy(norm))


def dominant_frequency(
    series: pd.Series,
    fs: float = 1.0,
    nperseg: int = 256,
) -> float:
    freqs, psd = welch_psd(series, fs=fs, nperseg=nperseg)
    if psd.size == 0:
        return float("nan")
    return float(freqs[int(np.argmax(psd))])


def spectral_features(series: pd.Series, fs: float = 1.0) -> dict[str, float]:
    nperseg = min(256, max(2, int(series.dropna().size)))
    freqs, psd = welch_psd(series, fs=fs, nperseg=nperseg)
    fft_freqs, fft_mag = fft_spectrum(series, fs=fs)
    if psd.size == 0 or np.all(psd <= 0):
        spec_ent = float("nan")
        dom_freq = float("nan")
        peak_ratio = float("nan")
    else:
        psd_pos = psd[psd > 0]
        norm = psd_pos / np.sum(psd_pos)
        spec_ent = float(entropy(norm)) if norm.size > 1 else 0.0
        dom_idx = int(np.argmax(psd))
        dom_freq = float(freqs[dom_idx])
        total = np.sum(psd)
        peak_ratio = float(psd[dom_idx] / total) if total > 0 else float("nan")
    if fft_mag.size == 0:
        fft_peak = float("nan")
    else:
        fft_peak = float(np.max(fft_mag))
    return {
        "spectral_entropy": spec_ent,
        "dominant_frequency": dom_freq,
        "peak_ratio": peak_ratio,
        "fft_peak_magnitude": fft_peak,
        "psd_mean": float(np.mean(psd)) if psd.size else float("nan"),
        "psd_max": float(np.max(psd)) if psd.size else float("nan"),
    }

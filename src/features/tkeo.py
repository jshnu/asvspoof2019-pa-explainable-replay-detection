"""Paper-inspired Teager Energy Cepstral Coefficients (TECC).

Reference:
M. R. Kamble and H. A. Patil, "Detection of replay spoof speech using
teager energy feature cues", Computer Speech & Language, 65 (2021), 101140.
https://doi.org/10.1016/j.csl.2020.101140

The implementation follows the paper's ASVspoof 2019 PA Table 2 settings:
40 linearly spaced Gabor subbands, 10--8000 Hz, 120-D output, 20 ms
aggregation windows, and 10 ms shift.

The paper does not report the Gabor bandwidth for this 40-filter setting.
We therefore make it an explicit parameter. The default bandwidth is set
to the linear center-frequency spacing, so adjacent filters have roughly
matching -3 dB bandwidths. This is an implementation assumption, not a
reported paper parameter.
"""
from __future__ import annotations

import numpy as np


def teager_energy_operator(x: np.ndarray) -> np.ndarray:
    """Discrete-time TEO: x[n]^2 - x[n-1] * x[n+1].

    Returns N-2 samples, aligned to input indices 1..N-2.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 3:
        raise ValueError("TEO requires at least 3 samples")
    return x[1:-1] ** 2 - x[:-2] * x[2:]


def _preemphasis(x: np.ndarray, coefficient: float) -> np.ndarray:
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    y[1:] = x[1:] - coefficient * x[:-1]
    return y


def _gabor_kernel(
    center_hz: float,
    sample_rate: int,
    bandwidth_hz: float,
    truncate_sigma: float = 4.0,
) -> np.ndarray:
    """Build a real Gaussian-windowed cosine Gabor FIR kernel.

    h(t) = exp(-b^2 t^2) cos(2*pi*f_c*t).

    bandwidth_hz is the full -3 dB amplitude bandwidth of the Gaussian
    spectral envelope. For exp(-b^2 t^2), b = pi*BW/sqrt(2*ln(2)).
    """
    if bandwidth_hz <= 0:
        raise ValueError("Gabor bandwidth must be positive")

    b = np.pi * bandwidth_hz / np.sqrt(2.0 * np.log(2.0))
    sigma_t = 1.0 / (np.sqrt(2.0) * b)
    radius = max(2, int(np.ceil(truncate_sigma * sigma_t * sample_rate)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    time_seconds = offsets / float(sample_rate)

    kernel = (
        np.exp(-(b * time_seconds) ** 2)
        * np.cos(2.0 * np.pi * center_hz * time_seconds)
    )
    norm = np.sqrt(np.sum(kernel * kernel))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        raise FloatingPointError("Invalid Gabor kernel normalization")
    return kernel / norm


def _gabor_filterbank(
    x: np.ndarray,
    sample_rate: int,
    n_filters: int,
    fmin: float,
    fmax: float,
    bandwidth_hz: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Filter waveform into linearly spaced Gabor subbands.

    Returns (subband_signals, center_frequencies, resolved_bandwidth).
    Subband output shape is (n_filters, samples).
    """
    from scipy.signal import fftconvolve

    if n_filters < 2:
        raise ValueError("n_filters must be at least 2")
    if not (0.0 < fmin < fmax <= sample_rate / 2):
        raise ValueError("Require 0 < fmin < fmax <= Nyquist")

    centers = np.linspace(fmin, fmax, n_filters, dtype=np.float64)
    spacing = float(centers[1] - centers[0])
    resolved_bandwidth = spacing if bandwidth_hz is None else float(bandwidth_hz)
    if resolved_bandwidth <= 0:
        raise ValueError("bandwidth_hz must be positive")

    subbands = np.empty((n_filters, x.size), dtype=np.float64)
    for index, center in enumerate(centers):
        kernel = _gabor_kernel(
            center_hz=float(center),
            sample_rate=sample_rate,
            bandwidth_hz=resolved_bandwidth,
        )
        subbands[index] = fftconvolve(x, kernel, mode="same")

    return subbands, centers, resolved_bandwidth


def _frame_mean(
    x: np.ndarray,
    frame_length: int,
    frame_shift: int,
) -> np.ndarray:
    """Mean each complete frame; returns (channels, frames)."""
    if x.ndim != 2:
        raise ValueError("Expected a (channels, samples) array")
    if frame_length < 1 or frame_shift < 1:
        raise ValueError("Frame length and shift must be positive")
    if x.shape[1] < frame_length:
        raise ValueError(
            f"Signal has {x.shape[1]} samples, shorter than frame length {frame_length}"
        )
    n_frames = 1 + (x.shape[1] - frame_length) // frame_shift
    starts = np.arange(n_frames) * frame_shift
    # The loop avoids materializing a potentially large overlapping 3-D view.
    framed_means = np.empty((x.shape[0], n_frames), dtype=np.float64)
    for frame_index, start in enumerate(starts):
        framed_means[:, frame_index] = np.mean(
            x[:, start:start + frame_length], axis=1
        )
    return framed_means


def _regression_delta(features: np.ndarray, window: int = 2) -> np.ndarray:
    """Standard centered regression delta with edge replication."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Delta input must be 2-D")
    if window < 1:
        raise ValueError("Delta window must be >= 1")
    padded = np.pad(features, ((0, 0), (window, window)), mode="edge")
    denominator = 2.0 * np.sum(
        np.arange(1, window + 1, dtype=np.float64) ** 2
    )
    delta = np.zeros_like(features, dtype=np.float64)
    for offset in range(1, window + 1):
        delta += offset * (
            padded[:, window + offset:window + offset + features.shape[1]]
            - padded[:, window - offset:window - offset + features.shape[1]]
        )
    return delta / denominator


def extract_tecc(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    *,
    preemphasis: float = 0.97,
    n_filters: int = 40,
    fmin: float = 10.0,
    fmax: float | None = None,
    bandwidth_hz: float | None = None,
    frame_length_ms: float = 20.0,
    frame_shift_ms: float = 10.0,
    n_ceps: int = 40,
    delta_window: int = 2,
    eps: float = 1e-12,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Extract TECC from one mono waveform.

    Returns float64 features of shape (3*n_ceps, frames): static, delta,
    delta-delta. The static coefficients are cepstral-mean normalized.

    `bandwidth_hz=None` resolves to the spacing between linearly spaced
    filter centers; the paper does not specify the bandwidth for its
    40-filter ASVspoof 2019 PA parameterization.
    """
    from scipy.fft import dct

    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if x.size < 3:
        raise ValueError("Waveform must contain at least 3 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("Waveform contains non-finite samples")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 <= preemphasis < 1.0:
        raise ValueError("preemphasis must be in [0, 1)")
    if n_ceps < 1 or n_ceps > n_filters:
        raise ValueError("n_ceps must be between 1 and n_filters")
    if eps <= 0:
        raise ValueError("eps must be positive")

    fmax = sample_rate / 2.0 if fmax is None else float(fmax)
    emphasized = _preemphasis(x, preemphasis)
    subbands, centers, resolved_bandwidth = _gabor_filterbank(
        emphasized,
        sample_rate=sample_rate,
        n_filters=n_filters,
        fmin=fmin,
        fmax=fmax,
        bandwidth_hz=bandwidth_hz,
    )

    # TEO is evaluated independently on each narrowband component.
    teo = np.empty((n_filters, max(0, x.size - 2)), dtype=np.float64)
    for index in range(n_filters):
        teo[index] = teager_energy_operator(subbands[index])

    # TEO can be negative for non-ideal/noisy subbands. Since the next
    # stage is log energy, clip negative estimates to zero before averaging.
    teo = np.maximum(teo, 0.0)
    frame_length = int(round(frame_length_ms * sample_rate / 1000.0))
    frame_shift = int(round(frame_shift_ms * sample_rate / 1000.0))
    if frame_length < 1 or frame_shift < 1:
        raise ValueError("Frame durations resolve to zero samples")

    mean_energy = _frame_mean(teo, frame_length, frame_shift)
    log_energy = np.log(mean_energy + eps)

    cepstra = dct(log_energy, type=2, axis=0, norm="ortho")[:n_ceps, :]

    # Cepstral mean normalization over the utterance, per coefficient.
    static = cepstra - np.mean(cepstra, axis=1, keepdims=True)
    delta = _regression_delta(static, window=delta_window)
    delta_delta = _regression_delta(delta, window=delta_window)

    features = np.vstack((static, delta, delta_delta))
    if features.shape[0] != 3 * n_ceps or features.shape[1] == 0:
        raise RuntimeError(f"Unexpected TECC shape: {features.shape}")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("TECC extraction produced non-finite values")

    if not return_details:
        return features

    details = {
        "sample_rate": int(sample_rate),
        "preemphasis": float(preemphasis),
        "n_filters": int(n_filters),
        "fmin_hz": float(fmin),
        "fmax_hz": float(fmax),
        "bandwidth_hz": float(resolved_bandwidth),
        "center_frequencies_hz": centers.tolist(),
        "frame_length_samples": frame_length,
        "frame_shift_samples": frame_shift,
        "n_ceps": int(n_ceps),
        "delta_window": int(delta_window),
        "feature_shape": list(features.shape),
        "teo_negative_values_clipped": True,
    }
    return features, details

"""LP-residual acoustic features (RLFCC) for replay-spoof countermeasures.

Research basis:
L. Wei, Y. Long, H. Wei and Y. Li, "New Acoustic Features for Synthetic and
Replay Spoofing Attack Detection", Symmetry 14(2), 274 (2022), DOI:
10.3390/sym14020274.

The paper's PA configuration specifies 25 ms frames, 12.5 ms shift and LP
order 12 for RLFCC. Its LFCC description uses a 512-point DFT, 20 linearly
spaced triangular filters and 19+0 DCT coefficients (60 dimensions including
static, delta and delta-delta). The paper does not restate those LFCC details
specifically for RLFCC; this implementation reuses that described LFCC setup.

RLFCC is computed from frame-wise LP residuals. The returned feature matrix is
(60, frames), consistent with the project's other frame-level feature groups.
"""
from __future__ import annotations

import numpy as np


def _frames(x: np.ndarray, frame_length: int, frame_shift: int) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("waveform must be mono and one-dimensional")
    if frame_length < 1 or frame_shift < 1:
        raise ValueError("frame length and shift must be positive")
    if x.size < frame_length:
        raise ValueError(
            f"waveform has {x.size} samples, shorter than frame length {frame_length}"
        )
    n_frames = 1 + (x.size - frame_length) // frame_shift
    starts = np.arange(n_frames, dtype=np.int64) * frame_shift
    idx = starts[:, None] + np.arange(frame_length, dtype=np.int64)[None, :]
    return x[idx]


def _levinson_durbin_batch(correlation: np.ndarray, order: int, eps: float = 1e-10) -> np.ndarray:
    """Vectorized Levinson-Durbin recursion for many frames at once."""
    r = np.asarray(correlation, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != order + 1:
        raise ValueError("correlation must have shape (frames, order + 1)")
    n_frames = r.shape[0]
    a = np.zeros((n_frames, order + 1), dtype=np.float64)
    a[:, 0] = 1.0
    error = np.maximum(r[:, 0], eps)
    valid = np.isfinite(r[:, 0]) & (r[:, 0] > eps)

    for i in range(1, order + 1):
        if i == 1:
            acc = r[:, i]
        else:
            acc = r[:, i] + np.sum(a[:, 1:i] * r[:, i - 1:0:-1], axis=1)
        reflection = np.zeros(n_frames, dtype=np.float64)
        reflection[valid] = -acc[valid] / error[valid]
        reflection = np.clip(reflection, -0.999, 0.999)
        previous = a.copy()
        if i > 1:
            a[:, 1:i] = previous[:, 1:i] + reflection[:, None] * previous[:, i - 1:0:-1]
        a[:, i] = reflection
        error *= np.maximum(1.0 - reflection * reflection, 1e-6)
    return a


def _levinson_durbin(frame: np.ndarray, order: int, eps: float = 1e-10) -> np.ndarray:
    """Scalar reference wrapper returning [1, a1, ..., ap]."""
    frame = np.asarray(frame, dtype=np.float64).reshape(-1)
    if order < 1 or order >= frame.size:
        raise ValueError("LP order must be >=1 and smaller than frame length")
    corr = np.asarray([
        np.dot(frame[:frame.size - lag], frame[lag:])
        for lag in range(order + 1)
    ], dtype=np.float64)[None, :]
    return _levinson_durbin_batch(corr, order, eps=eps)[0]


def lp_residual_frames(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 12.5,
    lp_order: int = 12,
) -> np.ndarray:
    """Compute Hamming-windowed, frame-wise LP residuals; shape (frames, samples)."""
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("waveform must be non-empty and finite")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    frame_length = int(round(frame_length_ms * sample_rate / 1000.0))
    frame_shift = int(round(frame_shift_ms * sample_rate / 1000.0))
    raw_frames = _frames(x, frame_length, frame_shift)
    window = np.hamming(frame_length).astype(np.float64)
    windowed = raw_frames * window[None, :]

    # Autocorrelation is needed only for lags 0..LP-order. Compute it in
    # vectorized form across all frames, then run batched Levinson-Durbin.
    correlations = np.empty((windowed.shape[0], lp_order + 1), dtype=np.float64)
    for lag in range(lp_order + 1):
        correlations[:, lag] = np.einsum(
            "ij,ij->i", windowed[:, :frame_length - lag], windowed[:, lag:]
        )
    coefficients = _levinson_durbin_batch(correlations, lp_order)

    # Apply each frame's own inverse filter. The short LP-order loop is
    # vectorized over every frame, avoiding one scipy.signal.lfilter call
    # per frame (important for the 218k-utterance corpus).
    residual = windowed.copy()
    for lag in range(1, lp_order + 1):
        residual[:, lag:] += coefficients[:, lag, None] * windowed[:, :-lag]
    return residual


def _linear_triangular_filterbank(
    n_fft: int, sample_rate: int, n_filters: int
) -> np.ndarray:
    """Create equally spaced triangular filters over [0, Nyquist]."""
    if n_filters < 2:
        raise ValueError("n_filters must be >= 2")
    frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    points = np.linspace(0.0, sample_rate / 2.0, n_filters + 2)
    bank = np.zeros((n_filters, frequencies.size), dtype=np.float64)
    for i in range(n_filters):
        left, center, right = points[i:i + 3]
        rising = (frequencies - left) / max(center - left, np.finfo(float).eps)
        falling = (right - frequencies) / max(right - center, np.finfo(float).eps)
        bank[i] = np.maximum(0.0, np.minimum(rising, falling))
    return bank


def _delta(features: np.ndarray, window: int = 2) -> np.ndarray:
    if window < 1:
        raise ValueError("delta window must be >=1")
    padded = np.pad(features, ((0, 0), (window, window)), mode="edge")
    denom = 2.0 * sum(i * i for i in range(1, window + 1))
    out = np.zeros_like(features)
    for i in range(1, window + 1):
        out += i * (
            padded[:, window + i:window + i + features.shape[1]]
            - padded[:, window - i:window - i + features.shape[1]]
        )
    return out / denom


def extract_rlfcc(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    *,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 12.5,
    lp_order: int = 12,
    n_fft: int = 512,
    n_filters: int = 20,
    n_ceps: int = 20,
    delta_window: int = 2,
    eps: float = 1e-10,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Extract residual LFCC (RLFCC), returning (3*n_ceps, frames)."""
    from scipy.fft import dct

    if n_ceps < 1 or n_ceps > n_filters:
        raise ValueError("n_ceps must be between 1 and n_filters")
    if n_fft < 2 or n_fft & (n_fft - 1):
        raise ValueError("n_fft must be a power of two >=2")
    if eps <= 0:
        raise ValueError("eps must be positive")

    residual = lp_residual_frames(
        waveform,
        sample_rate=sample_rate,
        frame_length_ms=frame_length_ms,
        frame_shift_ms=frame_shift_ms,
        lp_order=lp_order,
    )
    # Frames are already Hamming-windowed during LP analysis. Do not apply a
    # second window: compute the residual-frame periodogram directly.
    spectrum = np.fft.rfft(residual, n=n_fft, axis=1)
    power = np.abs(spectrum) ** 2
    bank = _linear_triangular_filterbank(n_fft, sample_rate, n_filters)
    energies = power @ bank.T
    log_energies = np.log(np.maximum(energies, eps))
    static = dct(log_energies, type=2, axis=1, norm="ortho")[:, :n_ceps].T
    delta = _delta(static, window=delta_window)
    delta_delta = _delta(delta, window=delta_window)
    features = np.vstack((static, delta, delta_delta))

    if not np.isfinite(features).all():
        raise FloatingPointError("RLFCC extraction produced non-finite values")
    if features.shape != (3 * n_ceps, residual.shape[0]):
        raise RuntimeError(f"Unexpected RLFCC shape: {features.shape}")

    if not return_details:
        return features
    details = {
        "feature_type": "RLFCC",
        "sample_rate": int(sample_rate),
        "frame_length_ms": float(frame_length_ms),
        "frame_shift_ms": float(frame_shift_ms),
        "lp_order": int(lp_order),
        "n_fft": int(n_fft),
        "n_filters": int(n_filters),
        "n_ceps": int(n_ceps),
        "delta_window": int(delta_window),
        "feature_shape": list(features.shape),
        "lfcc_filterbank_assumption": "20 linearly spaced triangular filters; inherited from paper's LFCC description",
        "windowing": "Hamming window applied to each frame before LPC analysis and residual filtering",
    }
    return features, details

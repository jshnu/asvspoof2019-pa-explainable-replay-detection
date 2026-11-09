"""NumPy port of the NSG constant-Q routines used by the ASVspoof CQCC toolbox.

This module ports the supplied MATLAB routines:
  - nsgcqwin.m
  - nsgtf_real.m
  - winfuns.m

It intentionally exposes the filterbank and NSGT separately so their
intermediate outputs can be compared with MATLAB before wiring in cqt.m.


Provenance and license notice:
This Python port/adaptation includes routines derived from the EURECOM CQCC
package (Copyright (C) 2016 EURECOM, France; CC BY-NC-SA 4.0) and CQT Toolbox
2013 routines (Copyright (C) 2013 Nicki Holighaus and, for relevant routines,
Christian Schörkhuber; CC BY-NC-SA 3.0). The adapted material in this file is
identified as CC BY-NC-SA 4.0. See licenses/CQCC_EURECOM_LICENSE.txt,
licenses/CQT_TOOLBOX_LICENSE.txt, and THIRD_PARTY_NOTICES.md for attribution,
source links, and license references. This file is excluded from the root MIT
License.
"""
from __future__ import annotations

import numpy as np


def _matlab_round_positive(x: np.ndarray) -> np.ndarray:
    """MATLAB round for the non-negative values used by this filterbank."""
    return np.floor(np.asarray(x) + 0.5).astype(np.int64)


def winfuns(name: str, x_or_n, length: int | None = None) -> np.ndarray:
    """Port of winfuns.m for the windows needed by this CQT implementation."""
    name = name.lower()
    if np.isscalar(x_or_n):
        n = int(x_or_n)
        out_len = n if length is None else int(length)
        if out_len < n:
            raise ValueError("Output length must be >= window support")
        if n % 2 == 0:
            left = np.arange(0, n // 2, dtype=np.float64) / n
            right = np.arange(-n // 2, 0, dtype=np.float64) / n
        else:
            left = np.arange(0, (n + 1) // 2, dtype=np.float64) / n
            right = np.arange(-(n // 2), 0, dtype=np.float64) / n
        x = np.concatenate((left, np.full(out_len - n, -n, dtype=np.float64), right))
    else:
        x = np.asarray(x_or_n, dtype=np.float64).reshape(-1)

    if name in ("hann", "nuttall10"):
        g = 0.5 + 0.5 * np.cos(2.0 * np.pi * x)
    elif name in ("cos", "cosine", "sqrthann"):
        g = np.cos(np.pi * x)
    elif name in ("hamming", "nuttall01"):
        g = 0.54 + 0.46 * np.cos(2.0 * np.pi * x)
    elif name in ("square", "rec"):
        g = (np.abs(x) < 0.5).astype(np.float64)
    elif name in ("tri", "triangular", "bartlett"):
        g = 1.0 - 2.0 * np.abs(x)
    elif name == "blackman":
        g = 0.42 + 0.5*np.cos(2*np.pi*x) + 0.08*np.cos(4*np.pi*x)
    elif name == "blackharr":
        g = 0.35875 + 0.48829*np.cos(2*np.pi*x) + 0.14128*np.cos(4*np.pi*x) + 0.01168*np.cos(6*np.pi*x)
    elif name == "modblackharr":
        g = 0.35872 + 0.48832*np.cos(2*np.pi*x) + 0.14128*np.cos(4*np.pi*x) + 0.01168*np.cos(6*np.pi*x)
    elif name in ("nuttall", "nuttall12"):
        g = 0.355768 + 0.487396*np.cos(2*np.pi*x) + 0.144232*np.cos(4*np.pi*x) + 0.012604*np.cos(6*np.pi*x)
    elif name == "nuttall20":
        g = 3/8 + 4/8*np.cos(2*np.pi*x) + 1/8*np.cos(4*np.pi*x)
    elif name == "nuttall11":
        g = 0.40897 + 0.5*np.cos(2*np.pi*x) + 0.09103*np.cos(4*np.pi*x)
    elif name == "nuttall02":
        g = 0.4243801 + 0.4973406*np.cos(2*np.pi*x) + 0.0782793*np.cos(4*np.pi*x)
    elif name == "nuttall30":
        g = 10/32 + 15/32*np.cos(2*np.pi*x) + 6/32*np.cos(4*np.pi*x) + 1/32*np.cos(6*np.pi*x)
    elif name == "nuttall21":
        g = 0.338946 + 0.481973*np.cos(2*np.pi*x) + 0.161054*np.cos(4*np.pi*x) + 0.018027*np.cos(6*np.pi*x)
    elif name == "nuttall03":
        g = 0.3635819 + 0.4891775*np.cos(2*np.pi*x) + 0.1365995*np.cos(4*np.pi*x) + 0.0106411*np.cos(6*np.pi*x)
    elif name in ("gauss", "truncgauss"):
        g = np.exp(-18.0*x*x)
    elif name == "wp2inp":
        g = np.exp(np.exp(-2*x)*25.0*(1.0+2*x))
        g = g / np.max(g)
    else:
        raise ValueError(f"Unknown window function: {name}")
    return np.asarray(g * (np.abs(x) < 0.5), dtype=np.float64)


def nsgcqwin(
    fmin: float,
    fmax: float,
    bins: int,
    sr: int,
    signal_length: int,
    *,
    min_win: int = 4,
    bwfac: int = 1,
    fractional: bool = False,
    winfun: str = "hann",
    gamma: float = 0.0,
):
    """Port of nsgcqwin.m. Returns (windows, shift, M, metadata)."""
    nf = sr / 2.0
    fmax = min(float(fmax), nf)
    fftres = sr / signal_length
    b = int(np.floor(bins * np.log2(fmax / fmin)))
    fbas = float(fmin) * 2.0 ** (np.arange(b + 1, dtype=np.float64) / bins)

    q = 2.0 ** (1.0 / bins) - 2.0 ** (-1.0 / bins)
    cqtbw = q * fbas + gamma

    too_high = np.flatnonzero(fbas + cqtbw / 2.0 > nf)
    if too_high.size:
        stop = int(too_high[0])
        fbas, cqtbw = fbas[:stop], cqtbw[:stop]

    too_low = np.flatnonzero(fbas - cqtbw / 2.0 < 0)
    if too_low.size:
        start = int(too_low[-1]) + 1
        fbas, cqtbw = fbas[start:], cqtbw[start:]

    lfbas = len(fbas)
    positive = np.empty(2 * lfbas + 2, dtype=np.float64)
    positive[:lfbas + 1] = np.concatenate(([0.0], fbas))
    positive[lfbas + 1] = nf
    positive[lfbas + 2:] = sr - fbas[::-1]
    bw_hz = np.concatenate(([2.0 * fmin], cqtbw, [positive[lfbas + 2] - positive[lfbas]], cqtbw[::-1]))

    bw = bw_hz / fftres
    fbin = positive / fftres
    posit = np.empty_like(fbin, dtype=np.int64)
    posit[:lfbas + 2] = np.floor(fbin[:lfbas + 2]).astype(np.int64)
    posit[lfbas + 2:] = np.ceil(fbin[lfbas + 2:]).astype(np.int64)

    shift = np.concatenate((np.array([(-posit[-1]) % signal_length], dtype=np.int64), np.diff(posit)))

    if fractional:
        corr_shift = fbin - posit
        M = np.ceil(bw + 1.0).astype(np.int64)
    else:
        bw = _matlab_round_positive(bw)
        M = bw.astype(np.int64)

    for i in range(2 * (lfbas + 1)):
        if bw[i] < min_win:
            bw[i] = min_win
            M[i] = int(bw[i])

    windows = []
    if fractional:
        for x, y, z in zip(corr_shift, bw, M):
            sample_positions = (np.concatenate((
                np.arange(0, int(np.ceil(z / 2)) + 1),
                np.arange(-int(np.floor(z / 2)), 0)
            )) - x) / y
            windows.append(winfuns(winfun, sample_positions) / np.sqrt(y))
    else:
        windows = [winfuns(winfun, int(n)) for n in bw]

    M = bwfac * np.ceil(M / bwfac).astype(np.int64)

    # Match the special DC and Nyquist Tukey-like windows in nsgcqwin.m.
    for k in (0, lfbas + 1):
        if M[k] > M[k + 1]:
            n = int(M[k])
            inner = int(M[k + 1])
            g = np.ones(n, dtype=np.float64)
            start = (n // 2 - inner // 2)
            stop = n // 2 + int(np.ceil(inner / 2))
            g[start:stop] = winfuns("hann", inner)
            windows[k] = g / np.sqrt(n)

    meta = {"fbas": fbas, "posit": posit, "bandwidth_samples": bw, "Lfbas": lfbas}
    return windows, shift, M, meta


def nsgtf_real(x, windows, shift, M, phasemode: str = "global"):
    """Port of nsgtf_real.m for a mono real-valued signal."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    ls = x.size
    shift = np.asarray(shift, dtype=np.int64).reshape(-1)
    M = np.asarray(M, dtype=np.int64).reshape(-1)
    if len(windows) != len(shift) or len(M) != len(shift):
        raise ValueError("windows, shift and M must have equal lengths")

    spectrum = np.fft.fft(x)
    fill = int(np.sum(shift) - ls)
    if fill < 0:
        raise ValueError(f"NSGT padding is negative ({fill}); check filterbank construction")
    if fill:
        spectrum = np.concatenate((spectrum, np.zeros(fill, dtype=np.complex128)))

    total = ls + fill
    posit = np.cumsum(shift) - shift[0]
    lengths = np.asarray([len(g) for g in windows], dtype=np.int64)
    valid = np.flatnonzero(posit - np.floor(lengths / 2) <= total / 2)
    if not valid.size:
        raise RuntimeError("No supported NSGT filters")
    n_out = int(valid[-1]) + 1
    coeffs = []

    for i in range(n_out):
        lg = int(lengths[i])
        half_floor = lg // 2
        half_ceil = (lg + 1) // 2
        idx = np.concatenate((np.arange(half_ceil, lg), np.arange(0, half_ceil)))
        offsets = np.arange(-half_floor, half_ceil, dtype=np.int64)
        win_range = (int(posit[i]) + offsets) % total
        vals = spectrum[win_range] * windows[i][idx]
        mi = int(M[i])

        if mi < lg:
            col = int(np.ceil(lg / mi))
            temp = np.zeros(col * mi, dtype=np.complex128)
            target = np.concatenate((np.arange(col*mi-half_floor, col*mi), np.arange(0, half_ceil)))
            temp[target] = vals
            temp = temp.reshape(col, mi).sum(axis=0)
        else:
            temp = np.zeros(mi, dtype=np.complex128)
            target = np.concatenate((np.arange(mi-half_floor, mi), np.arange(0, half_ceil)))
            temp[target] = vals
            if phasemode == "global":
                displace = int(posit[i] - np.floor(posit[i] / mi) * mi)
                temp = np.roll(temp, displace)
            elif phasemode != "local":
                raise ValueError("phasemode must be 'global' or 'local'")

        coeffs.append(np.fft.ifft(temp))

    if np.all(M == M[0]):
        return np.stack(coeffs, axis=1)
    return coeffs



def _cqcc_positive_cqt_gpu(x, windows, shift, M, meta, *, cp):
    """GPU implementation of the positive-frequency NSGT rows.

    The MATLAB-compatible filterbank is constructed on CPU (small metadata);
    FFT, spectral gathers, filter multiplication, and the batched IFFT run on
    the selected CuPy device. Only positive-frequency rows are computed,
    because CQCC discards DC/Nyquist and negative-frequency rows.
    """
    xg = cp.asarray(np.asarray(x, dtype=np.float64).reshape(-1))
    ls = int(xg.size)
    shift_np = np.asarray(shift, dtype=np.int64)
    M_np = np.asarray(M, dtype=np.int64)
    total = int(np.sum(shift_np))
    fill = total - ls
    if fill < 0:
        raise ValueError(f"NSGT padding is negative ({fill}); check filterbank construction")

    spectrum = cp.fft.fft(xg)
    if fill:
        spectrum = cp.pad(spectrum, (0, fill))

    posit = np.cumsum(shift_np) - shift_np[0]
    n_bins = int(meta["Lfbas"])
    lengths = np.asarray([len(g) for g in windows], dtype=np.int64)
    valid = np.flatnonzero(posit - np.floor(lengths / 2) <= total / 2)
    if not valid.size or int(valid[-1]) < n_bins:
        raise RuntimeError("Positive-frequency NSGT filters are not supported")

    common_m = int(M_np[1])
    rows = cp.zeros((n_bins, common_m), dtype=cp.complex128)

    # All positive-frequency filters share common_m due to rasterize='full'.
    # Construct each folded filter row, then execute one batched IFFT.
    for row, i in enumerate(range(1, n_bins + 1)):
        lg = int(lengths[i])
        half_floor = lg // 2
        half_ceil = (lg + 1) // 2
        idx = np.concatenate((
            np.arange(half_ceil, lg, dtype=np.int64),
            np.arange(0, half_ceil, dtype=np.int64),
        ))
        offsets = np.arange(-half_floor, half_ceil, dtype=np.int64)
        win_range = (int(posit[i]) + offsets) % total
        window_gpu = cp.asarray(windows[i], dtype=cp.float64)
        vals = spectrum[cp.asarray(win_range)] * window_gpu[cp.asarray(idx)]

        if common_m < lg:
            col = int(np.ceil(lg / common_m))
            temp = cp.zeros(col * common_m, dtype=cp.complex128)
            target = np.concatenate((
                np.arange(col * common_m - half_floor, col * common_m, dtype=np.int64),
                np.arange(0, half_ceil, dtype=np.int64),
            ))
            temp[cp.asarray(target)] = vals
            temp = temp.reshape(col, common_m).sum(axis=0)
        else:
            temp = cp.zeros(common_m, dtype=cp.complex128)
            target = np.concatenate((
                np.arange(common_m - half_floor, common_m, dtype=np.int64),
                np.arange(0, half_ceil, dtype=np.int64),
            ))
            temp[cp.asarray(target)] = vals
            displace = int(posit[i] - np.floor(posit[i] / common_m) * common_m)
            if displace:
                temp = cp.roll(temp, displace)

        rows[row] = temp

    positive_coefficients = cp.fft.ifft(rows, axis=1)
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(positive_coefficients)


def cqcc_cqt_magnitude(
    x,
    sr: int,
    bins: int = 96,
    fmin: float | None = None,
    fmax: float | None = None,
    gamma: float | None = None,
    device: str = "cpu",
):
    """Return the MATLAB-compatible complex positive-frequency CQT.

    The returned tuple is (c, windows, shift, M, metadata). `c` is the
    NSGT coefficient list; the positive-frequency CQT is c[1:Lfbas+1].
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("Input waveform is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input waveform contains NaN or infinity")
    if sr <= 0 or bins <= 0:
        raise ValueError("sr and bins must be positive")

    fmin = sr / (2 ** 10) if fmin is None else float(fmin)
    fmax = sr / 2 if fmax is None else float(fmax)
    if not (0 < fmin < fmax):
        raise ValueError("Require 0 < fmin < fmax")

    if gamma is None:
        gamma = 228.7 * (
            2.0 ** (1.0 / bins) - 2.0 ** (-1.0 / bins)
        )

    windows, shift, M, meta = nsgcqwin(
        fmin,
        fmax,
        bins,
        sr,
        len(x),
        min_win=4,
        bwfac=1,
        fractional=False,
        winfun="hann",
        gamma=gamma,
    )

    # cqt.m: rasterize='full'. The two special DC/Nyquist lengths
    # remain unchanged; all other filters use the common raster length.
    n_bins = int(meta["Lfbas"])
    M_transform = M.copy()
    common_M = int(M[n_bins])
    M_transform[1:n_bins + 1] = common_M
    M_transform[n_bins + 2:] = M[n_bins:0:-1]

    # cqt.m: normalize='sine'. Apply the positive-frequency
    # normalization factors and mirror them for negative frequencies.
    positive_norm = 2.0 * M_transform[:n_bins + 2] / len(x)
    norm_factors = np.concatenate(
        (positive_norm, positive_norm[-2:0:-1])
    )
    if len(norm_factors) != len(windows):
        raise RuntimeError(
            "CQT normalization length does not match filterbank length"
        )
    windows = [
        window * norm_factors[i]
        for i, window in enumerate(windows)
    ]

    if device == "cpu":
        coefficients = nsgtf_real(
            x, windows, shift, M_transform, phasemode="global"
        )
    elif device == "cuda":
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CUDA extraction requires CuPy. Install a compatible CuPy "
                "wheel (for CUDA 12.x, try: pip install cupy-cuda12x)."
            ) from exc
        positive = _cqcc_positive_cqt_gpu(
            x, windows, shift, M_transform, meta, cp=cp
        )
        # Keep the public return shape/type identical to the CPU path.
        coefficients = [None] + [positive[i] for i in range(positive.shape[0])]
    else:
        raise ValueError("device must be 'cpu' or 'cuda'")
    return coefficients, windows, shift, M_transform, meta


def _matlab_deltas(features: np.ndarray, hlen: int = 3) -> np.ndarray:
    """Reproduce the MATLAB Deltas.m filter and edge replication."""
    from scipy.signal import lfilter

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Delta input must be a 2-D coefficient matrix")
    if hlen < 1:
        raise ValueError("hlen must be >= 1")

    # MATLAB: win = hlen:-1:-hlen
    win = np.arange(hlen, -hlen - 1, -1, dtype=np.float64)

    # MATLAB repeats the first and last frames hlen times.
    padded = np.pad(
        features,
        ((0, 0), (hlen, hlen)),
        mode="edge",
    )

    # MATLAB: filter(win, 1, xx, [], 2)
    delta = lfilter(win, [1.0], padded, axis=1)

    # MATLAB: D(:, hlen*2+1:end)
    delta = delta[:, 2 * hlen:]

    denominator = 2.0 * np.sum(
        np.arange(1, hlen + 1, dtype=np.float64) ** 2
    )
    return delta / denominator


def extract_cqcc(
    x,
    sr: int,
    bins: int = 96,
    fmax: float | None = None,
    fmin: float | None = None,
    d: int = 16,
    cf: int = 29,
    delta_window: int = 3,
    device: str = "cpu",
) -> np.ndarray:
    """Extract MATLAB-compatible CQCC features for one mono utterance.

    Defaults correspond to the ASVspoof baseline call:
        cqcc(x, fs, 96, fs/2, fs/2**10, 16, 29, 'ZsdD')

    Returns
    -------
    np.ndarray
        Float64 matrix with shape (3 * (cf + 1), frames): static,
        delta, and delta-delta coefficients concatenated vertically.
    """
    from scipy.interpolate import CubicSpline
    from scipy.fft import dct

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("Input waveform is empty")
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if d <= 0:
        raise ValueError("CQCC resampling factor d must be positive")
    if cf < 0:
        raise ValueError("cf must be non-negative")

    fmin = sr / (2 ** 10) if fmin is None else float(fmin)
    fmax = sr / 2 if fmax is None else float(fmax)

    coefficients, _, _, _, meta = cqcc_cqt_magnitude(
        x=x,
        sr=sr,
        bins=bins,
        fmin=fmin,
        fmax=fmax,
    )

    n_bins = int(meta["Lfbas"])
    if cf >= 0 and cf + 1 > n_bins:
        raise ValueError(
            f"Requested {cf + 1} cepstral coefficients, "
            f"but CQT provides only {n_bins} positive-frequency bins"
        )

    # cqt.m returns the positive-frequency filters between DC and Nyquist.
    positive_cqt = np.stack(
        coefficients[1:n_bins + 1],
        axis=0,
    )

    # cqcc.m: log(abs(CQT).^2 + eps)
    log_power = np.log(
        np.abs(positive_cqt) ** 2 + np.finfo(np.float64).eps
    )

    # cqcc.m: nonuniform CQT center-frequency grid.
    freq_vec = fmin * 2.0 ** (
        np.arange(n_bins, dtype=np.float64) / bins
    )

    # MATLAB:
    # kl = bins * log2(1 + 1/d)
    # resample(..., 1/(fmin*(2^(kl/bins)-1)), 1, 1, 'spline')
    # The uniform grid begins at FreqVec(1), with the final point
    # not exceeding FreqVec(end).
    frequency_step = fmin * (2.0 ** (
        np.log2(1.0 + 1.0 / d)
    ) - 1.0)
    n_uniform = int(
        np.floor((freq_vec[-1] - freq_vec[0]) / frequency_step)
    ) + 1
    uniform_freq_vec = (
        freq_vec[0]
        + np.arange(n_uniform, dtype=np.float64) * frequency_step
    )

    # MATLAB resample(..., 'spline'); validated against the saved
    # MATLAB Ures_LogP_absCQT reference using SciPy CubicSpline.
    spline = CubicSpline(freq_vec, log_power, axis=0)
    uniform_log_power = spline(uniform_freq_vec)

    # MATLAB dct() default corresponds to orthonormal DCT-II here.
    cepstrum = dct(
        uniform_log_power,
        type=2,
        axis=0,
        norm="ortho",
    )

    static = cepstrum[:cf + 1, :]
    delta = _matlab_deltas(static, hlen=delta_window)
    delta_delta = _matlab_deltas(delta, hlen=delta_window)

    cqcc = np.vstack((static, delta, delta_delta))
    expected_rows = 3 * (cf + 1)
    if cqcc.shape[0] != expected_rows:
        raise RuntimeError(
            f"Unexpected CQCC dimension {cqcc.shape[0]}; "
            f"expected {expected_rows}"
        )
    if not np.all(np.isfinite(cqcc)):
        raise FloatingPointError("CQCC extraction produced non-finite values")

    return np.asarray(cqcc, dtype=np.float64)

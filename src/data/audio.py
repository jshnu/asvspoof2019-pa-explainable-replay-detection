
from __future__ import annotations

from pathlib import Path

import numpy as np
import av
import soundfile as sf


def inspect_audio(
    path: str | Path,
    expected_sample_rate: int = 16000,
    allowed_channels: tuple[int, ...] = (1, 2),
    min_duration_seconds: float = 0.10,
    max_duration_seconds: float | None = None,
    decode_check: bool = True,
) -> dict:
    """
    Inspect one audio file.

    Metadata is read using soundfile.
    Actual FLAC decoding is performed using PyAV.

    Audio is decoded frame by frame. The complete waveform
    is never retained in memory.
    """
    path = Path(path)

    result = {
        "audio_path": str(path),
        "exists": path.is_file(),
        "readable": False,
        "sample_rate": None,
        "channels": None,
        "frames": None,
        "duration_seconds": None,
        "subtype": None,
        "decoded_frames": None,
        "issues": [],
    }

    if not path.is_file():
        result["issues"].append("missing_file")
        return result

    try:
        # Read metadata.
        info = sf.info(str(path))

        sr = int(info.samplerate)
        channels = int(info.channels)
        frames = int(info.frames)
        duration = frames / sr if sr > 0 else 0.0

        result.update({
            "sample_rate": sr,
            "channels": channels,
            "frames": frames,
            "duration_seconds": duration,
            "subtype": info.subtype,
        })

        # Validate metadata.
        if sr != expected_sample_rate:
            result["issues"].append(
                f"unexpected_sample_rate:{sr}"
            )

        if channels not in allowed_channels:
            result["issues"].append(
                f"unexpected_channel_count:{channels}"
            )

        if duration < min_duration_seconds:
            result["issues"].append(
                "duration_below_minimum"
            )

        if (
            max_duration_seconds is not None
            and duration > max_duration_seconds
        ):
            result["issues"].append(
                "duration_above_maximum"
            )

        # Decode audio using PyAV.
        if decode_check:
            decoded_samples = 0

            with av.open(str(path), mode="r") as container:
                if not container.streams.audio:
                    raise ValueError("No audio stream found")

                stream = container.streams.audio[0]

                for frame in container.decode(stream):
                    decoded_samples += frame.samples

            result["decoded_frames"] = decoded_samples

            # Verify decoded sample count.
            if decoded_samples != frames:
                result["issues"].append(
                    f"decoded_frame_mismatch:"
                    f"metadata={frames},decoded={decoded_samples}"
                )
                return result

        result["readable"] = True

    except Exception as exc:
        result["issues"].append(
            f"inspection_error:{type(exc).__name__}:{exc}"
        )

    return result


def load_audio(
    path: str | Path,
    expected_sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:
    """
    Decode an audio file using PyAV.

    Returns:
        waveform: float32 NumPy array, shape (num_samples,)
        sample_rate: int

    Preserves the original sample rate.
    Does not resample, normalize, or trim silence.
    """
    path = Path(path)
    chunks = []

    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream: {path}")

        stream = container.streams.audio[0]

        if stream.codec_context.sample_rate != expected_sample_rate:
            raise ValueError(
                f"Unexpected sample rate in {path}: "
                f"{stream.codec_context.sample_rate}"
            )

        for frame in container.decode(stream):
            # Convert decoded samples to float32.
            chunk = frame.to_ndarray()

            # ASVspoof PA is mono.
            if frame.layout.nb_channels != 1:
                raise ValueError(f"Expected mono audio: {path}")

            chunk = chunk.reshape(-1)

            # PyAV may return integer PCM samples.
            # Convert signed 16-bit PCM to [-1, 1).
            if np.issubdtype(chunk.dtype, np.integer):
                bits = np.iinfo(chunk.dtype).bits
                scale = float(2 ** (bits - 1))
                chunk = chunk.astype(np.float32) / scale
            else:
                chunk = chunk.astype(np.float32)

            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No decoded samples: {path}")

    waveform = np.concatenate(chunks)

    if not np.all(np.isfinite(waveform)):
        raise ValueError(f"Non-finite audio samples: {path}")

    return waveform, expected_sample_rate


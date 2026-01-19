"""Audio processing utilities for normalization and volume matching."""

import numpy as np


def normalize_audio(audio: np.ndarray, target_peak: float = 0.99) -> np.ndarray:
    """
    Normalize audio to a target peak amplitude.

    Args:
        audio: Input audio samples
        target_peak: Target peak amplitude (default: 0.99 to avoid clipping)

    Returns:
        Normalized audio with peak at target_peak
    """
    audio = np.asarray(audio, dtype=np.float32)
    max_amp = np.abs(audio).max()

    if max_amp > 0:
        audio = audio * (target_peak / max_amp)

    return audio


def compute_rms(audio: np.ndarray) -> float:
    """
    Compute the RMS (Root Mean Square) energy of audio.

    Args:
        audio: Input audio samples

    Returns:
        RMS value
    """
    return np.sqrt(np.mean(audio ** 2))


def change_rms(
    source_audio: np.ndarray,
    target_audio: np.ndarray,
    rate: float = 1.0,
) -> np.ndarray:
    """
    Adjust the RMS energy of target audio to match source audio.

    This is used to preserve the loudness characteristics of the input
    after voice conversion.

    Args:
        source_audio: Reference audio (original input)
        target_audio: Audio to adjust (converted output)
        rate: Blend rate between original (0) and adjusted (1) RMS.
              1.0 = full RMS matching, 0.0 = no change

    Returns:
        Target audio with adjusted RMS
    """
    if rate == 0:
        return target_audio

    source_audio = np.asarray(source_audio, dtype=np.float32)
    target_audio = np.asarray(target_audio, dtype=np.float32)

    rms_source = compute_rms(source_audio)
    rms_target = compute_rms(target_audio)

    if rms_target == 0:
        return target_audio

    # Calculate the gain needed to match RMS
    gain = rms_source / rms_target

    # Apply rate-weighted gain
    if rate < 1.0:
        # Blend between no adjustment (gain=1) and full adjustment
        gain = 1.0 + rate * (gain - 1.0)

    adjusted = target_audio * gain

    return adjusted


def pad_audio(
    audio: np.ndarray,
    pad_seconds: float,
    sr: int,
    mode: str = "reflect",
) -> tuple[np.ndarray, int]:
    """
    Pad audio at both ends for processing context.

    Args:
        audio: Input audio samples
        pad_seconds: Seconds of padding to add on each side
        sr: Sample rate
        mode: Padding mode ('reflect', 'constant', 'edge')

    Returns:
        Tuple of (padded_audio, pad_samples) where pad_samples is the
        number of samples added to each side
    """
    pad_samples = int(pad_seconds * sr)

    if pad_samples == 0:
        return audio, 0

    if mode == "reflect":
        padded = np.pad(audio, pad_samples, mode="reflect")
    elif mode == "constant":
        padded = np.pad(audio, pad_samples, mode="constant", constant_values=0)
    elif mode == "edge":
        padded = np.pad(audio, pad_samples, mode="edge")
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    return padded, pad_samples


def remove_padding(
    audio: np.ndarray,
    pad_samples: int,
) -> np.ndarray:
    """
    Remove padding from processed audio.

    Args:
        audio: Padded audio
        pad_samples: Number of samples to remove from each side

    Returns:
        Unpadded audio
    """
    if pad_samples == 0:
        return audio

    return audio[pad_samples:-pad_samples]


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to a different sample rate using linear interpolation.

    Note: For high-quality resampling, prefer using load_audio with the
    target sample rate, which uses FFmpeg's resampler.

    Args:
        audio: Input audio samples
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio

    # Calculate the resampling ratio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)

    # Use linear interpolation for simple resampling
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    resampled = np.interp(new_indices, old_indices, audio)

    return resampled.astype(np.float32)


def to_int16(audio: np.ndarray) -> np.ndarray:
    """
    Convert float32 audio to int16 format.

    Args:
        audio: Float32 audio in range [-1, 1]

    Returns:
        Int16 audio in range [-32768, 32767]
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Normalize to prevent clipping
    max_amp = np.abs(audio).max()
    if max_amp > 0.99:
        audio = audio * (0.99 / max_amp)

    # Convert to int16
    return (audio * 32767).astype(np.int16)


def to_float32(audio: np.ndarray) -> np.ndarray:
    """
    Convert int16 audio to float32 format.

    Args:
        audio: Int16 audio in range [-32768, 32767]

    Returns:
        Float32 audio in range [-1, 1]
    """
    return audio.astype(np.float32) / 32768.0

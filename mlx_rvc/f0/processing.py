"""F0 processing utilities for pitch shifting and quantization."""

import numpy as np
from scipy import signal


# F0 range constants (matching RVC)
F0_MIN = 50.0
F0_MAX = 1100.0

# Mel scale constants for the F0 range
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


def shift_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
    """
    Shift F0 by a number of semitones.

    Args:
        f0: F0 contour (Hz). Unvoiced frames should be 0.
        semitones: Number of semitones to shift (positive = higher pitch)

    Returns:
        Shifted F0 contour
    """
    f0 = np.asarray(f0, dtype=np.float32).copy()

    # Only shift voiced frames (f0 > 0)
    voiced_mask = f0 > 0
    f0[voiced_mask] *= np.power(2.0, semitones / 12.0)

    return f0


def f0_to_mel(f0: np.ndarray) -> np.ndarray:
    """
    Convert F0 from Hz to mel scale.

    Uses the formula: mel = 1127 * ln(1 + f/700)

    Args:
        f0: F0 in Hz

    Returns:
        F0 in mel scale
    """
    f0 = np.asarray(f0, dtype=np.float32)
    f0_mel = np.zeros_like(f0)

    voiced_mask = f0 > 0
    f0_mel[voiced_mask] = 1127 * np.log(1 + f0[voiced_mask] / 700)

    return f0_mel


def f0_to_coarse(f0: np.ndarray) -> np.ndarray:
    """
    Convert F0 to coarse quantized representation (1-255).

    This is the format expected by the RVC synthesizer's pitch embedding.
    Unvoiced frames (f0=0) are mapped to 0.

    Args:
        f0: F0 contour in Hz

    Returns:
        Quantized F0 as int32 array with values in [0, 255]
        0 = unvoiced, 1-255 = voiced pitch levels
    """
    f0 = np.asarray(f0, dtype=np.float32)

    # Convert to mel scale
    f0_mel = f0_to_mel(f0)

    # Normalize to 1-255 range
    voiced_mask = f0_mel > 0
    f0_mel[voiced_mask] = (
        (f0_mel[voiced_mask] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    )

    # Clamp to valid range
    f0_mel = np.clip(f0_mel, 0, 255)

    # Round to integers
    f0_coarse = np.rint(f0_mel).astype(np.int32)

    return f0_coarse


def smooth_f0(f0: np.ndarray, filter_radius: int = 3) -> np.ndarray:
    """
    Smooth F0 contour using median filtering.

    Args:
        f0: F0 contour
        filter_radius: Median filter kernel size (must be odd)

    Returns:
        Smoothed F0 contour
    """
    if filter_radius < 3:
        return f0

    # Ensure kernel size is odd
    kernel_size = filter_radius if filter_radius % 2 == 1 else filter_radius + 1

    return signal.medfilt(f0, kernel_size)


def interpolate_f0(f0: np.ndarray, target_length: int) -> np.ndarray:
    """
    Interpolate F0 contour to a target length.

    Args:
        f0: F0 contour
        target_length: Desired output length

    Returns:
        Interpolated F0 contour
    """
    if len(f0) == target_length:
        return f0

    # Create interpolation indices
    old_indices = np.arange(len(f0))
    new_indices = np.linspace(0, len(f0) - 1, target_length)

    # Interpolate
    f0_interp = np.interp(new_indices, old_indices, f0)

    return f0_interp.astype(np.float32)


def pad_f0(f0: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate F0 to match target length.

    Args:
        f0: F0 contour
        target_length: Desired output length

    Returns:
        Padded/truncated F0 contour
    """
    current_length = len(f0)

    if current_length == target_length:
        return f0
    elif current_length > target_length:
        return f0[:target_length]
    else:
        # Pad with zeros (unvoiced)
        pad_size = target_length - current_length
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        return np.pad(f0, (pad_left, pad_right), mode="constant", constant_values=0)


def process_f0(
    f0: np.ndarray,
    target_length: int | None = None,
    transpose: float = 0.0,
    filter_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full F0 processing pipeline for RVC inference.

    Args:
        f0: Raw F0 contour from extraction (Hz)
        target_length: Target number of frames (if None, keeps original length)
        transpose: Pitch shift in semitones
        filter_radius: Median filter size for smoothing

    Returns:
        Tuple of (f0_coarse, f0_raw):
        - f0_coarse: Quantized pitch (int32, 0-255) for embedding lookup
        - f0_raw: Processed pitch in Hz (float32) for NSF decoder
    """
    f0 = np.asarray(f0, dtype=np.float32).copy()

    # Apply smoothing
    if filter_radius >= 3:
        f0 = smooth_f0(f0, filter_radius)

    # Apply pitch shift
    if transpose != 0.0:
        f0 = shift_f0(f0, transpose)

    # Adjust length if needed
    if target_length is not None and len(f0) != target_length:
        f0 = pad_f0(f0, target_length)

    # Keep raw f0 for NSF
    f0_raw = f0.copy()

    # Convert to coarse quantized format for embedding
    f0_coarse = f0_to_coarse(f0)

    return f0_coarse, f0_raw

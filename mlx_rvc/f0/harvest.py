"""F0 extraction using PyWorld's Harvest algorithm."""

import numpy as np

try:
    import pyworld
except ImportError:
    pyworld = None


def extract_f0_harvest(
    audio: np.ndarray,
    sr: int = 16000,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
    frame_period: float = 10.0,
    use_stonemask: bool = True,
) -> np.ndarray:
    """
    Extract F0 (fundamental frequency) using PyWorld's Harvest algorithm.

    Harvest is a high-quality F0 estimation algorithm that provides
    accurate pitch tracking, especially for singing voice.

    Args:
        audio: Input audio samples (mono, float)
        sr: Sample rate (default: 16000)
        f0_min: Minimum F0 to detect in Hz (default: 50)
        f0_max: Maximum F0 to detect in Hz (default: 1100)
        frame_period: Frame period in milliseconds (default: 10ms)
        use_stonemask: Whether to refine F0 with StoneMask (default: True)

    Returns:
        F0 contour as numpy array with shape (num_frames,).
        Unvoiced frames have F0 = 0.

    Raises:
        ImportError: If pyworld is not installed
    """
    if pyworld is None:
        raise ImportError(
            "pyworld is required for Harvest F0 extraction. "
            "Install it with: pip install pyworld"
        )

    # Ensure audio is float64 for pyworld
    audio = np.asarray(audio, dtype=np.float64)

    # Extract F0 using Harvest
    f0, t = pyworld.harvest(
        audio,
        fs=sr,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        frame_period=frame_period,
    )

    # Refine F0 with StoneMask for better accuracy
    if use_stonemask:
        f0 = pyworld.stonemask(audio, f0, t, sr)

    return f0.astype(np.float32)


def get_f0_frame_count(audio_length: int, sr: int = 16000, window: int = 160) -> int:
    """
    Calculate the expected number of F0 frames for a given audio length.

    Args:
        audio_length: Number of audio samples
        sr: Sample rate
        window: Window size in samples (default: 160 for 10ms at 16kHz)

    Returns:
        Expected number of F0 frames
    """
    # Frame period in seconds
    frame_period = window / sr

    # Number of frames (pyworld adds some frames at boundaries)
    return int(np.ceil(audio_length / sr / frame_period))

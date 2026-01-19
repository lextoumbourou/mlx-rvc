"""Audio file loading and saving utilities using FFmpeg."""

import subprocess
import numpy as np
from pathlib import Path


def load_audio(file_path: str | Path, sr: int = 16000) -> np.ndarray:
    """
    Load an audio file and resample to the target sample rate.

    Uses FFmpeg subprocess for broad format support and resampling.

    Args:
        file_path: Path to the audio file (supports any FFmpeg-compatible format)
        sr: Target sample rate (default: 16000 for ContentVec)

    Returns:
        Audio samples as float32 numpy array, mono, normalized to [-1, 1]

    Raises:
        FileNotFoundError: If the audio file doesn't exist
        RuntimeError: If FFmpeg fails to decode the file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # FFmpeg command to decode audio to raw PCM float32
    cmd = [
        "ffmpeg",
        "-nostdin",           # Don't read from stdin
        "-threads", "0",      # Auto-detect thread count
        "-i", str(file_path), # Input file
        "-f", "f32le",        # Output format: 32-bit float little-endian
        "-acodec", "pcm_f32le",
        "-ac", "1",           # Mono
        "-ar", str(sr),       # Target sample rate
        "-",                  # Output to stdout
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed to decode audio: {stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
        )

    # Convert raw bytes to numpy array
    audio = np.frombuffer(result.stdout, dtype=np.float32)

    return audio


def save_audio(
    file_path: str | Path,
    audio: np.ndarray,
    sr: int,
    format: str | None = None,
) -> None:
    """
    Save audio data to a file.

    Args:
        file_path: Output file path
        audio: Audio samples as numpy array (will be converted to float32)
        sr: Sample rate of the audio
        format: Output format (e.g., 'wav', 'mp3', 'flac').
                If None, inferred from file extension.

    Raises:
        RuntimeError: If FFmpeg fails to encode the file
    """
    file_path = Path(file_path)

    # Ensure audio is float32 and mono
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.flatten()

    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)

    # Infer format from extension if not specified
    if format is None:
        format = file_path.suffix.lstrip(".").lower()
        if not format:
            format = "wav"

    # Map common format aliases
    format_map = {
        "m4a": "ipod",  # FFmpeg uses 'ipod' for m4a
    }
    output_format = format_map.get(format, format)

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",                 # Overwrite output file
        "-nostdin",
        "-f", "f32le",        # Input format: raw float32
        "-ar", str(sr),       # Input sample rate
        "-ac", "1",           # Input channels: mono
        "-i", "-",            # Input from stdin
        "-f", output_format,  # Output format
    ]

    # Add format-specific options
    if format == "wav":
        cmd.extend(["-acodec", "pcm_s16le"])  # 16-bit PCM for WAV
    elif format == "mp3":
        cmd.extend(["-acodec", "libmp3lame", "-q:a", "2"])
    elif format == "flac":
        cmd.extend(["-acodec", "flac"])
    elif format in ("m4a", "aac"):
        cmd.extend(["-acodec", "aac", "-b:a", "192k"])
    elif format == "ogg":
        cmd.extend(["-acodec", "libvorbis", "-q:a", "5"])

    cmd.append(str(file_path))

    try:
        subprocess.run(
            cmd,
            input=audio.tobytes(),
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed to encode audio: {stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
        )


def get_audio_duration(file_path: str | Path) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        file_path: Path to the audio file

    Returns:
        Duration in seconds
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        return float(result.stdout.decode().strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get audio duration: {e}")

"""Tests for audio loading and processing modules."""

import tempfile
import numpy as np
import pytest
from pathlib import Path

from mlx_rvc.audio.io import load_audio, save_audio
from mlx_rvc.audio.processing import (
    normalize_audio,
    change_rms,
    compute_rms,
    pad_audio,
    remove_padding,
    to_int16,
    to_float32,
)


class TestAudioProcessing:
    """Tests for audio processing utilities."""

    def test_normalize_audio(self):
        """Test audio normalization to target peak."""
        audio = np.array([0.5, -0.3, 0.2], dtype=np.float32)
        normalized = normalize_audio(audio, target_peak=0.99)

        assert np.abs(normalized).max() == pytest.approx(0.99, rel=1e-5)
        # Check relative amplitudes preserved (0.5 > 0.3 > 0.2 in absolute value)
        assert np.abs(normalized[0]) > np.abs(normalized[1]) > np.abs(normalized[2])

    def test_normalize_silent_audio(self):
        """Test normalization of silent audio doesn't cause division by zero."""
        audio = np.zeros(100, dtype=np.float32)
        normalized = normalize_audio(audio)
        assert np.allclose(normalized, 0)

    def test_compute_rms(self):
        """Test RMS computation."""
        # Sine wave with known RMS
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        rms = compute_rms(sine)
        # RMS of sine wave is amplitude / sqrt(2)
        assert rms == pytest.approx(1.0 / np.sqrt(2), rel=0.01)

    def test_change_rms(self):
        """Test RMS matching between audio signals."""
        source = np.random.randn(16000).astype(np.float32) * 0.5
        target = np.random.randn(16000).astype(np.float32) * 0.1

        adjusted = change_rms(source, target, rate=1.0)

        source_rms = compute_rms(source)
        adjusted_rms = compute_rms(adjusted)
        assert adjusted_rms == pytest.approx(source_rms, rel=0.01)

    def test_change_rms_rate_zero(self):
        """Test that rate=0 returns unchanged audio."""
        source = np.random.randn(16000).astype(np.float32)
        target = np.random.randn(16000).astype(np.float32)

        adjusted = change_rms(source, target, rate=0.0)
        assert np.allclose(adjusted, target)

    def test_pad_audio(self):
        """Test audio padding."""
        audio = np.ones(16000, dtype=np.float32)
        padded, pad_samples = pad_audio(audio, pad_seconds=0.1, sr=16000)

        assert pad_samples == 1600
        assert len(padded) == 16000 + 2 * 1600

    def test_remove_padding(self):
        """Test padding removal."""
        audio = np.arange(100, dtype=np.float32)
        padded, pad_samples = pad_audio(audio, pad_seconds=0.01, sr=1000)
        unpadded = remove_padding(padded, pad_samples)

        assert len(unpadded) == len(audio)

    def test_to_int16(self):
        """Test float32 to int16 conversion."""
        audio = np.array([0.5, -0.5, 0.0, 1.0, -1.0], dtype=np.float32)
        int16_audio = to_int16(audio)

        assert int16_audio.dtype == np.int16
        assert int16_audio[2] == 0
        # Values should be scaled but clipped
        assert np.abs(int16_audio).max() <= 32767

    def test_to_float32(self):
        """Test int16 to float32 conversion."""
        int16_audio = np.array([16384, -16384, 0], dtype=np.int16)
        float_audio = to_float32(int16_audio)

        assert float_audio.dtype == np.float32
        assert float_audio[0] == pytest.approx(0.5, rel=0.01)
        assert float_audio[1] == pytest.approx(-0.5, rel=0.01)
        assert float_audio[2] == 0.0


class TestAudioIO:
    """Tests for audio file I/O."""

    def test_save_and_load_wav(self):
        """Test saving and loading a WAV file."""
        # Create test audio: 1 second of 440Hz sine wave
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save
            save_audio(temp_path, audio, sr)
            assert temp_path.exists()

            # Load
            loaded = load_audio(temp_path, sr=sr)

            # Check shape and approximate values
            assert len(loaded) == pytest.approx(sr, rel=0.01)
            # Audio should be similar (some quantization loss expected)
            correlation = np.corrcoef(audio[:len(loaded)], loaded[:len(audio)])[0, 1]
            assert correlation > 0.99
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.wav")

    def test_load_with_resampling(self):
        """Test loading audio with different sample rate."""
        sr_original = 44100
        sr_target = 16000
        duration = 0.5

        t = np.linspace(0, duration, int(sr_original * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_audio(temp_path, audio, sr_original)
            loaded = load_audio(temp_path, sr=sr_target)

            # Should be resampled to target SR
            expected_samples = int(sr_target * duration)
            assert len(loaded) == pytest.approx(expected_samples, rel=0.05)
        finally:
            temp_path.unlink(missing_ok=True)

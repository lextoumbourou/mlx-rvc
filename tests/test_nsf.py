"""Tests for NSF (Neural Source Filter) components."""

import numpy as np
import pytest
import mlx.core as mx

from rvc_mlx.models.nsf import SineGen, SourceModuleHnNSF

torch = pytest.importorskip("torch")
from torch import nn
import sys
sys.path.insert(0, "vendor/Retrieval-based-Voice-Conversion-WebUI")


class TestSineGen:
    """Test SineGen sine wave generation."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        batch, length, sr, upp = 2, 100, 16000, 160

        sine_gen = SineGen(sample_rate=sr, harmonic_num=0)

        f0 = mx.ones((batch, length)) * 440.0  # 440 Hz
        sine, uv, noise = sine_gen(f0, upp)

        assert sine.shape == (batch, length * upp, 1)
        assert uv.shape == (batch, length * upp, 1)
        assert noise.shape == (batch, length * upp, 1)

    def test_output_shape_with_harmonics(self):
        """Test output shape with harmonics."""
        batch, length, sr, upp = 2, 100, 16000, 160
        harmonic_num = 8

        sine_gen = SineGen(sample_rate=sr, harmonic_num=harmonic_num)

        f0 = mx.ones((batch, length)) * 440.0
        sine, uv, noise = sine_gen(f0, upp)

        expected_dim = harmonic_num + 1
        assert sine.shape == (batch, length * upp, expected_dim)

    def test_unvoiced_frames(self):
        """Test that unvoiced frames (f0=0) produce noise, not sine."""
        batch, length, sr, upp = 1, 10, 16000, 160

        sine_gen = SineGen(sample_rate=sr, harmonic_num=0)

        # All unvoiced
        f0 = mx.zeros((batch, length))
        sine, uv, noise = sine_gen(f0, upp)

        # UV should be all zeros
        assert np.all(np.array(uv) == 0)

    def test_voiced_frames(self):
        """Test that voiced frames produce UV=1."""
        batch, length, sr, upp = 1, 10, 16000, 160

        sine_gen = SineGen(sample_rate=sr, harmonic_num=0)

        # All voiced at 440 Hz
        f0 = mx.ones((batch, length)) * 440.0
        sine, uv, noise = sine_gen(f0, upp)

        # UV should be all ones
        assert np.all(np.array(uv) == 1)

    def test_sine_amplitude_range(self):
        """Test that sine amplitude is in expected range."""
        batch, length, sr, upp = 1, 100, 16000, 160
        sine_amp = 0.1

        sine_gen = SineGen(sample_rate=sr, harmonic_num=0, sine_amp=sine_amp)

        f0 = mx.ones((batch, length)) * 440.0
        sine, uv, noise = sine_gen(f0, upp)

        sine_np = np.array(sine)
        # With noise, values should be around sine_amp but can exceed slightly
        assert np.abs(sine_np).max() < sine_amp * 2


class TestSourceModuleHnNSF:
    """Test SourceModuleHnNSF source generation."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        batch, length, sr, upp = 2, 100, 48000, 480

        source_module = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        f0 = mx.ones((batch, length)) * 440.0
        source, _, _ = source_module(f0, upp)

        # Output should be (batch, length*upp, 1)
        assert source.shape == (batch, length * upp, 1)

    def test_output_range(self):
        """Test that output is in tanh range [-1, 1]."""
        batch, length, sr, upp = 1, 100, 48000, 480

        source_module = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        f0 = mx.ones((batch, length)) * 440.0
        source, _, _ = source_module(f0, upp)

        source_np = np.array(source)
        assert source_np.min() >= -1.0
        assert source_np.max() <= 1.0

    def test_load_checkpoint_weights(self):
        """Test loading weights from RVC checkpoint."""
        import os

        ckpt_path = "vendor/weights/f0G48k.pth"
        if not os.path.exists(ckpt_path):
            pytest.skip("Checkpoint not found")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ckpt["model"]

        # Get source module weights
        linear_w = model["dec.m_source.l_linear.weight"].float().numpy()  # (1, 1)
        linear_b = model["dec.m_source.l_linear.bias"].float().numpy()  # (1,)

        print(f"Linear weight shape: {linear_w.shape}")
        print(f"Linear bias shape: {linear_b.shape}")

        # Create MLX module
        mlx_source = SourceModuleHnNSF(sample_rate=48000, harmonic_num=0)

        # Load weights
        # MLX Linear weight shape should be (out_features, in_features) = (1, 1)
        mlx_source.l_linear.weight = mx.array(linear_w.astype(np.float32))
        mlx_source.l_linear.bias = mx.array(linear_b.astype(np.float32))

        # Test forward pass
        f0 = mx.ones((1, 100)) * 440.0
        source, _, _ = mlx_source(f0, upp=480)

        assert source.shape == (1, 100 * 480, 1)
        # Output should be in tanh range
        source_np = np.array(source)
        assert source_np.min() >= -1.0
        assert source_np.max() <= 1.0


class TestSineGenVsPyTorch:
    """Compare SineGen output with PyTorch reference."""

    def test_uv_mask_matches(self):
        """Test that UV mask generation matches PyTorch."""
        try:
            from infer.lib.infer_pack.models import SineGen as PTSineGen
        except ImportError:
            pytest.skip("PyTorch RVC not available")

        batch, length, sr = 1, 10, 16000

        # Create both models
        pt_sine = PTSineGen(sr, harmonic_num=0)
        mlx_sine = SineGen(sr, harmonic_num=0)

        # Test F0 with mix of voiced/unvoiced
        f0_np = np.array([[440, 0, 440, 0, 440, 0, 440, 440, 0, 440]], dtype=np.float32)

        # Get UV from both
        pt_uv = pt_sine._f02uv(torch.from_numpy(f0_np[:, :, None])).numpy()

        f0_mlx = mx.array(f0_np)[:, :, None]
        mlx_uv = np.array(mlx_sine._f02uv(f0_mlx))

        np.testing.assert_array_equal(mlx_uv, pt_uv)

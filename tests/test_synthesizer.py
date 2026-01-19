"""Tests for SynthesizerTrnMs768NSFsid implementation."""

import numpy as np
import pytest
import mlx.core as mx

from mlx_rvc.models import SynthesizerTrnMs768NSFsid


class TestSynthesizerBasic:
    """Basic tests for SynthesizerTrnMs768NSFsid."""

    # 48kHz config
    CONFIG_48K = {
        "spec_channels": 1025,
        "segment_size": 32,
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [12, 10, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [24, 20, 4, 4],
        "spk_embed_dim": 109,
        "gin_channels": 256,
        "sr": 48000,
    }

    def test_model_instantiation(self):
        """Test that the model can be instantiated."""
        model = SynthesizerTrnMs768NSFsid(**self.CONFIG_48K)
        assert model is not None

    def test_forward_pass_shapes(self):
        """Test that forward pass produces correct output shapes."""
        model = SynthesizerTrnMs768NSFsid(**self.CONFIG_48K)

        batch = 1
        length = 20  # Small for faster test

        # Create dummy inputs
        phone = mx.random.normal(shape=(batch, length, 768))  # ContentVec features
        phone_lengths = mx.array([length])
        pitch = mx.zeros((batch, length), dtype=mx.int32)  # Pitch indices (0-255)
        nsff0 = mx.ones((batch, length)) * 220.0  # F0 in Hz
        sid = mx.array([0])  # Speaker ID

        # Run inference
        audio, mask, _ = model.infer(phone, phone_lengths, pitch, nsff0, sid)

        # Check output shape
        # Audio should be (batch, 1, length * upp) where upp = 12 * 10 * 2 * 2 = 480
        expected_audio_length = length * 480
        assert audio.shape == (batch, 1, expected_audio_length), f"Audio shape: {audio.shape}"

        # Mask should be (batch, 1, length)
        assert mask.shape == (batch, 1, length), f"Mask shape: {mask.shape}"

    def test_output_range(self):
        """Test that output is in valid audio range [-1, 1]."""
        model = SynthesizerTrnMs768NSFsid(**self.CONFIG_48K)

        batch, length = 1, 10
        phone = mx.random.normal(shape=(batch, length, 768))
        phone_lengths = mx.array([length])
        pitch = mx.zeros((batch, length), dtype=mx.int32)
        nsff0 = mx.ones((batch, length)) * 220.0
        sid = mx.array([0])

        audio, _, _ = model.infer(phone, phone_lengths, pitch, nsff0, sid)
        audio_np = np.array(audio)

        # Output should be in tanh range
        assert audio_np.min() >= -1.0, f"Min audio: {audio_np.min()}"
        assert audio_np.max() <= 1.0, f"Max audio: {audio_np.max()}"


class TestSynthesizerComponents:
    """Test individual components of the synthesizer."""

    def test_text_encoder_shapes(self):
        """Test TextEncoder output shapes."""
        from mlx_rvc.models import TextEncoder

        enc = TextEncoder(
            in_channels=768,
            out_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0,
        )

        batch, length = 1, 20
        phone = mx.random.normal(shape=(batch, length, 768))
        pitch = mx.zeros((batch, length), dtype=mx.int32)
        lengths = mx.array([length])

        m, logs, mask = enc(phone, pitch, lengths)

        assert m.shape == (batch, 192, length)
        assert logs.shape == (batch, 192, length)
        assert mask.shape == (batch, 1, length)

    def test_flow_shapes(self):
        """Test ResidualCouplingBlock output shapes."""
        from mlx_rvc.models import ResidualCouplingBlock

        flow = ResidualCouplingBlock(
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=3,
            gin_channels=256,
        )

        batch, length = 1, 20
        z = mx.random.normal(shape=(batch, 192, length))
        z_mask = mx.ones((batch, 1, length))
        g = mx.random.normal(shape=(batch, 256, 1))

        # Forward (reverse=False)
        z_out = flow(z, z_mask, g=g, reverse=False)
        assert z_out.shape == z.shape

        # Inverse (reverse=True)
        z_inv = flow(z_out, z_mask, g=g, reverse=True)
        assert z_inv.shape == z.shape

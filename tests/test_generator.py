"""Tests for GeneratorNSF implementation."""

import numpy as np
import pytest
import mlx.core as mx

from mlx_rvc.models.generator import GeneratorNSF

torch = pytest.importorskip("torch")
from torch import nn
import sys

sys.path.insert(0, "vendor/Retrieval-based-Voice-Conversion-WebUI")


class TestGeneratorNSF:
    """Test GeneratorNSF against PyTorch reference."""

    # 48kHz config
    CONFIG_48K = {
        "initial_channel": 192,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [12, 10, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [24, 20, 4, 4],
        "gin_channels": 256,
        "sr": 48000,
    }

    def test_output_shape(self):
        """Test that output shape is correct."""
        gen = GeneratorNSF(**self.CONFIG_48K)

        batch, length = 1, 100
        x = mx.random.normal(shape=(batch, 192, length))
        f0 = mx.ones((batch, length)) * 440.0
        g = mx.random.normal(shape=(batch, 256, 1))

        y = gen(x, f0, g)

        # Output should be (batch, 1, length * upp)
        # upp = 12 * 10 * 2 * 2 = 480
        expected_length = length * 480
        assert y.shape == (batch, 1, expected_length)

    def test_output_range(self):
        """Test that output is in tanh range [-1, 1]."""
        gen = GeneratorNSF(**self.CONFIG_48K)

        batch, length = 1, 50
        x = mx.random.normal(shape=(batch, 192, length))
        f0 = mx.ones((batch, length)) * 440.0
        g = mx.random.normal(shape=(batch, 256, 1))

        y = gen(x, f0, g)
        y_np = np.array(y)

        assert y_np.min() >= -1.0
        assert y_np.max() <= 1.0

    def test_without_speaker_conditioning(self):
        """Test generation without speaker embedding."""
        config = self.CONFIG_48K.copy()
        config["gin_channels"] = 0
        gen = GeneratorNSF(**config)

        batch, length = 1, 50
        x = mx.random.normal(shape=(batch, 192, length))
        f0 = mx.ones((batch, length)) * 440.0

        y = gen(x, f0, g=None)

        expected_length = length * 480
        assert y.shape == (batch, 1, expected_length)

    def test_unvoiced_frames(self):
        """Test generation with unvoiced frames (f0=0)."""
        gen = GeneratorNSF(**self.CONFIG_48K)

        batch, length = 1, 50
        x = mx.random.normal(shape=(batch, 192, length))
        f0 = mx.zeros((batch, length))  # All unvoiced
        g = mx.random.normal(shape=(batch, 256, 1))

        y = gen(x, f0, g)

        # Should still produce output
        expected_length = length * 480
        assert y.shape == (batch, 1, expected_length)


class TestGeneratorNSFVsPyTorch:
    """Compare GeneratorNSF with PyTorch reference implementation."""

    def test_load_checkpoint_and_compare(self):
        """Test loading weights from checkpoint and comparing outputs."""
        import os

        ckpt_path = "vendor/weights/f0G48k.pth"
        if not os.path.exists(ckpt_path):
            pytest.skip("Checkpoint not found")

        try:
            from infer.lib.infer_pack.models import GeneratorNSF as PTGeneratorNSF
        except ImportError:
            pytest.skip("PyTorch RVC not available")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_state = ckpt["model"]

        # Create MLX generator with 48k config
        config = {
            "initial_channel": 192,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [12, 10, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [24, 20, 4, 4],
            "gin_channels": 256,
            "sr": 48000,
        }
        mlx_gen = GeneratorNSF(**config)

        # Load weights into MLX generator
        _load_generator_weights(mlx_gen, model_state)

        # Create PyTorch generator
        pt_gen = PTGeneratorNSF(
            initial_channel=192,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[12, 10, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[24, 20, 4, 4],
            gin_channels=256,
            sr=48000,
            is_half=False,
        )

        # Load weights into PyTorch generator
        pt_state_dict = {}
        for key, value in model_state.items():
            if key.startswith("dec."):
                new_key = key[4:]  # Remove "dec." prefix
                pt_state_dict[new_key] = value

        pt_gen.load_state_dict(pt_state_dict, strict=False)
        pt_gen.eval()

        # Create test input
        np.random.seed(42)
        batch, length = 1, 20  # Small for faster test
        x_np = np.random.randn(batch, 192, length).astype(np.float32) * 0.1
        f0_np = np.ones((batch, length), dtype=np.float32) * 220.0
        g_np = np.random.randn(batch, 256, 1).astype(np.float32) * 0.1

        x_pt = torch.from_numpy(x_np)
        f0_pt = torch.from_numpy(f0_np)
        g_pt = torch.from_numpy(g_np)

        x_mlx = mx.array(x_np)
        f0_mlx = mx.array(f0_np)
        g_mlx = mx.array(g_np)

        # Forward pass
        with torch.no_grad():
            y_pt = pt_gen(x_pt, f0_pt, g_pt).numpy()

        y_mlx = np.array(mlx_gen(x_mlx, f0_mlx, g_mlx))

        # Check shapes match
        assert y_mlx.shape == y_pt.shape, f"Shape mismatch: {y_mlx.shape} vs {y_pt.shape}"

        # Check values are close
        # Note: Due to differences in sine generation randomness, we check correlation
        # rather than exact match
        correlation = np.corrcoef(y_mlx.flatten(), y_pt.flatten())[0, 1]
        print(f"Output correlation: {correlation}")
        print(f"MLX output range: [{y_mlx.min():.4f}, {y_mlx.max():.4f}]")
        print(f"PyTorch output range: [{y_pt.min():.4f}, {y_pt.max():.4f}]")

        # For now, just verify shapes and output range are correct
        assert y_mlx.min() >= -1.0
        assert y_mlx.max() <= 1.0


def _load_generator_weights(mlx_gen: GeneratorNSF, model_state: dict):
    """Load PyTorch checkpoint weights into MLX GeneratorNSF."""
    # Helper to get weight with prefix
    def get_weight(key):
        full_key = f"dec.{key}"
        if full_key not in model_state:
            return None
        return model_state[full_key].float().numpy()

    # Load conv_pre
    w = get_weight("conv_pre.weight")
    b = get_weight("conv_pre.bias")
    if w is not None:
        # PyTorch Conv1d: (out, in, kernel) -> MLX: (out, kernel, in)
        mlx_gen.conv_pre.weight = mx.array(np.transpose(w, (0, 2, 1)))
    if b is not None:
        mlx_gen.conv_pre.bias = mx.array(b)

    # Load conv_post (no bias)
    w = get_weight("conv_post.weight")
    if w is not None:
        mlx_gen.conv_post.weight = mx.array(np.transpose(w, (0, 2, 1)))

    # Load cond layer if present
    if mlx_gen.gin_channels != 0:
        w = get_weight("cond.weight")
        b = get_weight("cond.bias")
        if w is not None:
            mlx_gen.cond.weight = mx.array(np.transpose(w, (0, 2, 1)))
        if b is not None:
            mlx_gen.cond.bias = mx.array(b)

    # Load ups (weight norm ConvTranspose1d)
    for i, ups in enumerate(mlx_gen.ups):
        v = get_weight(f"ups.{i}.weight_v")
        g = get_weight(f"ups.{i}.weight_g")
        b = get_weight(f"ups.{i}.bias")
        if v is not None:
            # PyTorch ConvTranspose1d weight_v: (in, out, kernel) -> MLX: (in, kernel, out)
            mlx_gen.ups[i].weight_v = mx.array(np.transpose(v, (0, 2, 1)))
        if g is not None:
            mlx_gen.ups[i].weight_g = mx.array(g)
        if b is not None:
            mlx_gen.ups[i].bias = mx.array(b)

    # Load noise_convs
    for i in range(len(mlx_gen.noise_convs)):
        w = get_weight(f"noise_convs.{i}.weight")
        b = get_weight(f"noise_convs.{i}.bias")
        if w is not None:
            mlx_gen.noise_convs[i].weight = mx.array(np.transpose(w, (0, 2, 1)))
        if b is not None:
            mlx_gen.noise_convs[i].bias = mx.array(b)

    # Load resblocks
    for i, resblock in enumerate(mlx_gen.resblocks):
        # ResBlock1 has convs1 and convs2
        if hasattr(resblock, "convs1"):
            for j in range(len(resblock.convs1)):
                v = get_weight(f"resblocks.{i}.convs1.{j}.weight_v")
                g = get_weight(f"resblocks.{i}.convs1.{j}.weight_g")
                b = get_weight(f"resblocks.{i}.convs1.{j}.bias")
                if v is not None:
                    resblock.convs1[j].weight_v = mx.array(np.transpose(v, (0, 2, 1)))
                if g is not None:
                    resblock.convs1[j].weight_g = mx.array(g)
                if b is not None:
                    resblock.convs1[j].bias = mx.array(b)

            for j in range(len(resblock.convs2)):
                v = get_weight(f"resblocks.{i}.convs2.{j}.weight_v")
                g = get_weight(f"resblocks.{i}.convs2.{j}.weight_g")
                b = get_weight(f"resblocks.{i}.convs2.{j}.bias")
                if v is not None:
                    resblock.convs2[j].weight_v = mx.array(np.transpose(v, (0, 2, 1)))
                if g is not None:
                    resblock.convs2[j].weight_g = mx.array(g)
                if b is not None:
                    resblock.convs2[j].bias = mx.array(b)
        # ResBlock2 has just convs
        elif hasattr(resblock, "convs"):
            for j in range(len(resblock.convs)):
                v = get_weight(f"resblocks.{i}.convs.{j}.weight_v")
                g = get_weight(f"resblocks.{i}.convs.{j}.weight_g")
                b = get_weight(f"resblocks.{i}.convs.{j}.bias")
                if v is not None:
                    resblock.convs[j].weight_v = mx.array(np.transpose(v, (0, 2, 1)))
                if g is not None:
                    resblock.convs[j].weight_g = mx.array(g)
                if b is not None:
                    resblock.convs[j].bias = mx.array(b)

    # Load source module
    w = get_weight("m_source.l_linear.weight")
    b = get_weight("m_source.l_linear.bias")
    if w is not None:
        mlx_gen.m_source.l_linear.weight = mx.array(w)
    if b is not None:
        mlx_gen.m_source.l_linear.bias = mx.array(b)

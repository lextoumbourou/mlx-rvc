"""Tests for weight normalization layers against PyTorch reference."""

import numpy as np
import pytest
import mlx.core as mx

from rvc_mlx.models.commons import WeightNormConv1d, WeightNormConvTranspose1d

# Only import torch for comparison tests
torch = pytest.importorskip("torch")
from torch.nn.utils import weight_norm
from torch import nn


class TestWeightNormConv1d:
    """Test WeightNormConv1d against PyTorch weight_norm(Conv1d)."""

    def test_weight_computation(self):
        """Test that weight normalization formula matches PyTorch."""
        in_ch, out_ch, kernel = 4, 8, 3

        # Create PyTorch reference
        pt_conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel, padding=1))

        # Create MLX layer
        mlx_conv = WeightNormConv1d(in_ch, out_ch, kernel, padding=1)

        # Copy weights from PyTorch to MLX
        # PyTorch weight_v: (out, in, kernel) -> MLX: (out, kernel, in)
        pt_v = pt_conv.weight_v.detach().numpy()
        pt_g = pt_conv.weight_g.detach().numpy()
        pt_bias = pt_conv.bias.detach().numpy()

        mlx_conv.weight_v = mx.array(np.transpose(pt_v, (0, 2, 1)))
        mlx_conv.weight_g = mx.array(pt_g)
        mlx_conv.bias = mx.array(pt_bias)

        # Compute effective weight in both
        with torch.no_grad():
            # PyTorch computes: g * v / ||v||
            pt_weight = pt_conv.weight.detach().numpy()

        mlx_weight = np.array(mlx_conv._compute_weight())
        # MLX weight is (out, kernel, in), convert to (out, in, kernel) for comparison
        mlx_weight = np.transpose(mlx_weight, (0, 2, 1))

        np.testing.assert_allclose(mlx_weight, pt_weight, rtol=1e-5, atol=1e-6)

    def test_forward_pass(self):
        """Test that forward pass matches PyTorch."""
        batch, in_ch, out_ch, kernel, length = 2, 4, 8, 3, 16

        # Create PyTorch reference
        pt_conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel, padding=1))

        # Create MLX layer
        mlx_conv = WeightNormConv1d(in_ch, out_ch, kernel, padding=1)

        # Copy weights
        pt_v = pt_conv.weight_v.detach().numpy()
        pt_g = pt_conv.weight_g.detach().numpy()
        pt_bias = pt_conv.bias.detach().numpy()

        mlx_conv.weight_v = mx.array(np.transpose(pt_v, (0, 2, 1)))
        mlx_conv.weight_g = mx.array(pt_g)
        mlx_conv.bias = mx.array(pt_bias)

        # Create input
        x_np = np.random.randn(batch, in_ch, length).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        # Forward pass
        with torch.no_grad():
            y_pt = pt_conv(x_pt).numpy()

        y_mlx = np.array(mlx_conv(x_mlx))

        np.testing.assert_allclose(y_mlx, y_pt, rtol=1e-4, atol=1e-5)

    def test_forward_with_dilation(self):
        """Test forward pass with dilation."""
        batch, in_ch, out_ch, kernel, length = 2, 4, 8, 3, 32
        dilation = 2
        padding = (kernel * dilation - dilation) // 2  # 'same' padding

        pt_conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation))
        mlx_conv = WeightNormConv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)

        # Copy weights
        pt_v = pt_conv.weight_v.detach().numpy()
        pt_g = pt_conv.weight_g.detach().numpy()
        pt_bias = pt_conv.bias.detach().numpy()

        mlx_conv.weight_v = mx.array(np.transpose(pt_v, (0, 2, 1)))
        mlx_conv.weight_g = mx.array(pt_g)
        mlx_conv.bias = mx.array(pt_bias)

        x_np = np.random.randn(batch, in_ch, length).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        with torch.no_grad():
            y_pt = pt_conv(x_pt).numpy()

        y_mlx = np.array(mlx_conv(x_mlx))

        np.testing.assert_allclose(y_mlx, y_pt, rtol=1e-4, atol=1e-5)


class TestWeightNormConvTranspose1d:
    """Test WeightNormConvTranspose1d against PyTorch weight_norm(ConvTranspose1d)."""

    def test_forward_pass(self):
        """Test that forward pass matches PyTorch."""
        batch, in_ch, out_ch, kernel, stride, length = 2, 8, 4, 4, 2, 16

        # Create PyTorch reference
        pt_conv = weight_norm(nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=1))

        # Create MLX layer
        mlx_conv = WeightNormConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=1)

        # Copy weights
        # PyTorch ConvTranspose1d weight_v: (in, out, kernel)
        pt_v = pt_conv.weight_v.detach().numpy()
        pt_g = pt_conv.weight_g.detach().numpy()
        pt_bias = pt_conv.bias.detach().numpy()

        # MLX stores as (in, kernel, out)
        mlx_conv.weight_v = mx.array(np.transpose(pt_v, (0, 2, 1)))
        mlx_conv.weight_g = mx.array(pt_g)
        mlx_conv.bias = mx.array(pt_bias)

        # Create input
        x_np = np.random.randn(batch, in_ch, length).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        # Forward pass
        with torch.no_grad():
            y_pt = pt_conv(x_pt).numpy()

        y_mlx = np.array(mlx_conv(x_mlx))

        # Check shapes match
        assert y_mlx.shape == y_pt.shape, f"Shape mismatch: {y_mlx.shape} vs {y_pt.shape}"

        np.testing.assert_allclose(y_mlx, y_pt, rtol=1e-4, atol=1e-5)

    def test_upsample_factor(self):
        """Test upsampling with stride > 1."""
        batch, in_ch, out_ch, kernel, stride, length = 1, 4, 2, 8, 4, 8

        pt_conv = weight_norm(nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=2))
        mlx_conv = WeightNormConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=2)

        pt_v = pt_conv.weight_v.detach().numpy()
        pt_g = pt_conv.weight_g.detach().numpy()
        pt_bias = pt_conv.bias.detach().numpy()

        mlx_conv.weight_v = mx.array(np.transpose(pt_v, (0, 2, 1)))
        mlx_conv.weight_g = mx.array(pt_g)
        mlx_conv.bias = mx.array(pt_bias)

        x_np = np.random.randn(batch, in_ch, length).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        with torch.no_grad():
            y_pt = pt_conv(x_pt).numpy()

        y_mlx = np.array(mlx_conv(x_mlx))

        # Should upsample by factor of stride
        assert y_mlx.shape[-1] == y_pt.shape[-1]
        np.testing.assert_allclose(y_mlx, y_pt, rtol=1e-4, atol=1e-5)


class TestLoadFromCheckpoint:
    """Test loading weights from actual RVC checkpoint."""

    def test_load_resblock_weights(self):
        """Test loading a ResBlock conv from the checkpoint."""
        import os

        ckpt_path = "vendor/weights/f0G48k.pth"
        if not os.path.exists(ckpt_path):
            pytest.skip("Checkpoint not found")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ckpt["model"]

        # Get first resblock conv weights
        key_v = "dec.resblocks.0.convs1.0.weight_v"
        key_g = "dec.resblocks.0.convs1.0.weight_g"
        key_b = "dec.resblocks.0.convs1.0.bias"

        # Convert to float32 (checkpoint may store as float16)
        pt_v = model[key_v].float().numpy()  # (256, 256, 3)
        pt_g = model[key_g].float().numpy()  # (256, 1, 1)
        pt_bias = model[key_b].float().numpy()  # (256,)

        # Create MLX conv with same config
        mlx_conv = WeightNormConv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1,  # dilation=1, so padding=(3-1)//2=1
            dilation=1,
        )

        # Load weights (transpose v from (out, in, kernel) to (out, kernel, in))
        mlx_conv.weight_v = mx.array(np.transpose(pt_v, (0, 2, 1)).astype(np.float32))
        mlx_conv.weight_g = mx.array(pt_g.astype(np.float32))
        mlx_conv.bias = mx.array(pt_bias.astype(np.float32))

        # Create PyTorch conv with same weights
        pt_conv = nn.Conv1d(256, 256, 3, padding=1)
        pt_conv = weight_norm(pt_conv)
        with torch.no_grad():
            pt_conv.weight_v.copy_(torch.from_numpy(pt_v))
            pt_conv.weight_g.copy_(torch.from_numpy(pt_g))
            pt_conv.bias.copy_(torch.from_numpy(pt_bias))

        # Test forward pass
        x_np = np.random.randn(1, 256, 100).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        with torch.no_grad():
            y_pt = pt_conv(x_pt).numpy()

        y_mlx = np.array(mlx_conv(x_mlx))

        np.testing.assert_allclose(y_mlx, y_pt, rtol=1e-4, atol=1e-5)

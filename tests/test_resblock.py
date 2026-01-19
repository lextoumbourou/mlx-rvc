"""Tests for ResBlock implementations."""

import numpy as np
import pytest
import mlx.core as mx

from rvc_mlx.models.resblock import ResBlock1, ResBlock2

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn.utils import weight_norm


class TestResBlock1:
    """Test ResBlock1 against PyTorch reference."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        batch, channels, length = 2, 256, 100

        block = ResBlock1(channels, kernel_size=3, dilation=(1, 3, 5))

        x = mx.random.normal(shape=(batch, channels, length))
        y = block(x)

        assert y.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connections preserve input when weights are zero."""
        batch, channels, length = 1, 64, 50

        block = ResBlock1(channels)

        # With zero-initialized weights, output should equal input
        # because conv(x) = 0, so x + conv(conv(x)) = x + 0 = x
        x = mx.random.normal(shape=(batch, channels, length))
        y = block(x)

        # With zero weights, residual connection means output = input
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=1e-5, atol=1e-6)

    def test_load_from_checkpoint(self):
        """Test loading ResBlock1 weights from RVC checkpoint."""
        import os

        ckpt_path = "vendor/weights/f0G48k.pth"
        if not os.path.exists(ckpt_path):
            pytest.skip("Checkpoint not found")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ckpt["model"]

        # Load first ResBlock (channels=256, kernel=3, dilation=(1,3,5))
        mlx_block = ResBlock1(256, kernel_size=3, dilation=(1, 3, 5))

        # Load weights for convs1 and convs2
        for i in range(3):
            # convs1
            v = model[f"dec.resblocks.0.convs1.{i}.weight_v"].float().numpy()
            g = model[f"dec.resblocks.0.convs1.{i}.weight_g"].float().numpy()
            b = model[f"dec.resblocks.0.convs1.{i}.bias"].float().numpy()

            mlx_block.convs1[i].weight_v = mx.array(np.transpose(v, (0, 2, 1)))
            mlx_block.convs1[i].weight_g = mx.array(g)
            mlx_block.convs1[i].bias = mx.array(b)

            # convs2
            v = model[f"dec.resblocks.0.convs2.{i}.weight_v"].float().numpy()
            g = model[f"dec.resblocks.0.convs2.{i}.weight_g"].float().numpy()
            b = model[f"dec.resblocks.0.convs2.{i}.bias"].float().numpy()

            mlx_block.convs2[i].weight_v = mx.array(np.transpose(v, (0, 2, 1)))
            mlx_block.convs2[i].weight_g = mx.array(g)
            mlx_block.convs2[i].bias = mx.array(b)

        # Create PyTorch ResBlock
        import sys
        sys.path.insert(0, "vendor/Retrieval-based-Voice-Conversion-WebUI")
        from infer.lib.infer_pack.modules import ResBlock1 as PTResBlock1

        pt_block = PTResBlock1(256, 3, (1, 3, 5))

        # Load same weights into PyTorch
        for i in range(3):
            v = model[f"dec.resblocks.0.convs1.{i}.weight_v"]
            g = model[f"dec.resblocks.0.convs1.{i}.weight_g"]
            b = model[f"dec.resblocks.0.convs1.{i}.bias"]

            with torch.no_grad():
                pt_block.convs1[i].weight_v.copy_(v)
                pt_block.convs1[i].weight_g.copy_(g)
                pt_block.convs1[i].bias.copy_(b)

            v = model[f"dec.resblocks.0.convs2.{i}.weight_v"]
            g = model[f"dec.resblocks.0.convs2.{i}.weight_g"]
            b = model[f"dec.resblocks.0.convs2.{i}.bias"]

            with torch.no_grad():
                pt_block.convs2[i].weight_v.copy_(v)
                pt_block.convs2[i].weight_g.copy_(g)
                pt_block.convs2[i].bias.copy_(b)

        # Test forward pass
        np.random.seed(42)
        x_np = np.random.randn(1, 256, 100).astype(np.float32)

        x_pt = torch.from_numpy(x_np)
        x_mlx = mx.array(x_np)

        with torch.no_grad():
            y_pt = pt_block(x_pt).numpy()

        y_mlx = np.array(mlx_block(x_mlx))

        # Check shapes match
        assert y_mlx.shape == y_pt.shape

        # Check values match
        np.testing.assert_allclose(y_mlx, y_pt, rtol=1e-4, atol=1e-5)


class TestResBlock2:
    """Test ResBlock2."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        batch, channels, length = 2, 128, 100

        block = ResBlock2(channels, kernel_size=3, dilation=(1, 3))

        x = mx.random.normal(shape=(batch, channels, length))
        y = block(x)

        assert y.shape == x.shape

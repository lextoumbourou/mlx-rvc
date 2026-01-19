"""Residual blocks for the HiFi-GAN vocoder."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from .commons import WeightNormConv1d, get_padding, leaky_relu, LRELU_SLOPE


class ResBlock1(nn.Module):
    """
    Residual block with dilated convolutions (HiFi-GAN style).

    Each block contains pairs of dilated conv + regular conv,
    with residual connections.

    Args:
        channels: Number of input/output channels
        kernel_size: Convolution kernel size
        dilation: Tuple of dilation rates for the dilated convs
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.lrelu_slope = LRELU_SLOPE

        # First set of convolutions (dilated)
        self.convs1 = [
            WeightNormConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=dilation[0],
                padding=get_padding(kernel_size, dilation[0]),
            ),
            WeightNormConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=dilation[1],
                padding=get_padding(kernel_size, dilation[1]),
            ),
            WeightNormConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=dilation[2],
                padding=get_padding(kernel_size, dilation[2]),
            ),
        ]

        # Second set of convolutions (non-dilated)
        self.convs2 = [
            WeightNormConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            ),
            WeightNormConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            ),
            WeightNormConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            ),
        ]

    def __call__(self, x: mx.array, x_mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, length)
            x_mask: Optional mask of shape (batch, 1, length)

        Returns:
            Output tensor of shape (batch, channels, length)
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = leaky_relu(xt, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x

        if x_mask is not None:
            x = x * x_mask

        return x


class ResBlock2(nn.Module):
    """
    Simplified residual block (HiFi-GAN style, type 2).

    Similar to ResBlock1 but with only one conv per stage.

    Args:
        channels: Number of input/output channels
        kernel_size: Convolution kernel size
        dilation: Tuple of dilation rates
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int] = (1, 3),
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.lrelu_slope = LRELU_SLOPE

        self.convs = [
            WeightNormConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=dilation[0],
                padding=get_padding(kernel_size, dilation[0]),
            ),
            WeightNormConv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=dilation[1],
                padding=get_padding(kernel_size, dilation[1]),
            ),
        ]

    def __call__(self, x: mx.array, x_mask: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, length)
            x_mask: Optional mask of shape (batch, 1, length)

        Returns:
            Output tensor of shape (batch, channels, length)
        """
        for c in self.convs:
            xt = leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x

        if x_mask is not None:
            x = x * x_mask

        return x

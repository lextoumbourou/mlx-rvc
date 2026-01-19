"""Common utilities and layers for RVC models."""

import mlx.core as mx
import mlx.nn as nn
import math


LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' output size with dilation."""
    return (kernel_size * dilation - dilation) // 2


def sequence_mask(length: mx.array, max_length: int | None = None) -> mx.array:
    """
    Create a sequence mask from lengths.

    Args:
        length: Tensor of lengths, shape (batch,)
        max_length: Maximum sequence length. If None, uses max(length).

    Returns:
        Boolean mask of shape (batch, max_length)
    """
    if max_length is None:
        max_length = int(length.max().item())

    # Create range tensor and compare with lengths
    positions = mx.arange(max_length)[None, :]  # (1, max_length)
    lengths = length[:, None]  # (batch, 1)

    return positions < lengths


def fused_add_tanh_sigmoid_multiply(
    input_a: mx.array,
    input_b: mx.array,
    n_channels: int,
) -> mx.array:
    """
    Fused gated activation: tanh(a) * sigmoid(b)

    Used in WaveNet-style architectures.
    """
    n_channels_int = n_channels
    in_act = input_a + input_b
    t_act = mx.tanh(in_act[:, :n_channels_int, :])
    s_act = mx.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


class WeightNormConv1d(nn.Module):
    """
    Conv1d with weight normalization baked in.

    Weight normalization decomposes the weight tensor into magnitude (g) and
    direction (v): w = g * (v / ||v||)

    For inference, we can pre-compute the normalized weight and use a standard conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight normalization parameters
        # In PyTorch weight_norm, weight_v has shape (out, in, kernel) for Conv1d
        # weight_g has shape (out, 1, 1)
        # MLX conv1d expects (out, kernel, in) so we store in that format
        self.weight_v = mx.zeros((out_channels, kernel_size, in_channels // groups))
        self.weight_g = mx.ones((out_channels, 1, 1))

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def _compute_weight(self) -> mx.array:
        """Compute the effective weight: w = g * (v / ||v||)"""
        # Normalize v over (kernel, in_channels) dimensions
        # v shape: (out, kernel, in)
        v = self.weight_v
        norm = mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True) + 1e-12)
        weight = self.weight_g * (v / norm)
        return weight

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, length)

        Returns:
            Output tensor of shape (batch, out_channels, new_length)
        """
        weight = self._compute_weight()

        # MLX conv1d: input (N, L, C_in), weight (C_out, K, C_in)
        # But we have input as (N, C_in, L), so we need to transpose
        x = mx.transpose(x, (0, 2, 1))  # (N, L, C_in)

        y = mx.conv1d(
            x,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Transpose back to (N, C_out, L)
        y = mx.transpose(y, (0, 2, 1))

        if self.bias is not None:
            y = y + self.bias[None, :, None]

        return y


class WeightNormConvTranspose1d(nn.Module):
    """
    ConvTranspose1d with weight normalization baked in.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Weight normalization parameters
        # PyTorch ConvTranspose1d weight shape: (in, out, kernel)
        # For weight_norm: weight_v same shape, weight_g is (in, 1, 1)
        # MLX conv_transpose1d expects (out, kernel, in) - we'll handle in forward
        self.weight_v = mx.zeros((in_channels, kernel_size, out_channels // groups))
        self.weight_g = mx.ones((in_channels, 1, 1))

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def _compute_weight(self) -> mx.array:
        """Compute the effective weight: w = g * (v / ||v||)"""
        v = self.weight_v
        norm = mx.sqrt(mx.sum(v * v, axis=(1, 2), keepdims=True) + 1e-12)
        weight = self.weight_g * (v / norm)
        return weight

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, length)

        Returns:
            Output tensor of shape (batch, out_channels, new_length)
        """
        weight = self._compute_weight()

        # x: (N, C_in, L) -> (N, L, C_in) for MLX
        x = mx.transpose(x, (0, 2, 1))

        # weight is (in, kernel, out), need (out, kernel, in) for conv_transpose
        weight = mx.transpose(weight, (2, 1, 0))

        y = mx.conv_transpose1d(
            x,
            weight,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )

        # Transpose back: (N, L, C_out) -> (N, C_out, L)
        y = mx.transpose(y, (0, 2, 1))

        if self.bias is not None:
            y = y + self.bias[None, :, None]

        return y


class Conv1d(nn.Module):
    """
    Standard Conv1d matching PyTorch interface (channels first).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # MLX conv1d weight shape: (out, kernel, in)
        self.weight = mx.zeros((out_channels, kernel_size, in_channels // groups))

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, length)

        Returns:
            Output tensor of shape (batch, out_channels, new_length)
        """
        # x: (N, C_in, L) -> (N, L, C_in) for MLX
        x = mx.transpose(x, (0, 2, 1))

        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Transpose back: (N, L, C_out) -> (N, C_out, L)
        y = mx.transpose(y, (0, 2, 1))

        if self.bias is not None:
            y = y + self.bias[None, :, None]

        return y


def leaky_relu(x: mx.array, negative_slope: float = LRELU_SLOPE) -> mx.array:
    """Leaky ReLU activation."""
    return mx.where(x > 0, x, x * negative_slope)

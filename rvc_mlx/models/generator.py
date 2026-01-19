"""GeneratorNSF - HiFi-GAN style vocoder with NSF (Neural Source Filter)."""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from .commons import (
    WeightNormConv1d,
    WeightNormConvTranspose1d,
    Conv1d,
    leaky_relu,
    LRELU_SLOPE,
)
from .resblock import ResBlock1, ResBlock2
from .nsf import SourceModuleHnNSF


class GeneratorNSF(nn.Module):
    """
    HiFi-GAN style generator with Neural Source Filter for pitch-aware synthesis.

    This vocoder takes mel-spectrogram-like features and F0 (fundamental frequency)
    to generate waveforms. The NSF component generates pitch-synchronous excitation
    that is combined with the upsampled features.

    Args:
        initial_channel: Number of input channels (e.g., 192)
        resblock: Type of residual block ("1" for ResBlock1, "2" for ResBlock2)
        resblock_kernel_sizes: Kernel sizes for resblocks (e.g., [3, 7, 11])
        resblock_dilation_sizes: Dilation patterns for resblocks
        upsample_rates: Upsampling rates for each stage (e.g., [12, 10, 2, 2])
        upsample_initial_channel: Initial channel size for upsampling (e.g., 512)
        upsample_kernel_sizes: Kernel sizes for upsampling convs
        gin_channels: Speaker embedding channels (0 to disable)
        sr: Sample rate in Hz
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        sr: int,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.sr = sr
        self.lrelu_slope = LRELU_SLOPE

        # Total upsampling factor
        self.upp = math.prod(upsample_rates)

        # Source module for harmonic generation
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        # Noise convolutions to process harmonic source at each scale
        self.noise_convs = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        # Initial convolution
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, stride=1, padding=3
        )

        # Select resblock type
        ResBlockClass = ResBlock1 if resblock == "1" else ResBlock2

        # Upsampling layers
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                WeightNormConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

        # Residual blocks
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlockClass(ch, k, tuple(d)))

        # Final output channel count
        final_ch = upsample_initial_channel // (2 ** len(self.ups))

        # Final convolution (no bias)
        self.conv_post = Conv1d(final_ch, 1, 7, stride=1, padding=3, bias=False)

        # Speaker conditioning
        self.gin_channels = gin_channels
        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

    def __call__(
        self,
        x: mx.array,
        f0: mx.array,
        g: Optional[mx.array] = None,
        n_res: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Generate audio from features and F0.

        Args:
            x: Input features, shape (batch, initial_channel, length)
            f0: F0 values in Hz, shape (batch, f0_length)
            g: Optional speaker embedding, shape (batch, gin_channels, 1)
            n_res: Optional length adjustment

        Returns:
            Generated audio, shape (batch, 1, length * upp)
        """
        # Generate harmonic source from F0
        har_source, _, _ = self.m_source(f0, self.upp)  # (batch, f0_length * upp, 1)
        har_source = mx.transpose(har_source, (0, 2, 1))  # (batch, 1, length * upp)

        # Handle length adjustment if specified
        if n_res is not None:
            n = int(n_res.item())
            if n * self.upp != har_source.shape[-1]:
                # Interpolate har_source to target length
                har_source = self._interpolate(har_source, n * self.upp)
            if n != x.shape[-1]:
                # Interpolate x to target length
                x = self._interpolate(x, n)

        # Initial convolution
        x = self.conv_pre(x)  # (batch, upsample_initial_channel, length)

        # Add speaker conditioning
        if g is not None:
            x = x + self.cond(g)

        # Upsampling stages
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = leaky_relu(x, self.lrelu_slope)
            x = ups(x)

            # Add source excitation
            x_source = noise_convs(har_source)
            x = x + x_source

            # Apply resblocks and average
            xs = None
            for j in range(self.num_kernels):
                resblock_idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[resblock_idx](x)
                else:
                    xs = xs + self.resblocks[resblock_idx](x)

            x = xs / self.num_kernels

        # Final output
        x = leaky_relu(x, self.lrelu_slope)
        x = self.conv_post(x)
        x = mx.tanh(x)

        return x

    def _interpolate(self, x: mx.array, target_length: int) -> mx.array:
        """
        Linear interpolation for 1D signals.

        Args:
            x: Input tensor, shape (batch, channels, length)
            target_length: Target length

        Returns:
            Interpolated tensor, shape (batch, channels, target_length)
        """
        batch, channels, length = x.shape
        if length == target_length:
            return x

        # Create interpolation indices
        scale = (length - 1) / (target_length - 1) if target_length > 1 else 0
        indices = mx.arange(target_length, dtype=mx.float32) * scale

        # Get integer indices and weights
        idx_low = mx.clip(mx.floor(indices).astype(mx.int32), 0, length - 1)
        idx_high = mx.clip(idx_low + 1, 0, length - 1)
        weight_high = indices - idx_low.astype(mx.float32)
        weight_low = 1.0 - weight_high

        # Gather and interpolate
        # x: (batch, channels, length)
        # Reshape for gathering: (batch * channels, length)
        x_flat = mx.reshape(x, (batch * channels, length))

        # Gather values
        low_vals = mx.take(x_flat, idx_low, axis=1)  # (batch * channels, target_length)
        high_vals = mx.take(x_flat, idx_high, axis=1)

        # Interpolate
        result = low_vals * weight_low[None, :] + high_vals * weight_high[None, :]

        # Reshape back
        return mx.reshape(result, (batch, channels, target_length))

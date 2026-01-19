"""Neural Source Filter components for pitch-aware audio synthesis."""

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class SineGen(nn.Module):
    """
    Sine wave generator with harmonic overtones.

    Generates pitch-synchronous sine waves from F0 (fundamental frequency),
    including harmonics. Used as the excitation source in NSF vocoders.

    Args:
        sample_rate: Audio sample rate in Hz
        harmonic_num: Number of harmonic overtones (0 = fundamental only)
        sine_amp: Amplitude of sine waveforms
        noise_std: Standard deviation of additive noise
        voiced_threshold: F0 threshold for voiced/unvoiced classification
    """

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.dim = harmonic_num + 1  # fundamental + harmonics

    def _f02uv(self, f0: mx.array) -> mx.array:
        """
        Generate voiced/unvoiced mask from F0.

        Args:
            f0: F0 values, shape (batch, length, 1)

        Returns:
            UV mask: 1 for voiced, 0 for unvoiced
        """
        uv = mx.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0: mx.array, upp: int) -> mx.array:
        """
        Generate sine waves from F0.

        This is the core algorithm that generates phase-continuous sine waves
        at the fundamental frequency and its harmonics.

        Args:
            f0: F0 values, shape (batch, length, 1)
            upp: Upsampling factor (samples per F0 frame)

        Returns:
            Sine waves, shape (batch, length * upp, dim)
        """
        batch_size = f0.shape[0]
        f0_length = f0.shape[1]

        # Create upsampling indices: [1, 2, ..., upp]
        a = mx.arange(1, upp + 1, dtype=f0.dtype)  # (upp,)

        # Compute phase increments: f0 / sr gives cycles per sample
        # f0 shape: (batch, length, 1)
        rad = f0 / self.sample_rate  # (batch, length, 1)
        # Expand for upsampling: (batch, length, 1, upp)
        rad = rad[:, :, :, None] * a[None, None, None, :]

        # Handle phase continuity between frames
        # Take the last phase value of each frame
        rad2 = mx.remainder(rad[:, :-1, :, -1:] + 0.5, 1.0) - 0.5  # (batch, length-1, 1, 1)

        # Cumulative sum for phase continuity
        rad_acc = mx.remainder(mx.cumsum(rad2, axis=1), 1.0)  # (batch, length-1, 1, 1)

        # Pad with zeros at the beginning for the first frame
        rad_acc = mx.pad(rad_acc, [(0, 0), (1, 0), (0, 0), (0, 0)])  # (batch, length, 1, 1)

        # Add accumulated phase to current frame phases
        rad = rad + rad_acc  # (batch, length, 1, upp)

        # Reshape to (batch, length * upp, 1)
        rad = mx.transpose(rad, (0, 1, 3, 2))  # (batch, length, upp, 1)
        rad = mx.reshape(rad, (batch_size, f0_length * upp, 1))

        # Create harmonic multipliers: [1, 2, ..., dim]
        b = mx.arange(1, self.dim + 1, dtype=f0.dtype)  # (dim,)
        b = b[None, None, :]  # (1, 1, dim)

        # Multiply phase by harmonic number to get all harmonics
        rad = rad * b  # (batch, length*upp, dim)

        # Add random initial phase for harmonics (not fundamental)
        rand_ini = mx.random.uniform(shape=(1, 1, self.dim))
        # Set fundamental phase to 0 using masking
        if self.dim > 1:
            mask = mx.concatenate([mx.zeros((1, 1, 1)), mx.ones((1, 1, self.dim - 1))], axis=2)
        else:
            mask = mx.zeros((1, 1, 1))
        rand_ini = rand_ini * mask
        rad = rad + rand_ini

        # Generate sine waves
        sines = mx.sin(2 * math.pi * rad)

        return sines

    def __call__(self, f0: mx.array, upp: int) -> tuple[mx.array, mx.array, mx.array]:
        """
        Generate sine waves with noise from F0.

        Args:
            f0: F0 values, shape (batch, length) in Hz. 0 = unvoiced.
            upp: Upsampling factor

        Returns:
            Tuple of (sine_waves, uv, noise):
            - sine_waves: shape (batch, length * upp, dim)
            - uv: voiced/unvoiced mask, shape (batch, length * upp, 1)
            - noise: noise signal, shape (batch, length * upp, dim)
        """
        # Add dimension for harmonics
        f0 = f0[:, :, None]  # (batch, length, 1)

        # Generate sine waves
        sine_waves = self._f02sine(f0, upp) * self.sine_amp  # (batch, length*upp, dim)

        # Generate UV mask
        uv = self._f02uv(f0)  # (batch, length, 1)

        # Upsample UV mask using nearest neighbor
        # (batch, length, 1) -> (batch, length*upp, 1)
        uv = mx.repeat(uv, upp, axis=1)

        # Generate noise
        # Voiced: use noise_std, Unvoiced: use sine_amp/3
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(shape=sine_waves.shape)

        # Combine: voiced frames get sine + noise, unvoiced get just noise
        sine_waves = sine_waves * uv + noise

        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """
    Source module for harmonic-plus-noise NSF.

    Combines the SineGen output into a single excitation signal
    using a learned linear combination of harmonics.

    Args:
        sample_rate: Audio sample rate in Hz
        harmonic_num: Number of harmonics (0 = fundamental only)
        sine_amp: Amplitude of sine source
        noise_std: Standard deviation of noise
        voiced_threshold: Threshold for voiced/unvoiced detection
    """

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std

        # Sine generator
        self.l_sin_gen = SineGen(
            sample_rate,
            harmonic_num,
            sine_amp,
            noise_std,
            voiced_threshold,
        )

        # Linear layer to combine harmonics into single source
        # Input: (harmonic_num + 1) harmonics, Output: 1 channel
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def __call__(self, f0: mx.array, upp: int = 1) -> tuple[mx.array, None, None]:
        """
        Generate source excitation from F0.

        Args:
            f0: F0 values, shape (batch, length)
            upp: Upsampling factor

        Returns:
            Tuple of (source, None, None) for compatibility
            source shape: (batch, length * upp, 1)
        """
        # Generate sine waves with harmonics
        sine_waves, uv, _ = self.l_sin_gen(f0, upp)  # (batch, length*upp, dim)

        # Combine harmonics with learned weights
        sine_merge = self.l_tanh(self.l_linear(sine_waves))  # (batch, length*upp, 1)

        return sine_merge, None, None

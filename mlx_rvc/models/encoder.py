"""Text encoder for RVC synthesizer."""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from .attentions import Encoder
from .commons import sequence_mask, Conv1d, leaky_relu


class TextEncoder(nn.Module):
    """
    Text encoder that processes phoneme features and pitch.

    Takes ContentVec features and F0 (pitch), processes them through
    a transformer encoder, and outputs mean and log-std for the latent space.

    Args:
        in_channels: Input feature dimension (768 for ContentVec)
        out_channels: Output latent dimension (192)
        hidden_channels: Hidden dimension (192)
        filter_channels: FFN filter dimension (768)
        n_heads: Number of attention heads (2)
        n_layers: Number of transformer layers (6)
        kernel_size: FFN kernel size (3)
        p_dropout: Dropout probability (0 for inference)
        f0: Whether to use F0 conditioning (True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.f0 = f0

        # Phone embedding (Linear layer)
        self.emb_phone = nn.Linear(in_channels, hidden_channels)

        # Pitch embedding (only if f0=True)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)

        # Transformer encoder
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout=p_dropout,
        )

        # Output projection (to mean and log-std)
        self.proj = Conv1d(hidden_channels, out_channels * 2, 1)

    def __call__(
        self,
        phone: mx.array,
        pitch: Optional[mx.array],
        lengths: mx.array,
        skip_head: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Encode phoneme features and pitch.

        Args:
            phone: Phoneme features, shape (batch, length, in_channels)
            pitch: Pitch indices (0-255), shape (batch, length), or None
            lengths: Sequence lengths, shape (batch,)
            skip_head: Optional number of frames to skip from beginning

        Returns:
            Tuple of (mean, log_std, mask):
            - mean: shape (batch, out_channels, length)
            - log_std: shape (batch, out_channels, length)
            - mask: shape (batch, 1, length)
        """
        # Embed phone features
        x = self.emb_phone(phone)  # (batch, length, hidden)

        # Add pitch embedding if available
        if pitch is not None and self.f0:
            x = x + self.emb_pitch(pitch)

        # Scale by sqrt(hidden_channels)
        x = x * math.sqrt(self.hidden_channels)

        # Apply leaky ReLU
        x = leaky_relu(x, 0.1)

        # Transpose to (batch, hidden, length) for encoder
        x = mx.transpose(x, (0, 2, 1))

        # Create mask
        max_len = x.shape[2]
        x_mask = sequence_mask(lengths, max_len)  # (batch, length)
        x_mask = x_mask[:, None, :]  # (batch, 1, length)
        x_mask = x_mask.astype(x.dtype)

        # Apply transformer encoder
        x = self.encoder(x * x_mask, x_mask)

        # Handle skip_head if specified
        if skip_head is not None:
            head = int(skip_head.item())
            x = x[:, :, head:]
            x_mask = x_mask[:, :, head:]

        # Project to mean and log-std
        stats = self.proj(x) * x_mask
        m = stats[:, : self.out_channels, :]
        logs = stats[:, self.out_channels :, :]

        return m, logs, x_mask

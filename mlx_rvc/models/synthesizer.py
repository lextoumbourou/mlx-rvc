"""Synthesizer model for RVC voice conversion."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from .encoder import TextEncoder
from .generator import GeneratorNSF
from .flow import ResidualCouplingBlock


class SynthesizerTrnMs768NSFsid(nn.Module):
    """
    Main synthesizer model for RVC voice conversion with NSF vocoder.

    This model takes ContentVec features, F0, and speaker embedding to
    generate audio. It uses a flow-based architecture for voice conversion.

    Architecture:
    - enc_p: TextEncoder - processes phone features and pitch
    - flow: ResidualCouplingBlock - normalizing flow for voice conversion
    - dec: GeneratorNSF - HiFi-GAN vocoder with neural source filter
    - emb_g: Speaker embedding

    Args:
        spec_channels: Spectrogram channels (not used in inference)
        segment_size: Segment size for training (not used in inference)
        inter_channels: Intermediate channels (192)
        hidden_channels: Hidden channels (192)
        filter_channels: Filter channels for FFN (768)
        n_heads: Number of attention heads (2)
        n_layers: Number of encoder layers (6)
        kernel_size: Kernel size for FFN (3)
        p_dropout: Dropout probability (0 for inference)
        resblock: Resblock type ("1" or "2")
        resblock_kernel_sizes: Kernel sizes for resblocks ([3, 7, 11])
        resblock_dilation_sizes: Dilation patterns ([[1,3,5], [1,3,5], [1,3,5]])
        upsample_rates: Upsampling rates ([12, 10, 2, 2] for 48kHz)
        upsample_initial_channel: Initial channel count for upsampling (512)
        upsample_kernel_sizes: Kernel sizes for upsampling ([24, 20, 4, 4])
        spk_embed_dim: Number of speaker embeddings (109)
        gin_channels: Speaker conditioning channels (256)
        sr: Sample rate in Hz (48000)
    """

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.sr = sr

        # Text encoder (768 input channels for ContentVec)
        self.enc_p = TextEncoder(
            768,  # ContentVec feature dimension
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )

        # Flow-based normalizing flow
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,  # kernel_size
            1,  # dilation_rate
            3,  # n_layers
            gin_channels=gin_channels,
        )

        # NSF-based decoder/vocoder
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
        )

        # Speaker embedding
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def infer(
        self,
        phone: mx.array,
        phone_lengths: mx.array,
        pitch: mx.array,
        nsff0: mx.array,
        sid: mx.array,
        skip_head: Optional[mx.array] = None,
        return_length: Optional[mx.array] = None,
        return_length2: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array, tuple]:
        """
        Inference forward pass.

        Args:
            phone: ContentVec features, shape (batch, length, 768)
            phone_lengths: Sequence lengths, shape (batch,)
            pitch: Pitch indices (0-255), shape (batch, length)
            nsff0: F0 in Hz, shape (batch, length)
            sid: Speaker ID, shape (batch,)
            skip_head: Optional frames to skip from beginning
            return_length: Optional return length (for skip mode)
            return_length2: Optional length adjustment for decoder

        Returns:
            Tuple of (audio, mask, (z, z_p, m_p, logs_p)):
            - audio: Generated audio, shape (batch, 1, length * upp)
            - mask: Sequence mask
            - (z, z_p, m_p, logs_p): Intermediate representations
        """
        # Get speaker embedding
        g = self.emb_g(sid)[:, :, None]  # (batch, gin_channels, 1)

        if skip_head is not None and return_length is not None:
            # Handle skip mode for streaming
            head = int(skip_head.item())
            length = int(return_length.item())
            flow_head = mx.maximum(skip_head - 24, mx.array(0))

            dec_head = head - int(flow_head.item())

            # Encode
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, flow_head)

            # Sample from prior with reduced variance (0.66666 factor)
            z_p = (m_p + mx.exp(logs_p) * mx.random.normal(shape=m_p.shape) * 0.66666) * x_mask

            # Flow (reverse direction for inference)
            z = self.flow(z_p, x_mask, g=g, reverse=True)

            # Slice to requested range
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
            nsff0 = nsff0[:, head : head + length]
        else:
            # Standard inference
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

            # Sample from prior
            z_p = (m_p + mx.exp(logs_p) * mx.random.normal(shape=m_p.shape) * 0.66666) * x_mask

            # Flow
            z = self.flow(z_p, x_mask, g=g, reverse=True)

        # Decode to audio
        o = self.dec(z * x_mask, nsff0, g=g, n_res=return_length2)

        return o, x_mask, (z, z_p, m_p, logs_p)


class SynthesizerTrnMs768NSFsid_nono(nn.Module):
    """
    Synthesizer without F0 conditioning (for non-pitch models).

    Same as SynthesizerTrnMs768NSFsid but without pitch embedding.
    Used for models that don't use F0 information.
    """

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int = None,
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim

        # Text encoder without F0
        self.enc_p = TextEncoder(
            768,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            f0=False,  # No F0 conditioning
        )

        # Flow
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )

        # Decoder (without F0-based NSF, uses regular HiFi-GAN)
        # Note: For non-F0 models, we still use GeneratorNSF but F0 will be zeros
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr or 40000,
        )

        # Speaker embedding
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def infer(
        self,
        phone: mx.array,
        phone_lengths: mx.array,
        sid: mx.array,
        skip_head: Optional[mx.array] = None,
        return_length: Optional[mx.array] = None,
        return_length2: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array, tuple]:
        """
        Inference forward pass (without F0).

        Args:
            phone: ContentVec features, shape (batch, length, 768)
            phone_lengths: Sequence lengths, shape (batch,)
            sid: Speaker ID, shape (batch,)
            skip_head: Optional frames to skip from beginning
            return_length: Optional return length
            return_length2: Optional length adjustment for decoder

        Returns:
            Tuple of (audio, mask, (z, z_p, m_p, logs_p))
        """
        g = self.emb_g(sid)[:, :, None]

        if skip_head is not None and return_length is not None:
            head = int(skip_head.item())
            length = int(return_length.item())
            flow_head = mx.maximum(skip_head - 24, mx.array(0))
            dec_head = head - int(flow_head.item())

            m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths, flow_head)
            z_p = (m_p + mx.exp(logs_p) * mx.random.normal(shape=m_p.shape) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
        else:
            m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
            z_p = (m_p + mx.exp(logs_p) * mx.random.normal(shape=m_p.shape) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)

        # Decode with zero F0
        f0_length = z.shape[2]
        zero_f0 = mx.zeros((z.shape[0], f0_length))
        o = self.dec(z * x_mask, zero_f0, g=g, n_res=return_length2)

        return o, x_mask, (z, z_p, m_p, logs_p)

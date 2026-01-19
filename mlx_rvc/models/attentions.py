"""Attention modules for RVC encoder."""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class LayerNorm(nn.Module):
    """
    Layer normalization for channels-first data.

    Input is (batch, channels, length), normalized over channels dimension.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = mx.ones((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, channels, length) -> (batch, length, channels)
        x = mx.transpose(x, (0, 2, 1))

        # Normalize over last dimension (channels)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        x = x * self.gamma + self.beta

        # (batch, length, channels) -> (batch, channels, length)
        return mx.transpose(x, (0, 2, 1))


class FFN(nn.Module):
    """
    Feed-Forward Network for transformer encoder.

    Uses 1D convolutions with same padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str = None,
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.is_gelu = activation == "gelu"

        # Store weights for Conv1d
        # MLX conv1d weight shape: (out, kernel, in)
        self.conv_1_weight = mx.zeros((filter_channels, kernel_size, in_channels))
        self.conv_1_bias = mx.zeros((filter_channels,))
        self.conv_2_weight = mx.zeros((out_channels, kernel_size, filter_channels))
        self.conv_2_bias = mx.zeros((out_channels,))

    def _pad(self, x: mx.array, x_mask: mx.array) -> mx.array:
        """Apply padding based on causal flag."""
        x = x * x_mask
        if self.kernel_size == 1:
            return x

        if self.causal:
            pad_l = self.kernel_size - 1
            pad_r = 0
        else:
            pad_l = (self.kernel_size - 1) // 2
            pad_r = self.kernel_size // 2

        # Pad along length dimension: (batch, channels, length)
        x = mx.pad(x, [(0, 0), (0, 0), (pad_l, pad_r)])
        return x

    def _conv1d(self, x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
        """Apply 1D convolution (channels-first)."""
        # x: (batch, in_ch, length) -> (batch, length, in_ch)
        x = mx.transpose(x, (0, 2, 1))
        y = mx.conv1d(x, weight, stride=1, padding=0)
        # y: (batch, length, out_ch) -> (batch, out_ch, length)
        y = mx.transpose(y, (0, 2, 1))
        if bias is not None:
            y = y + bias[None, :, None]
        return y

    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        x = self._conv1d(self._pad(x, x_mask), self.conv_1_weight, self.conv_1_bias)
        if self.is_gelu:
            # GELU approximation: x * sigmoid(1.702 * x)
            x = x * mx.sigmoid(1.702 * x)
        else:
            x = mx.maximum(x, 0)  # ReLU

        # Note: dropout is skipped during inference
        x = self._conv1d(self._pad(x, x_mask), self.conv_2_weight, self.conv_2_bias)
        return x * x_mask


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional relative position encoding.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        heads_share: bool = True,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.k_channels = channels // n_heads

        # Q, K, V, O projection weights (channels-first conv1d with kernel=1)
        self.conv_q_weight = mx.zeros((channels, 1, channels))
        self.conv_q_bias = mx.zeros((channels,))
        self.conv_k_weight = mx.zeros((channels, 1, channels))
        self.conv_k_bias = mx.zeros((channels,))
        self.conv_v_weight = mx.zeros((channels, 1, channels))
        self.conv_v_bias = mx.zeros((channels,))
        self.conv_o_weight = mx.zeros((out_channels, 1, channels))
        self.conv_o_bias = mx.zeros((out_channels,))

        # Relative position embeddings
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            self.emb_rel_k = mx.zeros((n_heads_rel, window_size * 2 + 1, self.k_channels))
            self.emb_rel_v = mx.zeros((n_heads_rel, window_size * 2 + 1, self.k_channels))

    def _conv1d(self, x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
        """Apply 1D convolution with kernel_size=1 (channels-first)."""
        # x: (batch, in_ch, length) -> (batch, length, in_ch)
        x = mx.transpose(x, (0, 2, 1))
        y = mx.conv1d(x, weight, stride=1, padding=0)
        # y: (batch, length, out_ch) -> (batch, out_ch, length)
        y = mx.transpose(y, (0, 2, 1))
        if bias is not None:
            y = y + bias[None, :, None]
        return y

    def __call__(
        self, x: mx.array, c: mx.array, attn_mask: Optional[mx.array] = None
    ) -> mx.array:
        q = self._conv1d(x, self.conv_q_weight, self.conv_q_bias)
        k = self._conv1d(c, self.conv_k_weight, self.conv_k_bias)
        v = self._conv1d(c, self.conv_v_weight, self.conv_v_bias)

        x, _ = self.attention(q, k, v, mask=attn_mask)
        x = self._conv1d(x, self.conv_o_weight, self.conv_o_bias)
        return x

    def attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        # query, key, value: (batch, channels, length)
        b, d, t_t = query.shape
        _, _, t_s = key.shape

        # Reshape to (batch, n_heads, length, k_channels)
        query = mx.reshape(query, (b, self.n_heads, self.k_channels, t_t))
        query = mx.transpose(query, (0, 1, 3, 2))  # (b, n_h, t_t, k)
        key = mx.reshape(key, (b, self.n_heads, self.k_channels, t_s))
        key = mx.transpose(key, (0, 1, 3, 2))  # (b, n_h, t_s, k)
        value = mx.reshape(value, (b, self.n_heads, self.k_channels, t_s))
        value = mx.transpose(value, (0, 1, 3, 2))  # (b, n_h, t_s, k)

        # Scaled dot-product attention
        scores = mx.matmul(query, mx.transpose(key, (0, 1, 3, 2))) / math.sqrt(self.k_channels)
        # scores: (b, n_h, t_t, t_s)

        # Add relative position bias if enabled
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention"
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        # Add proximal bias if enabled
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention"
            scores = scores + self._attention_bias_proximal(t_s)

        # Apply attention mask
        if mask is not None:
            scores = mx.where(mask == 0, mx.array(-1e4), scores)

        # Softmax and dropout (dropout skipped during inference)
        p_attn = mx.softmax(scores, axis=-1)

        # Apply attention to values
        output = mx.matmul(p_attn, value)  # (b, n_h, t_t, k)

        # Add relative position values if enabled
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )

        # Reshape back to (batch, channels, length)
        output = mx.transpose(output, (0, 1, 3, 2))  # (b, n_h, k, t_t)
        output = mx.reshape(output, (b, d, t_t))

        return output, p_attn

    def _matmul_with_relative_values(self, x: mx.array, y: mx.array) -> mx.array:
        """
        x: (b, h, l, m)
        y: (h or 1, m, d)
        ret: (b, h, l, d)
        """
        return mx.matmul(x, y[None, :, :, :])

    def _matmul_with_relative_keys(self, x: mx.array, y: mx.array) -> mx.array:
        """
        x: (b, h, l, d)
        y: (h or 1, m, d)
        ret: (b, h, l, m)
        """
        return mx.matmul(x, mx.transpose(y[None, :, :, :], (0, 1, 3, 2)))

    def _get_relative_embeddings(self, relative_embeddings: mx.array, length: int) -> mx.array:
        """Get relevant relative position embeddings for given length."""
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1

        if pad_length > 0:
            # Pad along the position dimension (axis 1)
            padded = mx.pad(relative_embeddings, [(0, 0), (pad_length, pad_length), (0, 0)])
        else:
            padded = relative_embeddings

        return padded[:, slice_start_position:slice_end_position, :]

    def _relative_position_to_absolute_position(self, x: mx.array) -> mx.array:
        """
        Convert relative position scores to absolute position.
        x: (b, h, l, 2*l-1)
        ret: (b, h, l, l)
        """
        batch, heads, length, _ = x.shape

        # Pad to shift from relative to absolute indexing
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1)])

        # Reshape and pad
        x_flat = mx.reshape(x, (batch, heads, length * 2 * length))
        x_flat = mx.pad(x_flat, [(0, 0), (0, 0), (0, length - 1)])

        # Reshape and slice
        x_final = mx.reshape(x_flat, (batch, heads, length + 1, 2 * length - 1))
        x_final = x_final[:, :, :length, length - 1:]

        return x_final

    def _absolute_position_to_relative_position(self, x: mx.array) -> mx.array:
        """
        Convert absolute position attention to relative position.
        x: (b, h, l, l)
        ret: (b, h, l, 2*l-1)
        """
        batch, heads, length, _ = x.shape

        # Pad along column
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, length - 1)])
        x_flat = mx.reshape(x, (batch, heads, length * length + length * (length - 1)))

        # Add zeros at beginning to skew after reshape
        x_flat = mx.pad(x_flat, [(0, 0), (0, 0), (length, 0)])
        x_final = mx.reshape(x_flat, (batch, heads, length, 2 * length))
        x_final = x_final[:, :, :, 1:]

        return x_final

    def _attention_bias_proximal(self, length: int) -> mx.array:
        """Bias for self-attention to encourage attention to close positions."""
        r = mx.arange(length, dtype=mx.float32)
        diff = r[None, :] - r[:, None]
        return -mx.log1p(mx.abs(diff))[None, None, :, :]


class Encoder(nn.Module):
    """
    Transformer encoder with multi-head attention and FFN.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.attn_layers = []
        self.norm_layers_1 = []
        self.ffn_layers = []
        self.norm_layers_2 = []

        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        # Create attention mask: (batch, 1, length) * (batch, length, 1) = (batch, length, length)
        attn_mask = x_mask[:, :, :, None] * x_mask[:, :, None, :]
        # Reshape for attention: (batch, 1, length, length)
        attn_mask = mx.transpose(attn_mask, (0, 1, 3, 2))

        x = x * x_mask

        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            y = attn(x, x, attn_mask)
            # Note: dropout skipped during inference
            x = norm1(x + y)

            y = ffn(x, x_mask)
            x = norm2(x + y)

        return x * x_mask

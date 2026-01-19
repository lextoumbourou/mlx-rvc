"""Flow modules for RVC (normalizing flows)."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from .commons import fused_add_tanh_sigmoid_multiply, WeightNormConv1d, Conv1d


class WN(nn.Module):
    """
    WaveNet-style dilated convolution network.

    Used in flow layers for modeling complex dependencies.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = []
        self.res_skip_layers = []

        if gin_channels != 0:
            self.cond_layer = WeightNormConv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(
                WeightNormConv1d(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )

            # Last layer has different output channels
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            self.res_skip_layers.append(
                WeightNormConv1d(hidden_channels, res_skip_channels, 1)
            )

    def __call__(
        self, x: mx.array, x_mask: mx.array, g: Optional[mx.array] = None
    ) -> mx.array:
        output = mx.zeros_like(x)

        if g is not None and self.gin_channels != 0:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(
            zip(self.in_layers, self.res_skip_layers)
        ):
            x_in = in_layer(x)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = mx.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            # Note: dropout skipped during inference

            res_skip_acts = res_skip_layer(acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return output * x_mask


class Flip(nn.Module):
    """
    Flip operation for flow - reverses channel order.
    """

    def __call__(
        self, x: mx.array, x_mask: mx.array, g: Optional[mx.array] = None, reverse: bool = False
    ) -> tuple[mx.array, mx.array]:
        # Simply reverse the channel order using indexing
        # x shape: (batch, channels, length)
        x = x[:, ::-1, :]
        if not reverse:
            return x, mx.zeros((1,))
        else:
            return x, mx.zeros((1,))


class ResidualCouplingLayer(nn.Module):
    """
    Residual coupling layer for normalizing flows.

    Splits input into two halves, transforms one half conditioned on the other.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()
        assert channels % 2 == 0, "channels should be divisible by 2"
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.gin_channels = gin_channels

        # Pre-projection
        self.pre = Conv1d(self.half_channels, hidden_channels, 1)

        # WaveNet encoder
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )

        # Post-projection (outputs mean, optionally log-std)
        out_channels = self.half_channels * (2 - int(mean_only))
        self.post = Conv1d(hidden_channels, out_channels, 1)

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        g: Optional[mx.array] = None,
        reverse: bool = False,
    ) -> tuple[mx.array, mx.array]:
        # Split into two halves
        x0 = x[:, : self.half_channels, :]
        x1 = x[:, self.half_channels :, :]

        # Transform with WN
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m = stats[:, : self.half_channels, :]
            logs = stats[:, self.half_channels :, :]
        else:
            m = stats
            logs = mx.zeros_like(m)

        if not reverse:
            # Forward: x1 = m + x1 * exp(logs)
            x1 = m + x1 * mx.exp(logs) * x_mask
            x = mx.concatenate([x0, x1], axis=1)
            logdet = mx.sum(logs, axis=(1, 2))
            return x, logdet
        else:
            # Reverse: x1 = (x1 - m) * exp(-logs)
            x1 = (x1 - m) * mx.exp(-logs) * x_mask
            x = mx.concatenate([x0, x1], axis=1)
            return x, mx.zeros((1,))


class ResidualCouplingBlock(nn.Module):
    """
    Stack of residual coupling layers with flip operations.

    This is the main flow block used in the RVC synthesizer.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = []
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        g: Optional[mx.array] = None,
        reverse: bool = False,
    ) -> mx.array:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            # Reverse order for inverse flow
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x

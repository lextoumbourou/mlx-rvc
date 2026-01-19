"""MLX model implementations for RVC."""

from .synthesizer import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from .generator import GeneratorNSF
from .encoder import TextEncoder
from .flow import ResidualCouplingBlock, ResidualCouplingLayer, WN, Flip
from .resblock import ResBlock1, ResBlock2
from .nsf import SineGen, SourceModuleHnNSF
from .attentions import Encoder, MultiHeadAttention, FFN, LayerNorm
from .commons import (
    WeightNormConv1d,
    WeightNormConvTranspose1d,
    Conv1d,
    leaky_relu,
    sequence_mask,
    fused_add_tanh_sigmoid_multiply,
    get_padding,
    LRELU_SLOPE,
)

__all__ = [
    # Main models
    "SynthesizerTrnMs768NSFsid",
    "SynthesizerTrnMs768NSFsid_nono",
    "GeneratorNSF",
    "TextEncoder",
    # Flow
    "ResidualCouplingBlock",
    "ResidualCouplingLayer",
    "WN",
    "Flip",
    # Encoder components
    "Encoder",
    "MultiHeadAttention",
    "FFN",
    "LayerNorm",
    # Vocoder components
    "ResBlock1",
    "ResBlock2",
    "SineGen",
    "SourceModuleHnNSF",
    # Common layers
    "WeightNormConv1d",
    "WeightNormConvTranspose1d",
    "Conv1d",
    "leaky_relu",
    "sequence_mask",
    "fused_add_tanh_sigmoid_multiply",
    "get_padding",
    "LRELU_SLOPE",
]

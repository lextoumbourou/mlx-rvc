"""Load weights into RVC-MLX models."""

import json
import math
from pathlib import Path
from typing import Any, Union

import mlx.core as mx
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from safetensors.numpy import load_file as load_safetensors


# Valid sample rate configurations
# Maps sample rate -> (expected_upsample_rates, expected_upsample_kernels)
SAMPLE_RATE_CONFIGS = {
    32000: {
        "upsample_rates": [10, 8, 2, 2],
        "upsample_kernel_sizes": [20, 16, 4, 4],
        "upp": 320,  # 10 * 8 * 2 * 2
    },
    40000: {
        "upsample_rates": [10, 10, 2, 2],
        "upsample_kernel_sizes": [20, 20, 4, 4],
        "upp": 400,  # 10 * 10 * 2 * 2
    },
    48000: {
        "upsample_rates": [12, 10, 2, 2],
        "upsample_kernel_sizes": [24, 20, 4, 4],
        "upp": 480,  # 12 * 10 * 2 * 2
    },
}


def load_checkpoint(path: Union[str, Path]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Load weights from a checkpoint file.

    Supports both PyTorch .pth files and SafeTensors .safetensors files.

    Args:
        path: Path to checkpoint file

    Returns:
        Tuple of (weights_dict, config_dict)
    """
    path = Path(path)

    if path.suffix == ".safetensors":
        weights = load_safetensors(str(path))
        config = {}

        # Try to load config from accompanying JSON (same name)
        config_path = path.with_suffix(".json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        # Also check for shared config.json in same directory
        shared_config_path = path.parent / "config.json"
        if not config and shared_config_path.exists():
            with open(shared_config_path) as f:
                all_configs = json.load(f)

            # Try to match by filename (e.g., f0G40k -> 40000)
            filename = path.stem.lower()
            for sr_key, sr_config in all_configs.items():
                sr_value = sr_config.get("sample_rate", sr_key)
                # Convert sample rate to various filename formats
                # e.g., 40000 -> "40k", "40000"
                sr_str = str(sr_value)
                sr_k = f"{sr_value // 1000}k"  # e.g., "40k"

                if sr_key in filename or sr_str in filename or sr_k in filename:
                    config = sr_config.copy()  # Don't modify original
                    # Normalize field names (sample_rate -> sr)
                    if "sample_rate" in config and "sr" not in config:
                        config["sr"] = config["sample_rate"]
                    break

        # Validate configuration if present
        if "sr" in config:
            validate_config(config)

        return weights, config

    elif path.suffix == ".pth":
        if not HAS_TORCH:
            raise ImportError("PyTorch required to load .pth files. Install with: pip install torch")

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

        # Extract weights
        if "weight" in ckpt:
            state_dict = ckpt["weight"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Convert to numpy with proper transpositions
        weights = {}
        for key, value in state_dict.items():
            arr = value.float().numpy()
            weights[key] = _convert_weight(key, arr)

        # Extract config (pass weights for v1/v2 detection)
        config = _parse_config(ckpt, weights)

        # Validate configuration
        validate_config(config)

        return weights, config

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _convert_weight(key: str, arr: np.ndarray) -> np.ndarray:
    """Convert a weight tensor from PyTorch to MLX format."""
    # Weight normalization weights need transposition
    if "weight_v" in key:
        if ".ups." in key:
            # ConvTranspose1d: (in, out, kernel) -> (in, kernel, out)
            return np.transpose(arr, (0, 2, 1))
        else:
            # Conv1d: (out, in, kernel) -> (out, kernel, in)
            return np.transpose(arr, (0, 2, 1))

    # Regular conv weights
    if ".weight" in key and len(arr.shape) == 3:
        if ".ups." in key:
            return np.transpose(arr, (0, 2, 1))
        elif "emb_" not in key and "l_linear" not in key:
            return np.transpose(arr, (0, 2, 1))

    return arr


def _parse_sample_rate(sr_value) -> int:
    """Parse sample rate from various formats (int, string like '40k')."""
    if isinstance(sr_value, int):
        return sr_value
    if isinstance(sr_value, str):
        sr_value = sr_value.lower().strip()
        if sr_value.endswith("k"):
            return int(sr_value[:-1]) * 1000
        return int(sr_value)
    return int(sr_value)


def _detect_model_version(weights: dict) -> tuple[str, int]:
    """
    Detect model version (v1 or v2) from weights.

    Returns:
        Tuple of (version, in_channels):
        - v1: ("v1", 256)
        - v2: ("v2", 768)
    """
    # Check emb_phone weight shape to determine input dimension
    emb_phone_key = "enc_p.emb_phone.weight"
    if emb_phone_key in weights:
        shape = weights[emb_phone_key].shape
        # Shape is (hidden_channels, in_channels) = (192, 256 or 768)
        in_channels = shape[1] if len(shape) == 2 else shape[0]
        if in_channels == 256:
            return "v1", 256
        elif in_channels == 768:
            return "v2", 768
    # Default to v2
    return "v2", 768


def _parse_config(ckpt: dict, weights: dict = None) -> dict[str, Any]:
    """Parse config from RVC checkpoint."""
    config = {}

    # Direct fields
    if "f0" in ckpt:
        config["f0"] = bool(ckpt["f0"])
    if "version" in ckpt:
        config["version"] = ckpt["version"]
    if "sr" in ckpt:
        config["sr"] = _parse_sample_rate(ckpt["sr"])

    # Parse config list
    if "config" in ckpt:
        cfg = ckpt["config"]
        if isinstance(cfg, (list, tuple)) and len(cfg) >= 17:
            config["spec_channels"] = cfg[0]
            config["segment_size"] = cfg[1]
            config["inter_channels"] = cfg[2]
            config["hidden_channels"] = cfg[3]
            config["filter_channels"] = cfg[4]
            config["n_heads"] = cfg[5]
            config["n_layers"] = cfg[6]
            config["kernel_size"] = cfg[7]
            config["p_dropout"] = cfg[8]
            config["resblock"] = cfg[9]
            config["resblock_kernel_sizes"] = cfg[10]
            config["resblock_dilation_sizes"] = cfg[11]
            config["upsample_rates"] = cfg[12]
            config["upsample_initial_channel"] = cfg[13]
            config["upsample_kernel_sizes"] = cfg[14]
            config["spk_embed_dim"] = cfg[15]
            config["gin_channels"] = cfg[16]
            if len(cfg) > 17:
                config["sr"] = _parse_sample_rate(cfg[17])

    # Detect model version from weights if available
    if weights is not None:
        detected_version, in_channels = _detect_model_version(weights)
        config["in_channels"] = in_channels
        # Only set version if not already present
        if "version" not in config:
            config["version"] = detected_version

    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate model configuration for sample rate and upsample factors.

    Args:
        config: Model configuration dictionary

    Raises:
        ValueError: If configuration is invalid or inconsistent
    """
    sr = config.get("sr")
    upsample_rates = config.get("upsample_rates")

    if sr is None:
        raise ValueError(
            "Model config missing required 'sr' (sample rate) field. "
            "This may be an unsupported model format."
        )

    if sr not in SAMPLE_RATE_CONFIGS:
        raise ValueError(
            f"Unsupported sample rate: {sr}. "
            f"Supported: {list(SAMPLE_RATE_CONFIGS.keys())}"
        )

    if upsample_rates is None:
        raise ValueError("Model config missing 'upsample_rates' field.")

    # Validate upsample factor matches sample rate
    actual_upp = math.prod(upsample_rates)
    expected_upp = SAMPLE_RATE_CONFIGS[sr]["upp"]

    if actual_upp != expected_upp:
        raise ValueError(
            f"Upsample factor mismatch for {sr}Hz model: "
            f"got {actual_upp}x from {upsample_rates}, expected {expected_upp}x. "
            f"Expected rates: {SAMPLE_RATE_CONFIGS[sr]['upsample_rates']}"
        )


def load_model(model, weights: dict[str, np.ndarray]) -> None:
    """
    Load weights into an RVC-MLX model.

    Args:
        model: SynthesizerTrnMs768NSFsid or similar model
        weights: Dictionary of weight arrays
    """
    # Load encoder weights (enc_p)
    _load_text_encoder(model.enc_p, weights, "enc_p")

    # Load flow weights
    _load_flow(model.flow, weights, "flow")

    # Load decoder/generator weights (dec)
    _load_generator(model.dec, weights, "dec")

    # Load speaker embedding
    if "emb_g.weight" in weights:
        model.emb_g.weight = mx.array(weights["emb_g.weight"])


def _load_text_encoder(enc, weights: dict, prefix: str) -> None:
    """Load TextEncoder weights."""
    # emb_phone (Linear)
    if f"{prefix}.emb_phone.weight" in weights:
        enc.emb_phone.weight = mx.array(weights[f"{prefix}.emb_phone.weight"])
    if f"{prefix}.emb_phone.bias" in weights:
        enc.emb_phone.bias = mx.array(weights[f"{prefix}.emb_phone.bias"])

    # emb_pitch (Embedding) - only if model has F0
    if hasattr(enc, "emb_pitch") and f"{prefix}.emb_pitch.weight" in weights:
        enc.emb_pitch.weight = mx.array(weights[f"{prefix}.emb_pitch.weight"])

    # Transformer encoder layers
    for i in range(len(enc.encoder.attn_layers)):
        _load_attention_layer(enc.encoder.attn_layers[i], weights, f"{prefix}.encoder.attn_layers.{i}")
        _load_layer_norm(enc.encoder.norm_layers_1[i], weights, f"{prefix}.encoder.norm_layers_1.{i}")
        _load_ffn(enc.encoder.ffn_layers[i], weights, f"{prefix}.encoder.ffn_layers.{i}")
        _load_layer_norm(enc.encoder.norm_layers_2[i], weights, f"{prefix}.encoder.norm_layers_2.{i}")

    # Output projection (Conv1d)
    if f"{prefix}.proj.weight" in weights:
        enc.proj.weight = mx.array(weights[f"{prefix}.proj.weight"])
    if f"{prefix}.proj.bias" in weights:
        enc.proj.bias = mx.array(weights[f"{prefix}.proj.bias"])


def _load_attention_layer(attn, weights: dict, prefix: str) -> None:
    """Load MultiHeadAttention weights."""
    # Conv projections (kernel_size=1)
    for name in ["conv_q", "conv_k", "conv_v", "conv_o"]:
        w_key = f"{prefix}.{name}.weight"
        b_key = f"{prefix}.{name}.bias"
        if w_key in weights:
            setattr(attn, f"{name}_weight", mx.array(weights[w_key]))
        if b_key in weights:
            setattr(attn, f"{name}_bias", mx.array(weights[b_key]))

    # Relative position embeddings
    if f"{prefix}.emb_rel_k" in weights:
        attn.emb_rel_k = mx.array(weights[f"{prefix}.emb_rel_k"])
    if f"{prefix}.emb_rel_v" in weights:
        attn.emb_rel_v = mx.array(weights[f"{prefix}.emb_rel_v"])


def _load_layer_norm(norm, weights: dict, prefix: str) -> None:
    """Load LayerNorm weights."""
    if f"{prefix}.gamma" in weights:
        norm.gamma = mx.array(weights[f"{prefix}.gamma"])
    if f"{prefix}.beta" in weights:
        norm.beta = mx.array(weights[f"{prefix}.beta"])


def _load_ffn(ffn, weights: dict, prefix: str) -> None:
    """Load FFN weights."""
    if f"{prefix}.conv_1.weight" in weights:
        ffn.conv_1_weight = mx.array(weights[f"{prefix}.conv_1.weight"])
    if f"{prefix}.conv_1.bias" in weights:
        ffn.conv_1_bias = mx.array(weights[f"{prefix}.conv_1.bias"])
    if f"{prefix}.conv_2.weight" in weights:
        ffn.conv_2_weight = mx.array(weights[f"{prefix}.conv_2.weight"])
    if f"{prefix}.conv_2.bias" in weights:
        ffn.conv_2_bias = mx.array(weights[f"{prefix}.conv_2.bias"])


def _load_flow(flow, weights: dict, prefix: str) -> None:
    """Load ResidualCouplingBlock weights."""
    for i, flow_layer in enumerate(flow.flows):
        layer_prefix = f"{prefix}.flows.{i}"

        # Skip Flip layers (no weights)
        if flow_layer.__class__.__name__ == "Flip":
            continue

        # ResidualCouplingLayer
        _load_conv1d(flow_layer.pre, weights, f"{layer_prefix}.pre")
        _load_conv1d(flow_layer.post, weights, f"{layer_prefix}.post")
        _load_wn(flow_layer.enc, weights, f"{layer_prefix}.enc")


def _load_wn(wn, weights: dict, prefix: str) -> None:
    """Load WN (WaveNet) weights."""
    # Condition layer
    if hasattr(wn, "cond_layer") and f"{prefix}.cond_layer.weight_v" in weights:
        wn.cond_layer.weight_v = mx.array(weights[f"{prefix}.cond_layer.weight_v"])
        wn.cond_layer.weight_g = mx.array(weights[f"{prefix}.cond_layer.weight_g"])
        if f"{prefix}.cond_layer.bias" in weights:
            wn.cond_layer.bias = mx.array(weights[f"{prefix}.cond_layer.bias"])

    # In layers and res_skip layers
    for i in range(len(wn.in_layers)):
        _load_weight_norm_conv(wn.in_layers[i], weights, f"{prefix}.in_layers.{i}")
        _load_weight_norm_conv(wn.res_skip_layers[i], weights, f"{prefix}.res_skip_layers.{i}")


def _load_generator(dec, weights: dict, prefix: str) -> None:
    """Load GeneratorNSF weights."""
    # Source module
    if f"{prefix}.m_source.l_linear.weight" in weights:
        dec.m_source.l_linear.weight = mx.array(weights[f"{prefix}.m_source.l_linear.weight"])
    if f"{prefix}.m_source.l_linear.bias" in weights:
        dec.m_source.l_linear.bias = mx.array(weights[f"{prefix}.m_source.l_linear.bias"])

    # conv_pre
    _load_conv1d(dec.conv_pre, weights, f"{prefix}.conv_pre")

    # conv_post
    _load_conv1d(dec.conv_post, weights, f"{prefix}.conv_post")

    # cond (speaker conditioning)
    if hasattr(dec, "cond") and f"{prefix}.cond.weight" in weights:
        _load_conv1d(dec.cond, weights, f"{prefix}.cond")

    # Upsample layers
    for i, ups in enumerate(dec.ups):
        _load_weight_norm_conv_transpose(ups, weights, f"{prefix}.ups.{i}")

    # Noise convs
    for i, conv in enumerate(dec.noise_convs):
        _load_conv1d(conv, weights, f"{prefix}.noise_convs.{i}")

    # ResBlocks
    for i, resblock in enumerate(dec.resblocks):
        if hasattr(resblock, "convs1"):
            # ResBlock1
            for j in range(len(resblock.convs1)):
                _load_weight_norm_conv(resblock.convs1[j], weights, f"{prefix}.resblocks.{i}.convs1.{j}")
                _load_weight_norm_conv(resblock.convs2[j], weights, f"{prefix}.resblocks.{i}.convs2.{j}")
        elif hasattr(resblock, "convs"):
            # ResBlock2
            for j in range(len(resblock.convs)):
                _load_weight_norm_conv(resblock.convs[j], weights, f"{prefix}.resblocks.{i}.convs.{j}")


def _load_conv1d(conv, weights: dict, prefix: str) -> None:
    """Load Conv1d weights."""
    if f"{prefix}.weight" in weights:
        conv.weight = mx.array(weights[f"{prefix}.weight"])
    if f"{prefix}.bias" in weights:
        conv.bias = mx.array(weights[f"{prefix}.bias"])


def _load_weight_norm_conv(conv, weights: dict, prefix: str) -> None:
    """Load WeightNormConv1d weights."""
    if f"{prefix}.weight_v" in weights:
        conv.weight_v = mx.array(weights[f"{prefix}.weight_v"])
    if f"{prefix}.weight_g" in weights:
        conv.weight_g = mx.array(weights[f"{prefix}.weight_g"])
    if f"{prefix}.bias" in weights:
        conv.bias = mx.array(weights[f"{prefix}.bias"])


def _load_weight_norm_conv_transpose(conv, weights: dict, prefix: str) -> None:
    """Load WeightNormConvTranspose1d weights."""
    if f"{prefix}.weight_v" in weights:
        conv.weight_v = mx.array(weights[f"{prefix}.weight_v"])
    if f"{prefix}.weight_g" in weights:
        conv.weight_g = mx.array(weights[f"{prefix}.weight_g"])
    if f"{prefix}.bias" in weights:
        conv.bias = mx.array(weights[f"{prefix}.bias"])

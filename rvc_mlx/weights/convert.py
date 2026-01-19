"""Convert RVC PyTorch weights to MLX SafeTensors format."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from safetensors.numpy import save_file


def convert_conv1d_weight(weight: np.ndarray) -> np.ndarray:
    """
    Convert Conv1d weight from PyTorch to MLX format.

    PyTorch: (out_channels, in_channels, kernel_size)
    MLX: (out_channels, kernel_size, in_channels)
    """
    return np.transpose(weight, (0, 2, 1))


def convert_conv_transpose1d_weight(weight: np.ndarray) -> np.ndarray:
    """
    Convert ConvTranspose1d weight from PyTorch to MLX format.

    PyTorch: (in_channels, out_channels, kernel_size)
    MLX: (in_channels, kernel_size, out_channels)
    """
    return np.transpose(weight, (0, 2, 1))


# Prefixes to exclude (training-only weights)
TRAINING_ONLY_PREFIXES = ["enc_q"]


def convert_checkpoint(
    checkpoint_path: str,
    output_path: str,
    inference_only: bool = True,
) -> dict[str, Any]:
    """
    Convert RVC PyTorch checkpoint to MLX SafeTensors.

    Args:
        checkpoint_path: Path to PyTorch .pth file
        output_path: Path to output .safetensors file
        inference_only: If True, exclude training-only weights (enc_q)

    Returns:
        Config dictionary extracted from checkpoint
    """
    if torch is None:
        raise ImportError("PyTorch is required for weight conversion. Install with: pip install torch")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract model state dict
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "weight" in ckpt:
        state_dict = ckpt["weight"]
    else:
        state_dict = ckpt

    # Extract config if available
    config = {}
    if "config" in ckpt:
        config["config"] = ckpt["config"]
    if "f0" in ckpt:
        config["f0"] = ckpt["f0"]
    if "version" in ckpt:
        config["version"] = ckpt["version"]
    if "sr" in ckpt:
        config["sr"] = ckpt["sr"]

    print(f"Found {len(state_dict)} parameters")

    # Filter out training-only weights if requested
    if inference_only:
        filtered = {}
        skipped = 0
        for key, value in state_dict.items():
            prefix = key.split(".")[0]
            if prefix in TRAINING_ONLY_PREFIXES:
                skipped += 1
                continue
            filtered[key] = value
        print(f"Skipped {skipped} training-only parameters")
        state_dict = filtered

    # Convert weights
    converted = {}
    for key, value in state_dict.items():
        # Convert to numpy float32
        arr = value.float().numpy()

        # Determine conversion based on key
        new_key, new_arr = convert_weight(key, arr)
        converted[new_key] = new_arr

    print(f"Converted {len(converted)} parameters")

    # Save as SafeTensors
    print(f"Saving to: {output_path}")
    save_file(converted, output_path)

    # Save config as JSON
    if config:
        config_path = Path(output_path).with_suffix(".json")
        print(f"Saving config to: {config_path}")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    return config


def convert_weight(key: str, arr: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Convert a single weight tensor.

    Handles:
    - Conv1d weights (transpose)
    - ConvTranspose1d weights (transpose)
    - Weight normalization (weight_v, weight_g)
    - Linear weights (no change needed)
    - Embeddings (no change needed)
    - Biases (no change needed)

    Returns:
        Tuple of (new_key, converted_array)
    """
    # Identify weight type and convert accordingly

    # Weight normalization weights need transposition
    if "weight_v" in key:
        if is_conv_transpose(key):
            # ConvTranspose1d weight_v: (in, out, kernel) -> (in, kernel, out)
            return key, convert_conv_transpose1d_weight(arr)
        else:
            # Conv1d weight_v: (out, in, kernel) -> (out, kernel, in)
            return key, convert_conv1d_weight(arr)

    # Regular conv weights (not weight-normalized)
    if ".weight" in key and not is_embedding(key) and not is_linear(key):
        if len(arr.shape) == 3:  # Conv1d weight
            if is_conv_transpose(key):
                return key, convert_conv_transpose1d_weight(arr)
            else:
                return key, convert_conv1d_weight(arr)

    # Everything else stays the same
    return key, arr


def is_conv_transpose(key: str) -> bool:
    """Check if key is from a transposed convolution."""
    # In RVC, ups.* are ConvTranspose1d
    return ".ups." in key


def is_embedding(key: str) -> bool:
    """Check if key is from an embedding layer."""
    return "emb_g" in key or "emb_pitch" in key or "emb_phone" in key


def is_linear(key: str) -> bool:
    """Check if key is from a linear layer."""
    # Linear layers in RVC: emb_phone (Linear), l_linear (source module)
    return "emb_phone" in key or "l_linear" in key


def get_model_config(sr: int = 48000) -> dict:
    """Get model configuration for a given sample rate."""
    configs = {
        32000: {
            "spec_channels": 1025,
            "segment_size": 32,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [20, 16, 4, 4],
            "spk_embed_dim": 109,
            "gin_channels": 256,
            "sr": 32000,
        },
        40000: {
            "spec_channels": 1025,
            "segment_size": 32,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 10, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [20, 20, 4, 4],
            "spk_embed_dim": 109,
            "gin_channels": 256,
            "sr": 40000,
        },
        48000: {
            "spec_channels": 1025,
            "segment_size": 32,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [12, 10, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [24, 20, 4, 4],
            "spk_embed_dim": 109,
            "gin_channels": 256,
            "sr": 48000,
        },
    }
    return configs.get(sr, configs[48000])


def main():
    parser = argparse.ArgumentParser(description="Convert RVC PyTorch weights to MLX SafeTensors")
    parser.add_argument("input", help="Input PyTorch .pth file")
    parser.add_argument("output", help="Output .safetensors file")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate (32000, 40000, 48000)")
    args = parser.parse_args()

    config = convert_checkpoint(args.input, args.output)

    # Print summary
    print("\nConversion complete!")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    if config:
        print(f"  Config: {config}")


if __name__ == "__main__":
    main()

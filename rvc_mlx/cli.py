#!/usr/bin/env python3
"""RVC-MLX command-line interface."""

import argparse
import sys
from pathlib import Path


def cmd_convert(args):
    """Convert voice in audio file."""
    from .pipeline import RVCPipeline

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Validate model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return 1

    # Load pipeline
    print(f"Loading model: {model_path}")
    pipeline = RVCPipeline.from_pretrained(model_path)

    # Run conversion
    pipeline.convert(
        input_path=args.input,
        output_path=args.output,
        speaker_id=args.speaker_id,
        f0_shift=args.pitch,
        f0_method=args.f0_method,
    )

    print(f"Done! Output saved to: {args.output}")
    return 0


def cmd_info(args):
    """Show model information."""
    from .weights import load_checkpoint

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Model: {model_path}")
    print("-" * 50)

    try:
        weights, config = load_checkpoint(model_path)

        # Basic info
        print(f"Version: {config.get('version', 'unknown')}")
        print(f"Sample Rate: {config.get('sr', 'unknown')} Hz")
        print(f"F0 (Pitch): {'Yes' if config.get('f0', True) else 'No'}")

        # Architecture
        if "hidden_channels" in config:
            print(f"\nArchitecture:")
            print(f"  Hidden Channels: {config.get('hidden_channels')}")
            print(f"  Inter Channels: {config.get('inter_channels')}")
            print(f"  Filter Channels: {config.get('filter_channels')}")
            print(f"  Attention Heads: {config.get('n_heads')}")
            print(f"  Encoder Layers: {config.get('n_layers')}")
            print(f"  Speaker Embed Dim: {config.get('spk_embed_dim')}")

        # Upsample config
        if "upsample_rates" in config:
            rates = config["upsample_rates"]
            total = 1
            for r in rates:
                total *= r
            print(f"\nUpsample Rates: {rates} (total: {total}x)")

        # Weight stats
        print(f"\nWeights: {len(weights)} tensors")

        # Count by prefix
        prefixes = {}
        for key in weights.keys():
            prefix = key.split(".")[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

        for prefix, count in sorted(prefixes.items()):
            print(f"  {prefix}: {count} params")

    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    return 0


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="rvc-mlx",
        description="RVC voice conversion for Apple Silicon using MLX",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert voice in audio file",
        description="Convert voice in an audio file using an RVC model.",
    )
    convert_parser.add_argument(
        "input",
        help="Input audio file (any format supported by FFmpeg)",
    )
    convert_parser.add_argument(
        "output",
        help="Output audio file path",
    )
    convert_parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to RVC model (.pth or .safetensors)",
    )
    convert_parser.add_argument(
        "-p", "--pitch",
        type=int,
        default=0,
        metavar="SEMITONES",
        help="Pitch shift in semitones (-12 to +12, default: 0)",
    )
    convert_parser.add_argument(
        "-s", "--speaker-id",
        type=int,
        default=0,
        help="Speaker ID for multi-speaker models (default: 0)",
    )
    convert_parser.add_argument(
        "--f0-method",
        choices=["harvest", "rmvpe"],
        default="harvest",
        help="F0 extraction method: harvest (fast) or rmvpe (better for singing, auto-downloads weights)",
    )
    convert_parser.set_defaults(func=cmd_convert)

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show model information",
        description="Display information about an RVC model.",
    )
    info_parser.add_argument(
        "model",
        help="Path to RVC model (.pth or .safetensors)",
    )
    info_parser.set_defaults(func=cmd_info)

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

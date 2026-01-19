# RVC-MLX

An MLX port of [Retrieval-based-Voice-Conversion-WebUI (RVC)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for Apple Silicon.

> **Status**: Core model implementation complete. Pipeline integration in progress.

## Features

- Native Apple Silicon acceleration via MLX
- Full SynthesizerTrnMs768NSFsid implementation
- Support for 32kHz, 40kHz, and 48kHz models
- F0 (pitch) extraction via pyworld

## Installation

```bash
pip install rvc-mlx
```

Or install from source:

```bash
git clone https://github.com/lucasnewman/rvc-mlx
cd rvc-mlx
pip install -e .
```

## Pretrained Weights

Download pretrained weights from HuggingFace:

```python
from huggingface_hub import hf_hub_download

# Download 48kHz model (recommended)
weights_path = hf_hub_download(
    repo_id="lexandstuff/rvc-mlx-weights",
    filename="v2/f0G48k.safetensors"
)

config_path = hf_hub_download(
    repo_id="lexandstuff/rvc-mlx-weights",
    filename="v2/config.json"
)
```

Available models:

| Model | Sample Rate | Size |
|-------|-------------|------|
| `v2/f0G48k.safetensors` | 48 kHz | 110 MB |
| `v2/f0G40k.safetensors` | 40 kHz | 105 MB |
| `v2/f0G32k.safetensors` | 32 kHz | 107 MB |

## Usage

```python
import json
import mlx.core as mx
from safetensors.numpy import load_file
from rvc_mlx.models import SynthesizerTrnMs768NSFsid

# Load config
with open(config_path) as f:
    config = json.load(f)["48000"]

# Create model
model = SynthesizerTrnMs768NSFsid(**config)

# Load weights (weight loading utility coming soon)
weights = load_file(weights_path)

# Run inference
# audio, mask, _ = model.infer(phone, phone_lengths, pitch, f0, sid)
```

## Architecture

RVC-MLX implements the full RVC v2 inference pipeline:

```
Audio Input
    ↓
ContentVec (feature extraction) ──→ Phone features (768-dim)
    ↓
F0 Extraction (Harvest) ──→ Pitch features
    ↓
┌─────────────────────────────────┐
│   SynthesizerTrnMs768NSFsid     │
│                                 │
│  ┌─────────────┐               │
│  │ TextEncoder │ ← phone + pitch│
│  └──────┬──────┘               │
│         ↓                       │
│  ┌─────────────┐               │
│  │    Flow     │ ← speaker emb │
│  └──────┬──────┘               │
│         ↓                       │
│  ┌─────────────┐               │
│  │GeneratorNSF │ ← F0          │
│  └──────┬──────┘               │
└─────────┼───────────────────────┘
          ↓
    Audio Output
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run specific test
pytest tests/test_synthesizer.py -v
```

## Acknowledgments

- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for the original implementation
- [MLX](https://github.com/ml-explore/mlx) team at Apple for the framework

## License

MIT License - see [LICENSE](LICENSE) for details.

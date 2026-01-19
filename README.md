# RVC-MLX

An MLX port of [Retrieval-based-Voice-Conversion-WebUI (RVC)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for Apple Silicon.

> **Status**: Phase 1 complete! Voice conversion is fully working.

## Features

- Native Apple Silicon acceleration via MLX
- Full SynthesizerTrnMs768NSFsid implementation
- Support for 32kHz, 40kHz, and 48kHz models
- F0 (pitch) extraction via Harvest or RMVPE
- RMVPE support for better singing voice pitch detection
- Simple CLI and Python API

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

### Command Line

```bash
# Basic voice conversion
rvc-mlx convert input.wav output.wav --model voice.pth

# With pitch shift (+5 semitones for higher pitch)
rvc-mlx convert input.wav output.wav --model voice.pth --pitch 5

# Use RMVPE for better singing voice detection
rvc-mlx convert input.wav output.wav --model voice.pth --f0-method rmvpe

# Show model information
rvc-mlx info voice.pth
```

### Python API

```python
from rvc_mlx import RVCPipeline

# Load pipeline from model file
pipeline = RVCPipeline.from_pretrained("voice.pth")

# Convert voice
pipeline.convert(
    input_path="input.wav",
    output_path="output.wav",
    f0_shift=0,  # Pitch shift in semitones
)
```

Or use the simple function:

```python
from rvc_mlx import convert_voice

convert_voice(
    input_path="input.wav",
    output_path="output.wav",
    model_path="voice.pth",
    f0_shift=5,  # Shift pitch up 5 semitones
)
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

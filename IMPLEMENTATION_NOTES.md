# MLX-RVC Implementation Notes

## Overview

This document outlines the plan for building an MLX-native implementation of RVC (Retrieval-based Voice Conversion) for Apple Silicon. The initial focus is on **inference-only** functionality via a command-line interface, with training and GUI to follow.

## Current Status

### Completed ✅

| Component | Location | Tests | Notes |
|-----------|----------|-------|-------|
| Audio I/O | `mlx_rvc/audio/io.py` | 12 tests | FFmpeg-based load/save |
| Audio Processing | `mlx_rvc/audio/processing.py` | ✅ | Normalize, RMS, padding |
| F0 Extraction (Harvest) | `mlx_rvc/f0/harvest.py` | 8 tests | pyworld wrapper |
| F0 Processing | `mlx_rvc/f0/processing.py` | ✅ | Pitch shift, quantization |
| Weight Norm Conv1d | `mlx_rvc/models/commons.py` | 6 tests | Validated vs PyTorch |
| SineGen | `mlx_rvc/models/nsf.py` | 5 tests | Harmonic source generation |
| SourceModuleHnNSF | `mlx_rvc/models/nsf.py` | 4 tests | NSF source module |
| ResBlock1/2 | `mlx_rvc/models/resblock.py` | 4 tests | HiFi-GAN residual blocks |
| GeneratorNSF | `mlx_rvc/models/generator.py` | 5 tests | Full decoder/vocoder |
| MultiHeadAttention | `mlx_rvc/models/attentions.py` | ✅ | With relative position encoding |
| Encoder (Transformer) | `mlx_rvc/models/attentions.py` | ✅ | 6-layer transformer |
| TextEncoder | `mlx_rvc/models/encoder.py` | 1 test | Phone + pitch encoding |
| WN (WaveNet) | `mlx_rvc/models/flow.py` | ✅ | Dilated convolutions |
| ResidualCouplingBlock | `mlx_rvc/models/flow.py` | 1 test | Normalizing flow |
| SynthesizerTrnMs768NSFsid | `mlx_rvc/models/synthesizer.py` | 5 tests | Full synthesizer model |
| Weight Conversion | `mlx_rvc/weights/convert.py` | ✅ | PyTorch → SafeTensors |
| Weight Loading | `mlx_rvc/weights/loader.py` | ✅ | Load weights into MLX models |
| Inference Pipeline | `mlx_rvc/pipeline.py` | ✅ | End-to-end voice conversion |
| CLI Interface | `mlx_rvc/cli.py` | ✅ | `mlx-rvc convert` and `mlx-rvc info` commands |
| RMVPE F0 Extraction | `mlx-rmvpe` (PyPI) | ✅ | Separate package, auto-downloads from HuggingFace |
| FAISS Index Blending | `mlx_rvc/index/faiss_index.py` | ✅ | Optional dependency, improves voice similarity |
| V1 Model Support | `mlx_rvc/weights/loader.py` | ✅ | Auto-detects v1 (256-dim) vs v2 (768-dim) |
| Multi-Sample-Rate | `mlx_rvc/weights/loader.py` | 19 tests | 32kHz, 40kHz, 48kHz validated |

**Total: 78 tests passing**

**🎉 Voice conversion is working!** Successfully tested with V1 and V2 RVC models.

### In Progress 🔄

*No items currently in progress*

### Not Started 📋

- No-F0 Model Support (models without pitch conditioning)

---

## RMVPE Implementation ✅ COMPLETE

**Goal**: Port RMVPE (Robust Model for Vocal Pitch Estimation) to MLX for better singing voice F0 extraction.

**Status**: RMVPE has been extracted into a separate PyPI package: **[mlx-rmvpe](https://pypi.org/project/mlx-rmvpe/)**

### Installation

RMVPE is automatically installed as a dependency of mlx-rvc. Weights are auto-downloaded from HuggingFace on first use.

```bash
# Already included via mlx-rvc dependencies
uv pip install mlx-rmvpe
```

### Usage

```python
from mlx_rmvpe import RMVPE

# Auto-downloads weights from HuggingFace
model = RMVPE.from_pretrained()

# Extract F0 from audio (16kHz)
f0 = model.infer_from_audio(audio)
```

Or via CLI:

```bash
mlx-rvc convert input.wav output.wav --model voice.pth --f0-method rmvpe
```

### Why RMVPE?

- Current Harvest method works for speech but struggles with singing
- RMVPE is specifically designed for "Vocal Pitch Estimation in Polyphonic Music"
- It's the recommended F0 method in RVC (described as "best effect" in the UI)
- Frame rate (100fps with hop=160) aligns better with ContentVec (~50fps) than Harvest

### Architecture

See [mlx-rmvpe IMPLEMENTATION_NOTES.md](https://github.com/lexandstuff/mlx-rmvpe/blob/main/IMPLEMENTATION_NOTES.md) for full architecture details.

Summary:
- Deep U-Net encoder/decoder with skip connections
- BiGRU (implemented as two separate GRUs for MLX)
- 360 pitch bins decoded to Hz via cents mapping
- ~15M parameters

### HuggingFace Weights

Weights are hosted at: `lexandstuff/mlx-rmvpe`

```python
# Automatic download (default)
model = RMVPE.from_pretrained()

# Custom repo
model = RMVPE.from_pretrained(repo_id="my-org/my-rmvpe")
```

---

## FAISS Index Blending ✅ COMPLETE

**Goal**: Blend ContentVec features with similar features from the training set to improve voice similarity.

**Status**: Implemented in `mlx_rvc/index/faiss_index.py` as an optional feature.

### Installation

FAISS is an optional dependency. Install with:

```bash
uv pip install faiss-cpu
# or
uv sync --extra index
```

### Usage

```python
from mlx_rvc import RVCPipeline

pipeline = RVCPipeline.from_pretrained("voice.pth")
pipeline.convert(
    input_path="input.wav",
    output_path="output.wav",
    index_path="voice.index",  # Path to FAISS index
    index_rate=0.5,            # Blend ratio (0=original, 1=index)
)
```

Or via CLI:

```bash
mlx-rvc convert input.wav output.wav --model voice.pth --index voice.index --index-rate 0.5
```

### How It Works

RVC uses FAISS to store ContentVec features from the training data. During inference, input features are blended with similar features retrieved from the index:

1. For each input frame, find k=8 nearest neighbors in the index
2. Compute weights as inverse square of distances: `weight = 1 / (distance² + ε)`
3. Weighted average of retrieved features
4. Blend with original: `output = retrieved * index_rate + original * (1 - index_rate)`

### Index File Format

RVC index files (`.index`) are FAISS IndexIVFFlat format:
- **Dimension**: 768 (matches ContentVec output)
- **Vectors**: Training set ContentVec features
- Features can be reconstructed via `index.reconstruct_n()`

### OpenMP Conflict Workaround

Both FAISS and MLX use OpenMP on macOS, which causes library conflicts and segfaults when both are loaded. The solution:

```python
import faiss
# Use single-threaded mode to avoid OpenMP conflicts with MLX
faiss.omp_set_num_threads(1)
```

This is automatically applied when importing `mlx_rvc.index`. The performance impact is minimal since index search is not the bottleneck.

---

## V1 Model Support ✅ COMPLETE

**Goal**: Support legacy RVC V1 models that use 256-dim HuBERT features instead of 768-dim ContentVec.

**Status**: Fully implemented with automatic detection.

### How It Works

V1 and V2 models differ only in their input feature dimension:
- **V2**: 768-dim ContentVec features
- **V1**: 256-dim HuBERT features

The model version is automatically detected by checking the shape of `enc_p.emb_phone.weight`:
- Shape `[192, 768]` → V2 model
- Shape `[192, 256]` → V1 model

For V1 models, we use the first 256 dimensions of ContentVec output, which provides compatible features.

### Usage

No special configuration needed - V1 models are automatically detected:

```bash
# V1 model (auto-detected)
mlx-rvc convert input.wav output.wav --model v1_model.pth

# V2 model (auto-detected)
mlx-rvc convert input.wav output.wav --model v2_model.pth
```

The CLI will show the detected version:
```
Loaded RVC model: v1 @ 40000Hz (in_channels=256)
```

### Technical Details

Changes made to support V1:
- `loader.py`: Added `_detect_model_version()` to check weight shapes
- `synthesizer.py`: Added `in_channels` parameter (default 768)
- `pipeline.py`: Slices ContentVec features to 256 dims for V1 models

---

## What We Already Have

### mlx-contentvec (PyPI Package)

Available as **[mlx-contentvec](https://pypi.org/project/mlx-contentvec/)** on PyPI. Provides the ContentVec/HuBERT feature extractor:

- **Input**: Raw audio waveform @ 16kHz, shape `(batch, samples)`
- **Output**: Semantic features @ ~50fps, shape `(batch, frames, 768)`
- **Purpose**: Extracts speaker-agnostic phonetic content from speech
- **Status**: Production-ready, numerically validated against PyTorch reference
- **Weights**: Auto-downloaded from HuggingFace (`lexandstuff/mlx-contentvec`)

```python
from mlx_contentvec import ContentvecModel

# Auto-downloads weights from HuggingFace
model = ContentvecModel.from_pretrained()
result = model(audio_tensor)
features = result["x"]  # (batch, frames, 768)
```

### mlx-rmvpe (PyPI Package)

Available as **[mlx-rmvpe](https://pypi.org/project/mlx-rmvpe/)** on PyPI. Provides RMVPE F0 extraction:

- **Input**: Raw audio waveform @ 16kHz
- **Output**: F0 contour @ 100fps in Hz
- **Purpose**: Robust pitch estimation for singing voice
- **Status**: Production-ready, validated against PyTorch reference (mean diff: 1.29 Hz)
- **Weights**: Auto-downloaded from HuggingFace (`lexandstuff/mlx-rmvpe`)

```python
from mlx_rmvpe import RMVPE

# Auto-downloads weights from HuggingFace
model = RMVPE.from_pretrained()
f0 = model.infer_from_audio(audio)  # F0 in Hz
```

## RVC Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RVC INFERENCE PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: audio.wav (any format/sample rate)                                   │
│              │                                                               │
│              ▼                                                               │
│  ┌───────────────────────┐                                                   │
│  │   Audio Preprocessing │  FFmpeg decode → 16kHz mono → normalize           │
│  └───────────┬───────────┘                                                   │
│              │                                                               │
│              ├──────────────────────────────┐                                │
│              ▼                              ▼                                │
│  ┌───────────────────────┐     ┌───────────────────────┐                     │
│  │  ContentVec (HuBERT)  │     │    F0 Extraction      │                     │
│  │  [ALREADY DONE]       │     │    (RMVPE/Harvest)    │                     │
│  │                       │     │                       │                     │
│  │  audio → (B,T,768)    │     │  audio → pitch curve  │                     │
│  └───────────┬───────────┘     └───────────┬───────────┘                     │
│              │                              │                                │
│              ▼                              ▼                                │
│  ┌───────────────────────┐     ┌───────────────────────┐                     │
│  │  FAISS Index Blend    │     │   Pitch Processing    │                     │
│  │  (Optional)           │     │                       │                     │
│  │                       │     │  - Transpose (±12)    │                     │
│  │  Blends with training │     │  - Mel conversion     │                     │
│  │  set features         │     │  - Quantize (1-255)   │                     │
│  └───────────┬───────────┘     └───────────┬───────────┘                     │
│              │                              │                                │
│              └──────────────┬───────────────┘                                │
│                             ▼                                                │
│              ┌───────────────────────────────┐                               │
│              │     SynthesizerTrnMs768NSFsid │                               │
│              │     (Voice Conversion Model)  │                               │
│              │                               │                               │
│              │  ┌─────────────────────────┐  │                               │
│              │  │ TextEncoder             │  │                               │
│              │  │ features + pitch → μ,σ  │  │                               │
│              │  └───────────┬─────────────┘  │                               │
│              │              ▼                │                               │
│              │  ┌─────────────────────────┐  │                               │
│              │  │ Flow (Coupling Layers)  │  │                               │
│              │  │ + Speaker Embedding     │  │                               │
│              │  └───────────┬─────────────┘  │                               │
│              │              ▼                │                               │
│              │  ┌─────────────────────────┐  │                               │
│              │  │ GeneratorNSF (Decoder)  │  │                               │
│              │  │ latent + pitch → audio  │  │                               │
│              │  └─────────────────────────┘  │                               │
│              └───────────────┬───────────────┘                               │
│                              ▼                                               │
│              ┌───────────────────────────────┐                               │
│              │     Post-Processing           │                               │
│              │     - RMS volume matching     │                               │
│              │     - Resample if needed      │                               │
│              └───────────────┬───────────────┘                               │
│                              ▼                                               │
│  OUTPUT: converted_audio.wav                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components to Implement

### 1. F0 (Pitch) Extraction

**Options** (in order of preference for MLX):

| Method | Quality | Speed | MLX Feasibility |
|--------|---------|-------|-----------------|
| **RMVPE** | Best | Medium | Port needed (~180MB model) |
| **Harvest (pyworld)** | Good | Slow | Use as-is (NumPy/CPU) |
| **Crepe** | Good | Fast (GPU) | Port possible |
| **PM (Praat)** | OK | Fast | Use as-is (CPU) |

**Recommendation**: Start with **Harvest** (pyworld) for simplicity, then port **RMVPE** for quality.

**F0 Processing Required**:
```python
# Pitch shift by semitones
f0 *= pow(2, f0_up_key / 12)

# Convert to mel scale
f0_mel = 1127 * np.log(1 + f0 / 700)

# Quantize to 1-255 range for embedding lookup
f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_min) * 254 / (f0_max - f0_min) + 1
f0_coarse = np.rint(f0_mel).astype(np.int64)
```

### 2. FAISS Index / Feature Blending (Optional but Important)

The index blending improves conversion quality by replacing input speaker characteristics with training set characteristics.

**Options**:
- Use FAISS directly (CPU, works fine)
- Implement simple k-NN in MLX (if index is small)
- Make it optional for initial version

**Algorithm**:
```python
# For each frame in HuBERT output:
distances, indices = faiss_index.search(feature, k=8)
weights = 1 / (distances ** 2 + 1e-9)
weights /= weights.sum()
blended = index_rate * (weights @ retrieved_features) + (1 - index_rate) * feature
```

### 3. SynthesizerTrnMs768NSFsid (Main Voice Conversion Model)

This is the core model that needs MLX porting. It has 4 sub-components:

#### 3.1 TextEncoder
Transforms HuBERT features + pitch embeddings into latent distribution parameters.

```
Input: phone_features (B, T, 768), pitch (B, T) quantized 1-255

Architecture:
- emb_phone: Linear(768 → 192)
- emb_pitch: Embedding(256, 192)
- encoder: 6 transformer layers (192-dim, 2 heads, 768 FFN)
- proj: Conv1d(192 → 384)  # outputs μ and log(σ)

Output: (m_p, logs_p, mask) - mean, log-variance, mask
```

#### 3.2 Speaker Embedding
Simple embedding lookup for speaker identity.

```
emb_g: Embedding(num_speakers, 256)
g = emb_g(speaker_id).unsqueeze(-1)  # (B, 256, 1)
```

#### 3.3 ResidualCouplingBlock (Flow)
Normalizing flow that transforms the latent distribution with speaker conditioning.

```
Architecture:
- 4 ResidualCouplingLayer blocks
- Each layer: WaveNet-style dilated convolutions
- Alternating which half is transformed vs conditioned
- reverse=True for inference (reverse direction)

Input: z_p (sampled latent), mask, speaker_embedding
Output: z (transformed latent)
```

#### 3.4 GeneratorNSF (Neural Source Filter Decoder)

The most complex component - generates audio from latents using pitch-aware synthesis.

```
Architecture:
├── F0 Source Generation:
│   ├── Upsample pitch to audio rate (320x for 32k, 400x for 40k)
│   ├── SineGen: Generate harmonics from F0
│   └── SourceModuleHnNSF: Combine harmonic + noise sources
│
├── Main Decoder:
│   ├── Pre-conv: Conv1d(192 → 512, kernel=7)
│   ├── 4 Upsample blocks:
│   │   ├── LeakyReLU + ConvTranspose1d
│   │   ├── Add noise source via Conv1d
│   │   └── 3 parallel ResBlocks (kernels 3,7,11)
│   └── Post-conv: LeakyReLU + Conv1d(ch → 1) + Tanh
│
└── Output: (B, 1, audio_samples)
```

**Key Details**:
- Upsample rates depend on target sample rate:
  - 32k: `[10, 8, 2, 2]` → 320x total
  - 40k: `[10, 10, 2, 2]` → 400x total
  - 48k: `[12, 10, 2, 2]` → 480x total
- ResBlocks use weight normalization
- NSF source generation is crucial for pitch accuracy

### 4. Audio I/O

**Input Processing**:
- Use FFmpeg (via subprocess or av) for format handling
- Resample to 16kHz for ContentVec
- Normalize amplitude

**Output Processing**:
- Apply RMS mix (blend input/output loudness)
- Resample to target sample rate if needed
- Normalize to int16 range

### 5. Weight Conversion

RVC checkpoints are PyTorch `.pth` files with structure:
```python
{
    "weight": {...},  # state_dict
    "config": [spec_channels, segment_size, inter_channels, ...],
    "version": "v2",
    "f0": 1,  # 1=with pitch, 0=without
}
```

Need conversion script: PyTorch → SafeTensors for MLX.

## Project Structure (Current)

```
mlx_rvc/
├── __init__.py
├── audio/
│   ├── __init__.py
│   ├── io.py                 # ✅ FFmpeg-based load/save
│   └── processing.py         # ✅ Normalize, RMS, padding
├── f0/
│   ├── __init__.py
│   ├── harvest.py           # ✅ pyworld wrapper
│   └── processing.py        # ✅ Pitch shift, mel conversion
├── index/
│   ├── __init__.py
│   └── faiss_index.py       # ✅ FAISS index loading & blending
├── models/
│   ├── __init__.py          # ✅ Exports all models
│   ├── synthesizer.py       # ✅ SynthesizerTrnMs768NSFsid
│   ├── encoder.py           # ✅ TextEncoder
│   ├── flow.py              # ✅ ResidualCouplingBlock, WN, Flip
│   ├── generator.py         # ✅ GeneratorNSF
│   ├── resblock.py          # ✅ ResBlock1, ResBlock2
│   ├── nsf.py               # ✅ SineGen, SourceModuleHnNSF
│   ├── attentions.py        # ✅ MultiHeadAttention, Encoder, FFN
│   └── commons.py           # ✅ WeightNormConv1d, utilities
├── weights/
│   ├── __init__.py          # ✅ Exports weight utilities
│   ├── convert.py           # ✅ PyTorch → SafeTensors conversion
│   └── loader.py            # ✅ Load weights into MLX models
├── pipeline.py              # ✅ End-to-end inference pipeline
└── cli.py                   # ✅ Command-line interface

tests/
├── test_audio.py            # ✅ 12 tests
├── test_f0.py               # ✅ 8 tests
├── test_weight_norm.py      # ✅ 6 tests
├── test_nsf.py              # ✅ 9 tests
├── test_resblock.py         # ✅ 4 tests
├── test_generator.py        # ✅ 5 tests
└── test_synthesizer.py      # ✅ 5 tests

scripts/
└── upload_weights.py        # ✅ Upload weights to HuggingFace

weights/                     # Converted MLX weights for HuggingFace
├── README.md                # HuggingFace model card
└── v2/
    ├── config.json          # Model configurations
    ├── f0G32k.safetensors   # V2 32kHz with F0
    ├── f0G40k.safetensors   # V2 40kHz with F0
    └── f0G48k.safetensors   # V2 48kHz with F0

vendor/
├── mlx-contentvec/          # Source for mlx-contentvec PyPI package
├── mlx-rmvpe/               # Source for mlx-rmvpe PyPI package
├── weights/
│   └── f0G48k.pth           # Reference PyTorch weights (72MB)
└── Retrieval-based-Voice-Conversion-WebUI/  # Reference implementation
```

## Implementation Order

### Phase 1: Minimal Working Pipeline ✅ COMPLETE
1. ✅ **Audio I/O** - Load/save with FFmpeg, basic processing
2. ✅ **F0 Extraction** - Integrate pyworld (Harvest method)
3. ✅ **GeneratorNSF** - Core decoder (most complex, start early)
4. ✅ **TextEncoder** - Relatively simple transformer
5. ✅ **ResidualCouplingBlock** - Flow layers
6. ✅ **SynthesizerTrnMs768NSFsid** - Combine all components
7. ✅ **Weight Conversion** - PyTorch → SafeTensors
8. ✅ **Pipeline Integration** - Wire everything together
9. ✅ **CLI** - Basic command-line interface

### Phase 2: Quality & Features
10. ✅ **FAISS Index Blending** - Improves conversion quality via feature retrieval
11. ✅ **RMVPE F0 Extraction** - Better pitch detection (now in `mlx-rmvpe` PyPI package)
12. ✅ **Multiple Sample Rates** - Support 32k/40k/48k models with validation
13. ✅ **V1 Model Support** - Auto-detect and support 256-dim HuBERT models

### Phase 3: Optimization
14. **Streaming/Chunked Processing** - For long audio
15. **Memory Optimization** - Efficient buffer management
16. **Performance Tuning** - Profile and optimize hot paths

## Key Technical Challenges

### 1. NSF Source Generation ✅ SOLVED
The sine wave generation with harmonics is mathematically sensitive:
```python
# Generate fundamental + harmonics
for i in range(num_harmonics):
    phase = cumsum(2 * pi * f0 * (i+1) / sr)
    harmonic = sin(phase)
```
**Solution**: Implemented in `mlx_rvc/models/nsf.py` with careful phase continuity handling using `mx.remainder` for modulo operations and masking for voiced/unvoiced transitions.

### 2. Transposed Convolutions ✅ SOLVED
MLX has `conv_transpose1d` but need to verify behavior matches PyTorch exactly, especially with:
- Non-unit stride
- Output padding
- Weight normalization

**Solution**: `WeightNormConvTranspose1d` in `mlx_rvc/models/commons.py` validates against PyTorch reference with 6 passing tests.

### 3. Weight Normalization ✅ SOLVED
Used extensively in generator. Need to implement:
```python
# w = g * (v / ||v||)
def weight_norm_forward(v, g):
    norm = sqrt(sum(v ** 2, axis=...))
    return g * v / norm
```
**Solution**: `WeightNormConv1d` and `WeightNormConvTranspose1d` classes in `mlx_rvc/models/commons.py`.

### 4. Flow Reversibility ✅ SOLVED
The coupling layers must work correctly in reverse mode for inference.

**Solution**: `ResidualCouplingBlock` in `mlx_rvc/models/flow.py` with proper forward/reverse modes. Uses `Flip` operation with channel reversal via `x[:, ::-1, :]`.

### 5. MLX API Differences (Discovered During Implementation)
Several MLX API differences from PyTorch required workarounds:
- **No `mx.flip`**: Used slice notation `x[:, ::-1, :]` instead
- **No `mx.mod`**: Use `mx.remainder` for modulo operations
- **No `.at[]` indexing**: Use masking for conditional updates
- **Conv1d expects channels-last**: Added transpose wrappers for channels-first compatibility

## Dependencies

**Required**:
- `mlx` - Apple ML framework
- `mlx-contentvec` - Feature extraction (PyPI)
- `mlx-rmvpe` - RMVPE F0 extraction (PyPI)
- `numpy` - Numerical operations
- `pyworld` - F0 extraction (Harvest)
- `safetensors` - Weight storage
- `huggingface-hub` - Model weight downloads
- `ffmpeg` - Audio codec (external)

**Optional**:
- `faiss-cpu` - Index blending
- `librosa` - Additional audio utilities (used by mlx-rmvpe)
- `soundfile` - Audio I/O alternative

## CLI Interface Design

```bash
# Basic usage
mlx-rvc convert input.wav output.wav --model voice.pth

# With RMVPE for singing (auto-downloads weights)
mlx-rvc convert input.wav output.wav --model voice.pth --f0-method rmvpe

# With pitch shift
mlx-rvc convert input.wav output.wav --model voice.pth --pitch 5

# With FAISS index blending for improved voice similarity
mlx-rvc convert input.wav output.wav --model voice.pth --index voice.index

# Adjust index blending rate (0 = original only, 1 = index only)
mlx-rvc convert input.wav output.wav --model voice.pth --index voice.index --index-rate 0.75

# Show model info
mlx-rvc info voice.pth
```

## Model Compatibility

| Model Type | Status | Notes |
|------------|--------|-------|
| V2 + F0 (768-dim) | ✅ Supported | Most common, best quality |
| V1 + F0 (256-dim) | ✅ Supported | Auto-detected, uses first 256 dims of ContentVec |
| V2 no-F0 | Future | Simpler, no pitch control |
| V1 no-F0 | Future | Legacy support |

## Testing Strategy

1. **Unit Tests**: Each component against PyTorch reference
2. **Integration Tests**: Full pipeline with known inputs
3. **Numerical Validation**: Compare outputs to original RVC
4. **Audio Quality**: Subjective listening tests

## Reference Weights

Reference PyTorch weights for development and testing are stored in `vendor/weights/`.

### Downloading Weights

From HuggingFace: https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/

```bash
# V2 Generator with F0 (48kHz) - primary development target
curl -L "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth" \
     -o vendor/weights/f0G48k.pth
```

### Available Models

| File | Model | Description |
|------|-------|-------------|
| `f0G48k.pth` | SynthesizerTrnMs768NSFsid | V2 generator with F0, 48kHz (~72MB) |
| `f0G40k.pth` | SynthesizerTrnMs768NSFsid | V2 generator with F0, 40kHz |
| `f0G32k.pth` | SynthesizerTrnMs768NSFsid | V2 generator with F0, 32kHz |

**Naming convention:**
- `f0G` = Generator with F0 (pitch)
- `G` = Generator without F0
- `D` = Discriminator (training only)
- `32k/40k/48k` = sample rate

### Checkpoint Structure

The pretrained checkpoints use training format:
```python
{
    "model": {...},        # state_dict (560 keys for full model)
    "iteration": 392,      # training iteration
    "learning_rate": 0.0001
}
```

Key weight prefixes:
- `enc_p.*` - TextEncoder (113 params)
- `dec.*` - GeneratorNSF (243 params)
- `enc_q.*` - PosteriorEncoder (103 params, training only)
- `flow.*` - ResidualCouplingBlock (100 params)
- `emb_g.*` - Speaker embedding (1 param)

### GeneratorNSF Weight Structure (48kHz)

```
dec.m_source.l_linear     - Linear(1→1) harmonic combiner
dec.noise_convs.[0-3]     - F0 source injection convs
dec.conv_pre              - Conv1d(192→512, k=7)
dec.ups.[0-3]             - ConvTranspose1d with weight norm
dec.resblocks.[0-11]      - ResBlock1 with weight norm
dec.conv_post             - Conv1d(32→1, k=7)
dec.cond                  - Conv1d(256→512, k=1) speaker conditioning
```

Weight normalization uses `weight_g` and `weight_v` decomposition:
```python
# Effective weight: w = g * (v / ||v||)
weight_g: torch.Size([out_ch, 1, 1])  # magnitude
weight_v: torch.Size([out_ch, in_ch, kernel])  # direction
```

## Next Steps

### Phase 1 Complete!

1. ✅ **Weight Conversion Script** (`mlx_rvc/weights/convert.py`)
2. ✅ **Pipeline Integration** (`mlx_rvc/pipeline.py`)
3. ✅ **CLI Interface** (`mlx_rvc/cli.py`)

```bash
# Convert voice
mlx-rvc convert input.wav output.wav --model voice.pth

# With pitch shift
mlx-rvc convert input.wav output.wav --model voice.pth --pitch 5

# Show model info
mlx-rvc info voice.pth
```

### HuggingFace Weights

Pre-converted MLX weights available at: `lexandstuff/mlx-rvc-weights`

```python
from huggingface_hub import hf_hub_download

# Download 40kHz model
weights_path = hf_hub_download("lexandstuff/mlx-rvc-weights", "v2/f0G40k.safetensors")
```

### Weight Loading Strategy

The synthesizer has these weight groups to load:
```python
# TextEncoder (enc_p)
enc_p.emb_phone.weight       # Linear(768, 192)
enc_p.emb_pitch.weight       # Embedding(256, 192)
enc_p.encoder.*              # 6 transformer layers
enc_p.proj.*                 # Conv1d(192, 384)

# Flow (flow)
flow.flows.{0,2,4,6}.*       # 4 ResidualCouplingLayers
flow.flows.{1,3,5,7}         # 4 Flip (no weights)

# Generator (dec)
dec.m_source.*               # SourceModuleHnNSF
dec.conv_pre.*               # Conv1d(192, 512)
dec.ups.{0-3}.*              # 4 ConvTranspose1d (weight norm)
dec.noise_convs.{0-3}.*      # 4 Conv1d
dec.resblocks.{0-11}.*       # 12 ResBlock1
dec.conv_post.*              # Conv1d(32, 1)
dec.cond.*                   # Conv1d(256, 512)

# Speaker Embedding
emb_g.weight                 # Embedding(109, 256)
```

### Testing with Real Weights

After weight conversion:
```python
from mlx_rvc.models import SynthesizerTrnMs768NSFsid
from safetensors import safe_open

model = SynthesizerTrnMs768NSFsid(**config)
with safe_open("model.safetensors", framework="mlx") as f:
    model.load_weights(f)

# Run inference
audio, mask, _ = model.infer(phone, phone_lengths, pitch, f0, sid)
```

## References

- [RVC WebUI Repository](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [VITS Paper](https://arxiv.org/abs/2106.06103)
- [HuBERT Paper](https://arxiv.org/abs/2106.07447)
- [ContentVec Paper](https://arxiv.org/abs/2204.09224)
- [NSF Paper](https://arxiv.org/abs/1904.12088)

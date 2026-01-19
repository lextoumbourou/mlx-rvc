# RVC-MLX Implementation Notes

## Overview

This document outlines the plan for building an MLX-native implementation of RVC (Retrieval-based Voice Conversion) for Apple Silicon. The initial focus is on **inference-only** functionality via a command-line interface, with training and GUI to follow.

## What We Already Have

### mlx-contentvec (Complete)

Located in `vendor/mlx-contentvec`, this provides the ContentVec/HuBERT feature extractor:

- **Input**: Raw audio waveform @ 16kHz, shape `(batch, samples)`
- **Output**: Semantic features @ ~50fps, shape `(batch, frames, 768)`
- **Purpose**: Extracts speaker-agnostic phonetic content from speech
- **Status**: Production-ready, numerically validated against PyTorch reference

```python
from mlx_contentvec import ContentvecModel

model = ContentvecModel(encoder_layers_1=0)  # No speaker conditioning
model.load_weights("contentvec_base.safetensors")
result = model(audio_tensor)
features = result["x"]  # (batch, frames, 768)
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

## Proposed Project Structure

```
rvc_mlx/
├── __init__.py
├── cli.py                    # Command-line interface
├── config.py                 # Configuration management
├── audio/
│   ├── __init__.py
│   ├── io.py                 # Load/save audio files
│   └── processing.py         # Resample, normalize, RMS
├── f0/
│   ├── __init__.py
│   ├── harvest.py           # pyworld wrapper
│   ├── rmvpe.py             # RMVPE model (future)
│   └── processing.py        # Pitch shift, mel conversion
├── index/
│   ├── __init__.py
│   └── faiss_blend.py       # FAISS index loading and blending
├── models/
│   ├── __init__.py
│   ├── synthesizer.py       # SynthesizerTrnMs768NSFsid
│   ├── encoder.py           # TextEncoder
│   ├── flow.py              # ResidualCouplingBlock
│   ├── generator.py         # GeneratorNSF
│   ├── nsf.py               # Neural source filter components
│   └── commons.py           # Shared utilities
├── pipeline.py              # Main inference pipeline
└── weights/
    └── convert.py           # PyTorch → SafeTensors conversion

scripts/
├── convert_weights.py       # CLI for weight conversion
└── download_models.py       # Download pre-trained models
```

## Implementation Order

### Phase 1: Minimal Working Pipeline
1. **Audio I/O** - Load/save with FFmpeg, basic processing
2. **F0 Extraction** - Integrate pyworld (Harvest method)
3. **GeneratorNSF** - Core decoder (most complex, start early)
4. **TextEncoder** - Relatively simple transformer
5. **ResidualCouplingBlock** - Flow layers
6. **SynthesizerTrnMs768NSFsid** - Combine all components
7. **Weight Conversion** - PyTorch → SafeTensors
8. **Pipeline Integration** - Wire everything together
9. **CLI** - Basic command-line interface

### Phase 2: Quality & Features
10. **FAISS Index Blending** - Improve conversion quality
11. **RMVPE F0 Extraction** - Better pitch detection
12. **Multiple Sample Rates** - Support 32k/40k/48k models
13. **V1 Model Support** - Add 256-dim variant

### Phase 3: Optimization
14. **Streaming/Chunked Processing** - For long audio
15. **Memory Optimization** - Efficient buffer management
16. **Performance Tuning** - Profile and optimize hot paths

## Key Technical Challenges

### 1. NSF Source Generation
The sine wave generation with harmonics is mathematically sensitive:
```python
# Generate fundamental + harmonics
for i in range(num_harmonics):
    phase = cumsum(2 * pi * f0 * (i+1) / sr)
    harmonic = sin(phase)
```
Need careful handling of phase continuity and voiced/unvoiced transitions.

### 2. Transposed Convolutions
MLX has `conv_transpose1d` but need to verify behavior matches PyTorch exactly, especially with:
- Non-unit stride
- Output padding
- Weight normalization

### 3. Weight Normalization
Used extensively in generator. Need to implement:
```python
# w = g * (v / ||v||)
def weight_norm_forward(v, g):
    norm = sqrt(sum(v ** 2, axis=...))
    return g * v / norm
```

### 4. Flow Reversibility
The coupling layers must work correctly in reverse mode for inference.

## Dependencies

**Required**:
- `mlx` - Apple ML framework
- `mlx-contentvec` - Feature extraction (vendor)
- `numpy` - Numerical operations
- `pyworld` - F0 extraction (Harvest)
- `safetensors` - Weight storage
- `ffmpeg` - Audio codec (external)

**Optional**:
- `faiss-cpu` - Index blending
- `librosa` - Additional audio utilities
- `soundfile` - Audio I/O alternative

## CLI Interface Design

```bash
# Basic usage
rvc-mlx convert input.wav output.wav --model voice.pth

# Full options
rvc-mlx convert input.wav output.wav \
    --model voice.pth \
    --index voice.index \
    --f0-method harvest \
    --transpose 0 \
    --index-rate 0.66 \
    --protect 0.33 \
    --rms-mix 1.0

# List available models
rvc-mlx list-models

# Download pre-trained models
rvc-mlx download rmvpe
```

## Model Compatibility

| Model Type | Status | Notes |
|------------|--------|-------|
| V2 + F0 (768-dim) | Primary Target | Most common, best quality |
| V2 no-F0 | Future | Simpler, no pitch control |
| V1 + F0 (256-dim) | Future | Legacy support |
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

## References

- [RVC WebUI Repository](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [VITS Paper](https://arxiv.org/abs/2106.06103)
- [HuBERT Paper](https://arxiv.org/abs/2106.07447)
- [ContentVec Paper](https://arxiv.org/abs/2204.09224)
- [NSF Paper](https://arxiv.org/abs/1904.12088)

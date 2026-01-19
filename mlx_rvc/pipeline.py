"""RVC-MLX inference pipeline."""

from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from mlx_contentvec import ContentvecModel

from .audio.io import load_audio, save_audio
from .audio.processing import normalize_audio
from .f0.harvest import extract_f0_harvest
from .f0.processing import f0_to_coarse, shift_f0
from .models import SynthesizerTrnMs768NSFsid
from .weights import load_checkpoint, load_model


class RVCPipeline:
    """
    End-to-end RVC voice conversion pipeline.

    Example:
        pipeline = RVCPipeline.from_pretrained("model.pth")
        pipeline.convert("input.wav", "output.wav", speaker_id=0)
    """

    def __init__(
        self,
        synthesizer: SynthesizerTrnMs768NSFsid,
        contentvec: ContentvecModel,
        sample_rate: int = 40000,
        f0_method: str = "harvest",
        in_channels: int = 768,
    ):
        self.synthesizer = synthesizer
        self.contentvec = contentvec
        self.sample_rate = sample_rate
        self.f0_method = f0_method
        self.in_channels = in_channels  # 768 for v2, 256 for v1

        # ContentVec operates at 16kHz with ~50fps output
        self.contentvec_sr = 16000
        self.contentvec_hop = 320  # 16000 / 50 = 320 samples per frame

        # RMVPE model (lazy loaded from HuggingFace)
        self._rmvpe_model = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        contentvec_path: Optional[str] = None,
    ) -> "RVCPipeline":
        """
        Load pipeline from pretrained weights.

        Args:
            model_path: Path to RVC model (.pth or .safetensors)
            contentvec_path: Optional path to ContentVec weights
                            (downloads from HuggingFace if not provided)

        Returns:
            Initialized RVCPipeline
        """
        # Load RVC model
        weights, config = load_checkpoint(model_path)
        synthesizer = SynthesizerTrnMs768NSFsid(**config)
        load_model(synthesizer, weights)

        version = config.get('version', 'v2' if config.get('in_channels', 768) == 768 else 'v1')
        in_channels = config.get('in_channels', 768)
        print(f"Loaded RVC model: {version} @ {config.get('sr', 'unknown')}Hz (in_channels={in_channels})")

        # Load ContentVec (auto-downloads weights if needed)
        if contentvec_path is None:
            contentvec = ContentvecModel.from_pretrained()
        else:
            contentvec = ContentvecModel.from_pretrained(weights_path=contentvec_path)
        print("Loaded ContentVec model")

        return cls(
            synthesizer=synthesizer,
            contentvec=contentvec,
            sample_rate=config.get("sr", 40000),
            in_channels=in_channels,
        )

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        speaker_id: int = 0,
        f0_shift: int = 0,
        f0_method: Optional[str] = None,
        index_path: Optional[Union[str, Path]] = None,
        index_rate: float = 0.5,
    ) -> np.ndarray:
        """
        Convert voice in audio file.

        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            speaker_id: Speaker ID for voice conversion
            f0_shift: Pitch shift in semitones (-12 to +12)
            f0_method: F0 extraction method (default: "harvest")
            index_path: Path to FAISS index file for feature blending (optional)
            index_rate: Index blending rate (0 = original only, 1 = index only, default: 0.5)

        Returns:
            Converted audio as numpy array
        """
        f0_method = f0_method or self.f0_method

        # Load and preprocess audio
        print(f"Loading audio: {input_path}")
        audio_16k = load_audio(input_path, sr=self.contentvec_sr)
        audio_16k = normalize_audio(audio_16k)

        # Extract features
        print("Extracting ContentVec features...")
        phone = self._extract_contentvec(audio_16k)

        # Apply index blending if provided
        if index_path is not None and index_rate > 0:
            phone = self._blend_with_index(phone, index_path, index_rate)

        print(f"Extracting F0 ({f0_method})...")
        f0, f0_coarse = self._extract_f0(audio_16k, f0_shift, f0_method)

        # Align lengths (F0 and phone features must match)
        min_len = min(phone.shape[1], f0.shape[0])
        phone = phone[:, :min_len, :]
        f0 = f0[:min_len]
        f0_coarse = f0_coarse[:min_len]

        # Convert to MLX arrays
        phone = mx.array(phone)
        phone_lengths = mx.array([min_len])
        pitch = mx.array(f0_coarse[None, :].astype(np.int32))
        nsff0 = mx.array(f0[None, :].astype(np.float32))
        sid = mx.array([speaker_id])

        # Run inference
        print("Running voice conversion...")
        audio_out, _, _ = self.synthesizer.infer(
            phone, phone_lengths, pitch, nsff0, sid
        )

        # Convert to numpy and squeeze
        audio_out = np.array(audio_out).squeeze()

        # Save output
        print(f"Saving output: {output_path}")
        save_audio(output_path, audio_out, sr=self.sample_rate)

        return audio_out

    def _blend_with_index(
        self,
        phone: np.ndarray,
        index_path: Union[str, Path],
        index_rate: float,
    ) -> np.ndarray:
        """
        Blend ContentVec features with FAISS index for improved voice similarity.

        Args:
            phone: ContentVec features, shape (batch, frames, 768)
            index_path: Path to FAISS index file
            index_rate: Blend ratio (0 = original, 1 = index features)

        Returns:
            Blended features, same shape as input
        """
        from .index import load_index

        index_path = Path(index_path)
        if not index_path.exists():
            print(f"Warning: Index file not found: {index_path}, skipping blending")
            return phone

        print(f"Loading FAISS index: {index_path}")
        index = load_index(index_path)

        if index is None:
            print("Warning: FAISS not available, skipping index blending")
            print("Install with: uv pip install faiss-cpu")
            return phone

        print(f"Blending features (index_rate={index_rate})...")
        # phone is (batch, frames, 768), blend expects (frames, 768) or (batch, frames, 768)
        blended = index.blend(phone, index_rate=index_rate, k=8)

        return blended

    def _extract_contentvec(self, audio: np.ndarray) -> np.ndarray:
        """Extract ContentVec features from audio."""
        # Ensure audio is 2D: (batch, samples)
        if audio.ndim == 1:
            audio = audio[None, :]

        # Convert to MLX and run
        audio_mx = mx.array(audio.astype(np.float32))
        result = self.contentvec(audio_mx)

        # Get features: (batch, frames, 768)
        features = np.array(result["x"])

        # For v1 models, slice to first 256 dimensions
        if self.in_channels == 256:
            features = features[:, :, :256]

        return features

    def _extract_f0(
        self,
        audio: np.ndarray,
        f0_shift: int = 0,
        f0_method: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 and convert to coarse pitch.

        Args:
            audio: Audio at 16kHz
            f0_shift: Pitch shift in semitones
            f0_method: F0 extraction method ("harvest" or "rmvpe")

        Returns:
            Tuple of (f0_hz, f0_coarse)
        """
        f0_method = f0_method or self.f0_method

        if f0_method == "rmvpe":
            f0 = self._extract_f0_rmvpe(audio)
        else:
            # Default: Harvest
            f0 = extract_f0_harvest(
                audio,
                sr=self.contentvec_sr,
                f0_min=50.0,
                f0_max=1100.0,
                frame_period=10.0,  # 10ms = 100fps, but harvest may adjust
            )

        # Apply pitch shift
        if f0_shift != 0:
            f0 = shift_f0(f0, f0_shift)

        # Convert to coarse (quantized 1-255)
        f0_coarse = f0_to_coarse(f0)

        return f0, f0_coarse

    def _extract_f0_rmvpe(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using RMVPE model (auto-downloads from HuggingFace)."""
        # Lazy load RMVPE model
        if self._rmvpe_model is None:
            from mlx_rmvpe import RMVPE

            print("Loading RMVPE model...")
            self._rmvpe_model = RMVPE.from_pretrained()

        # Run inference
        f0 = self._rmvpe_model.infer_from_audio(audio, threshold=0.03)

        return f0


def convert_voice(
    input_path: str,
    output_path: str,
    model_path: str,
    speaker_id: int = 0,
    f0_shift: int = 0,
) -> np.ndarray:
    """
    Simple function to convert voice.

    Args:
        input_path: Input audio file
        output_path: Output audio file
        model_path: Path to RVC model
        speaker_id: Speaker ID (default 0)
        f0_shift: Pitch shift in semitones

    Returns:
        Converted audio
    """
    pipeline = RVCPipeline.from_pretrained(model_path)
    return pipeline.convert(
        input_path,
        output_path,
        speaker_id=speaker_id,
        f0_shift=f0_shift,
    )

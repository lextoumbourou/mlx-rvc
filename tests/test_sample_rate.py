"""Tests for sample rate handling and validation."""

import math
import pytest
import numpy as np

from rvc_mlx.weights import validate_config, SAMPLE_RATE_CONFIGS
from rvc_mlx.models.generator import GeneratorNSF


class TestSampleRateConfigs:
    """Test SAMPLE_RATE_CONFIGS consistency."""

    def test_all_configs_have_required_keys(self):
        """All sample rate configs should have required keys."""
        required_keys = {"upsample_rates", "upsample_kernel_sizes", "upp"}
        for sr, cfg in SAMPLE_RATE_CONFIGS.items():
            assert required_keys <= set(cfg.keys()), f"Missing keys for {sr}Hz"

    def test_upp_matches_upsample_rates(self):
        """upp should be product of upsample_rates."""
        for sr, cfg in SAMPLE_RATE_CONFIGS.items():
            expected_upp = math.prod(cfg["upsample_rates"])
            assert cfg["upp"] == expected_upp, (
                f"{sr}Hz: upp={cfg['upp']} != product({cfg['upsample_rates']})={expected_upp}"
            )

    def test_upp_matches_sample_rate(self):
        """upp should equal sr/100 for ContentVec frame rate alignment."""
        for sr, cfg in SAMPLE_RATE_CONFIGS.items():
            expected_upp = sr // 100
            assert cfg["upp"] == expected_upp, (
                f"{sr}Hz: upp={cfg['upp']} != sr/100={expected_upp}"
            )

    def test_supported_sample_rates(self):
        """Should support 32k, 40k, and 48k sample rates."""
        assert 32000 in SAMPLE_RATE_CONFIGS
        assert 40000 in SAMPLE_RATE_CONFIGS
        assert 48000 in SAMPLE_RATE_CONFIGS

    def test_kernel_sizes_match_upsample_rates(self):
        """Kernel sizes should be 2x upsample rate (standard HiFi-GAN)."""
        for sr, cfg in SAMPLE_RATE_CONFIGS.items():
            for rate, kernel in zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"]):
                assert kernel == rate * 2, (
                    f"{sr}Hz: kernel {kernel} != rate {rate} * 2"
                )


class TestValidateConfig:
    """Test validate_config function."""

    def test_valid_32k_config(self):
        """Valid 32kHz config should pass."""
        config = {
            "sr": 32000,
            "upsample_rates": [10, 8, 2, 2],
        }
        validate_config(config)  # Should not raise

    def test_valid_40k_config(self):
        """Valid 40kHz config should pass."""
        config = {
            "sr": 40000,
            "upsample_rates": [10, 10, 2, 2],
        }
        validate_config(config)  # Should not raise

    def test_valid_48k_config(self):
        """Valid 48kHz config should pass."""
        config = {
            "sr": 48000,
            "upsample_rates": [12, 10, 2, 2],
        }
        validate_config(config)  # Should not raise

    def test_missing_sr_raises(self):
        """Missing sample rate should raise ValueError."""
        config = {"upsample_rates": [10, 10, 2, 2]}
        with pytest.raises(ValueError, match="missing required 'sr'"):
            validate_config(config)

    def test_unsupported_sr_raises(self):
        """Unsupported sample rate should raise ValueError."""
        config = {
            "sr": 44100,
            "upsample_rates": [10, 10, 2, 2],
        }
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            validate_config(config)

    def test_missing_upsample_rates_raises(self):
        """Missing upsample_rates should raise ValueError."""
        config = {"sr": 40000}
        with pytest.raises(ValueError, match="missing 'upsample_rates'"):
            validate_config(config)

    def test_mismatched_upsample_factor_raises(self):
        """Mismatched upsample factor should raise ValueError."""
        # 32kHz upsample rates with 40kHz sample rate
        config = {
            "sr": 40000,
            "upsample_rates": [10, 8, 2, 2],  # 320x, but 40kHz needs 400x
        }
        with pytest.raises(ValueError, match="Upsample factor mismatch"):
            validate_config(config)

    def test_wrong_rates_for_48k_raises(self):
        """Wrong upsample rates for 48kHz should raise."""
        config = {
            "sr": 48000,
            "upsample_rates": [10, 10, 2, 2],  # 400x, but 48kHz needs 480x
        }
        with pytest.raises(ValueError, match="Upsample factor mismatch"):
            validate_config(config)


class TestGeneratorNSFValidation:
    """Test GeneratorNSF sample rate validation."""

    def test_valid_32k_generator(self):
        """32kHz generator with correct upsample rates should work."""
        gen = GeneratorNSF(
            initial_channel=192,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[20, 16, 4, 4],
            gin_channels=256,
            sr=32000,
        )
        assert gen.sr == 32000
        assert gen.upp == 320

    def test_valid_40k_generator(self):
        """40kHz generator with correct upsample rates should work."""
        gen = GeneratorNSF(
            initial_channel=192,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 10, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[20, 20, 4, 4],
            gin_channels=256,
            sr=40000,
        )
        assert gen.sr == 40000
        assert gen.upp == 400

    def test_valid_48k_generator(self):
        """48kHz generator with correct upsample rates should work."""
        gen = GeneratorNSF(
            initial_channel=192,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[12, 10, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[24, 20, 4, 4],
            gin_channels=256,
            sr=48000,
        )
        assert gen.sr == 48000
        assert gen.upp == 480

    def test_mismatched_upsample_raises(self):
        """Generator with mismatched upsample factor should raise."""
        with pytest.raises(ValueError, match="Upsample factor"):
            GeneratorNSF(
                initial_channel=192,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 8, 2, 2],  # 320x
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 16, 4, 4],
                gin_channels=256,
                sr=40000,  # Expects 400x
            )


class TestSampleRateConsistency:
    """Test sample rate consistency across the pipeline."""

    def test_contentvec_frame_rate_assumption(self):
        """Verify the ContentVec frame rate assumption (100fps)."""
        # ContentVec operates at 16kHz with 320-sample hop
        # This gives 16000 / 320 = 50 fps... but RVC uses 160 hop internally
        # Actually, the comment in generator says sr/100, let's verify
        for sr, cfg in SAMPLE_RATE_CONFIGS.items():
            # Expected frames per second at this sample rate
            # Audio samples per ContentVec frame = sr / contentvec_fps
            # With upp upsampling from ContentVec frames to audio samples:
            # audio_samples = contentvec_frames * upp
            # So: upp = sr / contentvec_fps
            # If upp = sr/100, then contentvec_fps = 100
            contentvec_fps = sr / cfg["upp"]
            assert contentvec_fps == 100, f"{sr}Hz implies {contentvec_fps}fps ContentVec"

    def test_all_sample_rates_produce_correct_audio_length(self):
        """Verify audio length calculation for all sample rates."""
        for sr, cfg in SAMPLE_RATE_CONFIGS.items():
            # 1 second of audio at this sample rate
            num_samples = sr

            # Number of ContentVec frames (at 100fps)
            num_frames = 100

            # Generated audio length
            generated_length = num_frames * cfg["upp"]

            assert generated_length == num_samples, (
                f"{sr}Hz: {num_frames} frames * {cfg['upp']} upp = {generated_length}, "
                f"expected {num_samples}"
            )

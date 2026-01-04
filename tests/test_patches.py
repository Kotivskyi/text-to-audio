"""Tests for text_to_audio.patches module."""

import pytest
from text_to_audio.patches import DummyWatermarker


class TestDummyWatermarker:
    """Tests for DummyWatermarker class."""

    def test_apply_returns_audio_unchanged(self):
        """apply() should return audio unchanged."""
        watermarker = DummyWatermarker()
        audio = [1, 2, 3, 4, 5]
        result = watermarker.apply(audio, sr=22050)
        assert result == audio

    def test_apply_watermark_returns_signal_unchanged(self):
        """apply_watermark() should return signal unchanged."""
        watermarker = DummyWatermarker()
        signal = [0.1, 0.2, 0.3]
        result = watermarker.apply_watermark(signal, sample_rate=22050)
        assert result == signal

    def test_get_watermark_returns_signal_unchanged(self):
        """get_watermark() should return signal unchanged."""
        watermarker = DummyWatermarker()
        signal = [0.5, 0.6, 0.7]
        result = watermarker.get_watermark(signal, sample_rate=22050)
        assert result == signal

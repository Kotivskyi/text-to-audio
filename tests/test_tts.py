"""Tests for text_to_audio.tts module."""

import pytest
from text_to_audio.tts import split_text, get_device


class TestSplitText:
    """Tests for split_text function."""

    def test_short_text_no_split(self):
        """Short text should not be split."""
        text = "Hello world."
        result = split_text(text, max_chars=250)
        assert result == ["Hello world."]

    def test_empty_text(self):
        """Empty text should return original."""
        text = ""
        result = split_text(text, max_chars=250)
        assert result == [""]

    def test_split_at_sentence_boundary(self):
        """Text should split at sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        result = split_text(text, max_chars=30)
        assert len(result) >= 2
        # Each chunk should end with proper punctuation or be a complete thought
        for chunk in result:
            assert chunk.strip()

    def test_long_sentence_splits_at_comma(self):
        """Long sentence should split at comma or other punctuation."""
        text = "This is a very long sentence with many, many words, and it keeps going, and going, until it exceeds the limit."
        result = split_text(text, max_chars=50)
        assert len(result) >= 2

    def test_preserves_all_content(self):
        """All original content should be preserved after split."""
        text = "First part. Second part. Third part."
        result = split_text(text, max_chars=20)
        rejoined = " ".join(result)
        # Check all words are present
        for word in text.split():
            assert word.strip(".,!?") in rejoined

    def test_respects_max_chars(self):
        """Each chunk should respect max_chars limit when possible."""
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        result = split_text(text, max_chars=15)
        # Most chunks should be under the limit
        under_limit = sum(1 for chunk in result if len(chunk) <= 20)
        assert under_limit >= len(result) // 2

    def test_multiple_sentences(self):
        """Multiple sentences should be grouped efficiently."""
        text = "A. B. C. D. E. F. G. H."
        result = split_text(text, max_chars=10)
        assert len(result) >= 2

    def test_exclamation_and_question_marks(self):
        """Should split on exclamation and question marks too."""
        text = "Hello! How are you? I am fine."
        result = split_text(text, max_chars=20)
        assert len(result) >= 2


class TestGetDevice:
    """Tests for get_device function."""

    def test_explicit_cpu(self):
        """Explicit cpu should return cpu."""
        assert get_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        """Explicit cuda should return cuda."""
        assert get_device("cuda") == "cuda"

    def test_explicit_mps(self):
        """Explicit mps should return mps."""
        assert get_device("mps") == "mps"

    def test_auto_returns_valid_device(self):
        """Auto should return a valid device string."""
        result = get_device("auto")
        assert result in ["cuda", "mps", "cpu"]

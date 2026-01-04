"""Tests for text_to_audio.cli module."""

import pytest
from text_to_audio.cli import ProgressTracker


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_format_time_seconds(self):
        """Should format seconds correctly."""
        assert ProgressTracker.format_time(30) == "30s"
        assert ProgressTracker.format_time(5) == "5s"
        assert ProgressTracker.format_time(59) == "59s"

    def test_format_time_minutes(self):
        """Should format minutes correctly."""
        assert ProgressTracker.format_time(60) == "1m 0s"
        assert ProgressTracker.format_time(90) == "1m 30s"
        assert ProgressTracker.format_time(125) == "2m 5s"

    def test_format_time_zero(self):
        """Should handle zero."""
        assert ProgressTracker.format_time(0) == "0s"

    def test_tracker_initializes(self):
        """Tracker should initialize with None values."""
        tracker = ProgressTracker()
        assert tracker.start_time is None
        assert tracker.total_estimate is None

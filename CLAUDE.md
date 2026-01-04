# Claude Code Instructions

## Project Overview

Text-to-Audio CLI tool using Chatterbox TTS with automatic long text handling.

## Development Commands

Always use `uv` for running commands:

```bash
# Install dependencies
uv sync

# Run the CLI
uv run text-to-audio "Hello world" -o output.wav

# Run with options
uv run text-to-audio "Text" -r voice.wav -m standard -e 0.3

# Run from file
uv run text-to-audio -i input.txt -o output.wav
```

## Testing

**IMPORTANT: Always run tests before pushing changes.**

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_tts.py -v
```

## Project Structure

```
text_to_audio/
├── __init__.py    # Package exports
├── cli.py         # CLI interface with argparse
├── patches.py     # Fixes for perth/torch compatibility
└── tts.py         # Main TTS wrapper with text splitting
tests/
├── test_cli.py    # ProgressTracker tests
├── test_patches.py # DummyWatermarker tests
└── test_tts.py    # split_text, get_device tests
```

## Key Functions

- `split_text(text, max_chars)` - Splits long text at sentence boundaries
- `get_device(device_arg)` - Auto-detects best device (cuda/mps/cpu)
- `TextToAudio` - Main class for TTS generation
- `apply_patches(device)` - Applies perth/torch.load fixes

## Before Pushing

1. Run tests: `uv run pytest tests/ -v`
2. Ensure all tests pass
3. Commit with descriptive message

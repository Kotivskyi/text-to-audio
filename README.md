# Text-to-Audio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Text-to-speech generator using [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) with automatic long text handling. Runs locally on CPU, NVIDIA GPU (CUDA), or Apple Silicon (M1-M4).

## Installation

```bash
uv sync
```

## CLI Usage

```bash
# Basic usage
uv run text-to-audio "Hello world"

# Specify output file
uv run text-to-audio "Hello world" -o hello.wav

# Voice cloning with reference audio
uv run text-to-audio "Hello world" -r voice.wav -o output.wav

# Generate from text file (handles long text automatically)
uv run text-to-audio -i input.txt -o output.wav

# Use different models
uv run text-to-audio "Hello" -m standard    # Higher quality (default)
uv run text-to-audio "Hello" -m turbo       # Faster 
uv run text-to-audio "Bonjour" -m multilingual -l fr

# Adjust expressiveness
uv run text-to-audio "Excited text!" -e 0.8 --cfg 0.5

# With emotion tags
uv run text-to-audio "That's funny [laugh] really!"

# Quiet mode (no progress output)
uv run text-to-audio "Hello" -q -o output.wav
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `text` | Text to convert (positional) | - |
| `-i, --input` | Input text file | - |
| `-o, --output` | Output audio file | `output.wav` |
| `-r, --reference` | Reference audio for voice cloning | - |
| `-m, --model` | Model: `turbo`, `standard`, `multilingual` | `turbo` |
| `-l, --language` | Language code (for multilingual) | `en` |
| `-d, --device` | Device: `auto`, `cuda`, `mps`, `cpu` | `auto` |
| `-e, --exaggeration` | Expressiveness (0.0-1.0) | `0.3` |
| `--cfg` | CFG weight (0.0-1.0) | `0.3` |
| `--max-chunk` | Max chars per chunk for long text | `250` |
| `-q, --quiet` | Suppress progress output | - |
| `-v, --version` | Show version | - |

## Python API

```python
from text_to_audio import TextToAudio

# Initialize
tts = TextToAudio(
    model_type="turbo",  # or "standard", "multilingual"
    device="auto",       # or "cuda", "mps", "cpu"
    max_chunk_chars=250, # for long text splitting
)

# Generate audio
wav = tts.generate(
    text="Your text here. Can be very long - it will be split automatically.",
    audio_prompt_path="voice.wav",  # optional: voice cloning
    exaggeration=0.3,
    cfg_weight=0.3,
    language="en",  # for multilingual model
)

# Save to file
tts.save(wav, "output.wav")

# Access sample rate
print(f"Sample rate: {tts.sample_rate}")
```

### Progress Callback

For long text, track generation progress:

```python
def on_progress(current, total):
    print(f"Generating chunk {current}/{total}")

wav = tts.generate(
    text=long_text,
    progress_callback=on_progress,
)
```

## Long Text Handling

Text longer than `max_chunk_chars` (default 250) is automatically split at sentence boundaries. Audio chunks are concatenated seamlessly. This prevents quality degradation that occurs when generating very long audio in one pass.

## Models

| Model | Description |
|-------|-------------|
| `turbo` | Fast generation, good quality (default) |
| `standard` | Higher quality, slower |
| `multilingual` | 23+ languages support |

## Supported Languages (Multilingual Model)

Use `-m multilingual -l <code>`:

Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Swahili, Turkish

## Emotion Tags (Turbo Model)

Paralinguistic tags are native to the Turbo model:
- `[laugh]` - laughter
- `[chuckle]` - light laughter
- `[sigh]` - sighing
- `[cough]` - coughing
- `[gasp]` - gasping
- `[groan]` - groaning
- `[sniff]` - sniffing
- `[shush]` - shushing
- `[clear throat]` - throat clearing
- `[yawn]` - yawning

Example: `"That's hilarious [laugh] tell me more!"`

## Troubleshooting

### Perth Watermarker Error

If you see `TypeError: 'NoneType' object is not callable` related to `PerthImplicitWatermarker`, the package automatically applies a workaround. This is a known issue when using `uv` without `setuptools`.

### Device Selection

- **Apple Silicon (M1/M2/M3/M4)**: Uses `mps` automatically
- **NVIDIA GPU**: Uses `cuda` automatically
- **CPU fallback**: Works on any system

Force a specific device with `-d cpu` or `-d mps`.

## License

MIT License - see [LICENSE](LICENSE) for details.

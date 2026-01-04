#!/usr/bin/env python3
"""Example usage of text-to-audio package."""

from text_to_audio import TextToAudio

# Initialize TTS (auto-detects best device: cuda > mps > cpu)
tts = TextToAudio(
    model_type="standard",  # "turbo" (fast), "standard" (quality), "multilingual"
    device="auto",
)

# Example text (long text is automatically split and rejoined)
text = """
Hello! This is an example of text-to-speech generation.
The text will be automatically split into chunks if it's too long.
Each chunk is generated separately and then seamlessly joined together.
"""

# Generate audio (optionally with voice cloning)
wav = tts.generate(
    text=text,
    audio_prompt_path="voice.flac",  # Optional: reference audio for voice cloning
    exaggeration=0.3,               # Expressiveness (0.0-1.0)
    cfg_weight=0.3,                 # CFG weight (0.0-1.0)
)

# Save to file
tts.save(wav, "output.wav")
print(f"Saved output.wav ({wav.shape[1] / tts.sample_rate:.1f}s)")

"""Text-to-Audio generator using Chatterbox TTS."""

__version__ = "0.1.0"

from .patches import apply_patches
from .tts import TextToAudio

__all__ = ["TextToAudio", "apply_patches", "__version__"]

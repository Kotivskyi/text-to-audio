"""Text-to-Audio generator using Chatterbox TTS."""

__version__ = "1.0.1"

from .patches import apply_patches
from .tts import TextToAudio

__all__ = ["TextToAudio", "apply_patches", "__version__"]

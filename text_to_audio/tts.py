"""Text-to-Speech wrapper with long text support."""

import re
from typing import Optional

import torch
import torchaudio as ta

from .patches import apply_patches


def get_device(device_arg: str = "auto") -> str:
    """Determine the best available device."""
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_text(text: str, max_chars: int = 250) -> list[str]:
    """Split long text into chunks at sentence boundaries.

    Args:
        text: Input text to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    # Split on sentence boundaries
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence exceeds max_chars, start a new chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Handle case where a single sentence exceeds max_chars
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # Split on commas or other natural breaks
            sub_parts = re.split(r"(?<=[,;:])\s+", chunk)
            sub_chunk = ""
            for part in sub_parts:
                if sub_chunk and len(sub_chunk) + len(part) + 1 > max_chars:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = part
                else:
                    sub_chunk = sub_chunk + " " + part if sub_chunk else part
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())

    return final_chunks if final_chunks else [text]


class TextToAudio:
    """Text-to-Audio generator with long text support."""

    def __init__(
        self,
        model_type: str = "turbo",
        device: str = "auto",
        max_chunk_chars: int = 250,
    ):
        """Initialize the TTS model.

        Args:
            model_type: Model type - 'turbo', 'standard', or 'multilingual'.
            device: Device to use - 'cuda', 'mps', 'cpu', or 'auto'.
            max_chunk_chars: Maximum characters per chunk for long text.
        """
        self.device = get_device(device)
        self.model_type = model_type
        self.max_chunk_chars = max_chunk_chars

        # Apply patches before loading models
        apply_patches(self.device)

        # Load the model
        self.model = self._load_model()

    def _load_model(self):
        """Load the appropriate Chatterbox model."""
        if self.model_type == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            return ChatterboxTurboTTS.from_pretrained(device=self.device)
        elif self.model_type == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            return ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        else:
            from chatterbox.tts import ChatterboxTTS
            return ChatterboxTTS.from_pretrained(device=self.device)

    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        return self.model.sr

    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.3,
        cfg_weight: float = 0.3,
        language: str = "en",
        progress_callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Generate audio from text, handling long text automatically.

        Args:
            text: Text to convert to speech.
            audio_prompt_path: Path to reference audio for voice cloning.
            exaggeration: Expressiveness level (0.0-1.0).
            cfg_weight: CFG weight (0.0-1.0).
            language: Language code for multilingual model.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Audio tensor.
        """
        chunks = split_text(text, self.max_chunk_chars)

        if len(chunks) == 1:
            return self._generate_chunk(
                chunks[0], audio_prompt_path, exaggeration, cfg_weight, language
            )

        # Generate audio for each chunk and concatenate
        audio_parts = []
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))

            wav = self._generate_chunk(
                chunk, audio_prompt_path, exaggeration, cfg_weight, language
            )
            audio_parts.append(wav)

        # Concatenate all audio parts along the time dimension
        return torch.cat(audio_parts, dim=1)

    def _generate_chunk(
        self,
        text: str,
        audio_prompt_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
        language: str,
    ) -> torch.Tensor:
        """Generate audio for a single chunk of text."""
        gen_kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }

        if audio_prompt_path:
            gen_kwargs["audio_prompt_path"] = audio_prompt_path

        if self.model_type == "multilingual":
            gen_kwargs["language_id"] = language

        return self.model.generate(text, **gen_kwargs)

    def save(self, audio: torch.Tensor, output_path: str) -> None:
        """Save audio tensor to file.

        Args:
            audio: Audio tensor to save.
            output_path: Output file path.
        """
        ta.save(output_path, audio, self.sample_rate)

"""Command-line interface for Text-to-Audio generator."""

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .tts import TextToAudio


class ProgressTracker:
    """Track progress and estimate total time."""

    def __init__(self):
        self.start_time = None
        self.total_estimate = None

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"

    def __call__(self, current: int, total: int) -> None:
        """Print progress with total time estimate."""
        now = time.time()

        if current == 1:
            self.start_time = now
            print(f"  Chunk {current}/{total}...")
            return

        elapsed = now - self.start_time
        avg_time_per_chunk = elapsed / (current - 1)
        self.total_estimate = avg_time_per_chunk * total

        elapsed_str = self.format_time(elapsed)
        total_str = self.format_time(self.total_estimate)

        print(f"  Chunk {current}/{total} ({elapsed_str} / ~{total_str})")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="text-to-audio",
        description="Generate speech from text using Chatterbox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  text-to-audio "Hello world"
  text-to-audio "Hello world" -o hello.wav
  text-to-audio "Hello world" -r voice.wav
  text-to-audio "Bonjour" -m multilingual -l fr
  text-to-audio -i long_text.txt -o output.wav
  text-to-audio "Text with [laugh] emotions"
        """,
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to convert to speech (or use -i for file input)",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input text file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.wav",
        help="Output audio file (default: output.wav)",
    )
    parser.add_argument(
        "-r", "--reference",
        type=str,
        help="Reference audio file for voice cloning",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="standard",
        choices=["turbo", "standard", "multilingual"],
        help="Model type: turbo (fast), standard, or multilingual (default: turbo)",
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default="en",
        help="Language code for multilingual model (default: en)",
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "-e", "--exaggeration",
        type=float,
        default=0.3,
        help="Expressiveness level 0.0-1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=0.3,
        help="CFG weight 0.0-1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=250,
        help="Max characters per chunk for long text (default: 250)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    # Get text from argument or file
    if args.text:
        text = args.text
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            return 1
        text = input_path.read_text().strip()
    else:
        parser.error("Provide text as argument or use -i for input file")

    if not text:
        print("Error: Empty text provided", file=sys.stderr)
        return 1

    # Print configuration
    if not args.quiet:
        print(f"Model: {args.model}")
        print(f"Device: {args.device}")
        if args.reference:
            print(f"Voice reference: {args.reference}")
        if args.model == "multilingual":
            print(f"Language: {args.language}")
        print(f"Text length: {len(text)} characters")

    # Initialize TTS
    if not args.quiet:
        print("Loading model...")

    try:
        tts = TextToAudio(
            model_type=args.model,
            device=args.device,
            max_chunk_chars=args.max_chunk,
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Model loaded! (device: {tts.device})")
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"Generating: {preview}")

    # Generate audio
    gen_start = time.time()
    try:
        callback = None if args.quiet else ProgressTracker()
        wav = tts.generate(
            text=text,
            audio_prompt_path=args.reference,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg,
            language=args.language,
            progress_callback=callback,
        )
    except Exception as e:
        print(f"Error generating audio: {e}", file=sys.stderr)
        return 1
    gen_elapsed = time.time() - gen_start

    # Save output
    try:
        tts.save(wav, args.output)
    except Exception as e:
        print(f"Error saving audio: {e}", file=sys.stderr)
        return 1

    # Print result
    duration = wav.shape[1] / tts.sample_rate
    if not args.quiet:
        if gen_elapsed < 60:
            time_str = f"{gen_elapsed:.1f}s"
        else:
            minutes = int(gen_elapsed // 60)
            seconds = int(gen_elapsed % 60)
            time_str = f"{minutes}m {seconds}s"
        print(f"Saved: {args.output} ({duration:.1f}s audio, generated in {time_str})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

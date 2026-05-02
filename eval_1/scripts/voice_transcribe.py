#!/usr/bin/env python3
"""Transcribe a WAV file with Whisper large-v3-turbo, biased for SO-101 prompts.

Usage:
    voice_transcribe.py <path-to-wav>

Prints only the transcript text to stdout. Logs to stderr.
"""
import sys
from faster_whisper import WhisperModel

DOMAIN_HINT = (
    "The user is commanding a SO-101 robot arm. "
    "Common words: banana, bowl, blue, red, green, put, place, pick, colored. "
    "Common phrases: Put the banana in the blue colored bowl. "
    "Place the banana in the red bowl. "
    "Pick the banana and put it in the green bowl. "
    "Put the banana in the second bowl from the left. "
    "Put the banana in the bowl that is not green and not blue."
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: voice_transcribe.py <wav>", file=sys.stderr)
        return 2
    wav = sys.argv[1]

    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        wav,
        language="en",
        initial_prompt=DOMAIN_HINT,
        temperature=0,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    print(f"transcribe: duration={info.duration:.2f}s", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

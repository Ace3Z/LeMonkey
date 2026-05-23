#!/usr/bin/env python3
"""Strip the "from the robot perspective" qualifier from Eval 2 prompts.

Eval-day spatial prompts can
have the qualifier embedded inline, e.g.:

    "Put the banana into the 2nd bowl from the left from the robot perspective"
    "Put the banana into the bowl on the right of the red bowl from the robot perspective"

The SmolVLA Eval 2 model was NOT trained on that qualifier — the 180-episode
dataset's prompts never contain "from the robot perspective". Feeding the OOD
phrase to the policy pushes it off-distribution. Solution: strip the qualifier
before sending to the model. The trajectory the model produces should be the
same as if the qualifier were never there.

Matched phrase (exact, case-insensitive, optional trailing period):
    "from the robot perspective"

Anything else — including "robot's perspective", "camera perspective", etc.
— is left untouched. The filter is intentionally
narrow.

Usage:
    from filter_prompt import filter_prompt
    cleaned = filter_prompt(raw)

CLI:
    filter_prompt.py "Put the banana in the leftmost bowl from the robot perspective"
    → Put the banana in the leftmost bowl
"""
from __future__ import annotations

import re
import sys

# Match: optional leading whitespace, the literal phrase
# "from the robot perspective", optional trailing period, trailing whitespace.
_PERSPECTIVE_RE = re.compile(
    r"\s*from\s+the\s+robot\s+perspective\s*\.?\s*",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def filter_prompt(prompt: str) -> str:
    """Strip the perspective qualifier from a prompt.

    Idempotent — calling twice yields the same result.
    Returns the prompt unchanged (with whitespace normalized) if no
    qualifier is present.
    """
    cleaned = _PERSPECTIVE_RE.sub(" ", prompt)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: filter_prompt.py '<prompt>'", file=sys.stderr)
        sys.exit(2)
    print(filter_prompt(sys.argv[1]))

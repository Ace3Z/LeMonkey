#!/usr/bin/env python3
"""Strip the "from the robot perspective" qualifier from Eval 2 prompts.

Eval-day spatial prompts can carry the qualifier embedded inline, e.g.::

    "Put the banana into the 2nd bowl from the left from the robot perspective"
    "Put the banana into the bowl on the right of the red bowl from the robot perspective"

The SmolVLA Eval 2 model was not trained on that qualifier: the 180-episode
dataset's prompts never contain "from the robot perspective". Feeding the
out-of-distribution phrase to the policy pushes it off-distribution. The
filter normalises the prompt by stripping the qualifier *before* it reaches
the model; the trajectory the policy then produces is the same as if the
qualifier were never present.

The filter is intentionally narrow:

* Matched (case-insensitive, optional trailing period): ``from the robot perspective``.
* Not matched: ``robot's perspective``, ``camera perspective``, etc.

Usage (library)::

    from filter_prompt import filter_prompt
    cleaned = filter_prompt(raw)

Usage (CLI)::

    $ python filter_prompt.py "Put the banana in the leftmost bowl from the robot perspective"
    Put the banana in the leftmost bowl
"""
from __future__ import annotations

import re
import sys
from typing import Final

# Matches the literal phrase with surrounding whitespace and an optional
# trailing period, case-insensitively.
_PERSPECTIVE_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*from\s+the\s+robot\s+perspective\s*\.?\s*",
    re.IGNORECASE,
)

# Collapses runs of whitespace to a single space so the cleaned prompt is
# tidy regardless of where the qualifier was removed.
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def filter_prompt(prompt: str) -> str:
    """Strip the "from the robot perspective" qualifier from ``prompt``.

    Args:
        prompt: Raw eval-day instruction string, possibly with the
            qualifier embedded.

    Returns:
        The prompt with every occurrence of the qualifier removed and
        internal whitespace collapsed. Idempotent: ``filter_prompt(s) ==
        filter_prompt(filter_prompt(s))`` for every ``s``. If the prompt
        does not contain the qualifier, the only effect is whitespace
        normalisation.
    """
    cleaned = _PERSPECTIVE_RE.sub(" ", prompt)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: filter_prompt.py '<prompt>'", file=sys.stderr)
        sys.exit(2)
    print(filter_prompt(sys.argv[1]))

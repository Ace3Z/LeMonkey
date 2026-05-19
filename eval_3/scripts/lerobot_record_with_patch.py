#!/usr/bin/env python3
"""Drop-in replacement for `lerobot-record` that applies the SmolVLM
boundaries-device patch before invoking lerobot's record entry point.

transformers==4.55.0's `SmolVLMVisionEmbeddings.forward` builds the
`boundaries` tensor on CPU while the rest of the math is on the device,
so the bare `lerobot-record` command dies the first time it processes a
camera frame. This wrapper applies the patch once and then calls the same
`lerobot.scripts.lerobot_record.main` that `lerobot-record` itself invokes
— all CLI args are passed through unchanged.

Usage (same as lerobot-record):

    python eval_3/scripts/lerobot_record_with_patch.py \\
        --robot.type=so101_follower ... \\
        --policy.path=<dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make eval_3/aug importable so we can pull in the inference patch.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aug"))
from smolvlm_inference_patch import apply as _apply_smolvlm_patch  # noqa: E402

_apply_smolvlm_patch()

from lerobot.scripts.lerobot_record import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

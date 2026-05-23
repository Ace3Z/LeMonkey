#!/usr/bin/env python3
"""Fast cotrain merge via shutil.copy → os.link monkey-patch.

The vendored lerobot's `aggregate_datasets()` uses `shutil.copy(src, dst)`
to copy each of the ~18,800 mp4s (aggregate.py:386,408). On the same
filesystem that's wasteful — `os.link(src, dst)` is O(1).

Risk: hardlinks share inode → if either path is later modified, the other
sees the change. Acceptable here because:
  1. We only read from the merged dataset to push to HF; we never write to it.
  2. The source per-episode dirs are also read-only after generation.
  3. HF's upload_large_folder reads file contents, so hardlinks transmit fine.

Falls back to real copy on EXDEV (cross-device).

Usage: identical to merge_eval3_episodes.py — same args forwarded.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

# ── Monkey-patch shutil.copy → os.link before any lerobot import ──────────
_orig_copy = shutil.copy
_hardlink_count = 0
_realcopy_count = 0

def _hardlink_copy(src, dst, *args, **kwargs):
    global _hardlink_count, _realcopy_count
    src, dst = str(src), str(dst)
    try:
        os.link(src, dst)
        _hardlink_count += 1
        return dst
    except OSError as e:
        if e.errno in (17, 18):  # EEXIST (link exists) or EXDEV (cross-device)
            if e.errno == 17:
                # dest already exists — overwrite via real copy
                os.unlink(dst)
                try:
                    os.link(src, dst)
                    _hardlink_count += 1
                    return dst
                except OSError:
                    pass
            _realcopy_count += 1
            return _orig_copy(src, dst, *args, **kwargs)
        raise

shutil.copy = _hardlink_copy
# Also patch shutil.copy2 in case anything uses that
_orig_copy2 = shutil.copy2
def _hardlink_copy2(src, dst, *args, **kwargs):
    try:
        os.link(str(src), str(dst))
        global _hardlink_count
        _hardlink_count += 1
        return str(dst)
    except OSError:
        return _orig_copy2(src, dst, *args, **kwargs)
shutil.copy2 = _hardlink_copy2

# ── Now invoke the standard merge script ─────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "eval_3" / "scripts"))

# Hack: import merge_eval3_episodes as a module so its main() runs
import importlib.util
spec = importlib.util.spec_from_file_location(
    "merge_eval3_episodes",
    str(_REPO / "eval_3" / "scripts" / "merge_eval3_episodes.py"),
)
_merge_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_merge_mod)

if __name__ == "__main__":
    t = time.time()
    rc = _merge_mod.main()
    elapsed = time.time() - t
    print(f"\n  ── hardlink stats ──")
    print(f"     hardlinks: {_hardlink_count}")
    print(f"     real copies (fallback): {_realcopy_count}")
    print(f"     total wall: {elapsed/60:.1f} min")
    sys.exit(rc)

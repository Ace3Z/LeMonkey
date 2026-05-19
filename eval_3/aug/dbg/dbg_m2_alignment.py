"""Smoke test + visual gate for `eval_3/aug/m2_alignment.py`.

What this verifies:

1. **Geometry**: bbox_to_patch_mask correctly maps a face bbox in 480×640
   pixel space to an 8×8 patch grid (the SmolVLA post-connector image grid).
   Draws an overlay PNG so we can eyeball-check.
2. **Forward**: `m2_align_loss` produces a finite scalar on dummy inputs and
   handles all-invalid frames gracefully.
3. **Backward**: gradients flow back through the `hidden_state` (so the M2
   loss can train SmolLM2 layer 0..N) even though the projector is frozen.
4. **End-to-end on a real frame**: pulls one frame from a variant we've
   already validated visually (`quick_swift_SOL_ep04_*_v00`), computes the
   loss with a random hidden_state, and renders the patch mask overlay so
   the bbox-vs-patch mapping is humanly verifiable.

Run from project root:

    python eval_3/aug/dbg/dbg_m2_alignment.py

Outputs PNGs into `eval_3/aug/stats/face_labels_dbg/m2_smoke/`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Make the parent eval_3/aug importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m2_alignment import (
    ARCFACE_EMBED_DIM,
    CAMERA1_PATCH_GRID,
    FrozenProjector,
    NUM_CAMERA1_PATCHES,
    SMOLLM2_HIDDEN_SIZE,
    bbox_to_patch_mask,
    build_supervision_for_frame,
    m2_align_loss,
    slot_to_celeb,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS_DIR = REPO_ROOT / "eval_3/aug/stats/face_labels"
BANK_ROOT = Path.home() / "Downloads/eval3_celebs"
MANIFEST = REPO_ROOT / "eval_3/aug/stats/celeb_embeddings.json"
AUG_ROOT = Path.home() / "Downloads/eval3_track3_aug"
OUT_DIR = REPO_ROOT / "eval_3/aug/stats/face_labels_dbg/m2_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) GEOMETRY: a synthetic bbox and the mask it produces
# ---------------------------------------------------------------------------

def test_geometry() -> bool:
    print("\n[1/4] geometry test (bbox → patch mask)")
    # A typical face bbox: ~80×80 px around (170, 280) in the 480×640 frame.
    bbox = (130.0, 240.0, 210.0, 320.0)
    mask = bbox_to_patch_mask(bbox)
    print(f"  bbox in 480x640: {bbox}")
    print(f"  mask shape: {mask.shape}, n_active: {mask.sum()}")
    # Expected (LeRobot resize_with_pad with ratio=1.25 + left+top pad):
    # (130/1.25=104, 240/1.25+128=320, 210/1.25=168, 320/1.25+128=384)
    # → col_lo=floor(104/64)=1, col_hi=ceil(168/64)-1=2 → cols 1-2
    # → row_lo=floor(320/64)=5, row_hi=ceil(384/64)-1=5 → row 5 only
    # 2 cols × 1 row = 2 active patches.
    grid = mask.reshape(CAMERA1_PATCH_GRID, CAMERA1_PATCH_GRID)
    print(f"  mask grid (rows=y, cols=x):")
    for row in grid:
        print(f"    {''.join('█' if c else '·' for c in row)}")
    n_active = mask.sum()
    ok = n_active >= 1 and n_active <= 6  # 2-6 patches is the realistic range
    print(f"  → {'OK' if ok else 'FAIL'} (n_active={n_active} in [1..6])")
    return bool(ok)


# ---------------------------------------------------------------------------
# 2) FORWARD/BACKWARD: synthetic inputs, verify shapes + grad flow
# ---------------------------------------------------------------------------

def test_loss_forward_backward() -> bool:
    print("\n[2/4] loss forward + backward with synthetic batch")
    torch.manual_seed(0)
    B = 4
    prefix_len = 64 + 64 + 48 + 1  # cam1 + cam2_zero + lang + state
    H = SMOLLM2_HIDDEN_SIZE

    # Wrap the hidden state in a leaf tensor with requires_grad
    hidden = torch.randn(B, prefix_len, H, requires_grad=True)
    masks = torch.zeros(B, 3, NUM_CAMERA1_PATCHES, dtype=torch.bool)
    # Make 2 patches active per (b, s) at varying positions
    for b in range(B):
        for s in range(3):
            base = (b * 3 + s) % NUM_CAMERA1_PATCHES
            masks[b, s, base] = True
            masks[b, s, (base + 1) % NUM_CAMERA1_PATCHES] = True
    valid = torch.tensor([[True, True, True],
                          [True, True, False],
                          [True, False, True],
                          [False, True, True]])
    targets = torch.randn(B, 3, ARCFACE_EMBED_DIM)
    targets = torch.nn.functional.normalize(targets, dim=-1)

    proj = FrozenProjector()
    result = m2_align_loss(hidden, masks, valid, targets, proj)

    expected_valid = int(valid.sum().item())  # 3 + 2 + 2 + 2 = 9
    print(f"  loss = {result.loss.item():+.4f}  (negative-mean cos; expect range [-1, +1])")
    print(f"  n_valid = {result.n_valid}  (expected {expected_valid} out of 12 total slots)")
    print(f"  per_slot_cos NaN count = {torch.isnan(result.per_slot_cos).sum().item()}  (expected {12 - expected_valid})")

    # Backward — should produce non-NaN grad on hidden, zero grad on projector
    result.loss.backward()
    print(f"  hidden.grad: any non-zero = {(hidden.grad.abs() > 0).any().item()}, "
          f"any NaN = {torch.isnan(hidden.grad).any().item()}")
    proj_grad = next(p.grad for p in proj.parameters() if p.grad is not None) if any(p.grad is not None for p in proj.parameters()) else None
    print(f"  projector grad accumulated: {proj_grad is not None}  (expected: None, since frozen)")

    ok = (result.n_valid == expected_valid
          and (hidden.grad.abs() > 0).any().item()
          and not torch.isnan(hidden.grad).any().item()
          and proj_grad is None)
    print(f"  → {'OK' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# 3) EDGE CASE: all-invalid frame returns zero loss without crashing
# ---------------------------------------------------------------------------

def test_all_invalid() -> bool:
    print("\n[3/4] all-invalid frames return zero loss with attached grad")
    B = 2
    hidden = torch.randn(B, 200, SMOLLM2_HIDDEN_SIZE, requires_grad=True)
    masks = torch.zeros(B, 3, NUM_CAMERA1_PATCHES, dtype=torch.bool)
    valid = torch.zeros(B, 3, dtype=torch.bool)  # everything invalid
    targets = torch.randn(B, 3, ARCFACE_EMBED_DIM)
    proj = FrozenProjector()
    result = m2_align_loss(hidden, masks, valid, targets, proj)
    print(f"  loss = {result.loss.item()}  n_valid = {result.n_valid}")
    result.loss.backward()  # must not crash
    ok = result.n_valid == 0 and result.loss.item() == 0.0
    print(f"  → {'OK' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# 4) END-TO-END: pull a real frame + bbox + ArcFace centroid, render the overlay
# ---------------------------------------------------------------------------

def test_real_frame() -> bool:
    print("\n[4/4] end-to-end on a real frame from quick_swift_SOL_ep04_185956")
    if not MANIFEST.exists():
        print(f"  SKIP — manifest not found at {MANIFEST}")
        return True

    manifest = json.loads(MANIFEST.read_text())
    centroids = {c: np.asarray(info["centroid"], dtype=np.float32)
                 for c, info in manifest["celebs"].items() if info["centroid"]}

    src = "quick_swift_SOL_ep04_20260511_185956"
    labels_path = LABELS_DIR / f"{src}.face_labels.json"
    if not labels_path.exists():
        print(f"  SKIP — face_labels not found at {labels_path}")
        return True

    fl = json.loads(labels_path.read_text())
    aug_p = AUG_ROOT / fl["representative_variant"] / "augmentation.json"
    aug = json.loads(aug_p.read_text())
    new_lmr = aug["new_layout_camera_lmr"]
    celebs = slot_to_celeb(new_lmr)
    print(f"  representative: {fl['representative_variant']}")
    print(f"  new_layout_camera_lmr: {new_lmr} → slots {celebs}")

    # frame 0 — known to have 3 faces from earlier validation
    frame_entry = fl["frames"][0]
    print(f"  frame_idx=0, n_visible_faces={frame_entry['n_visible_faces']}")

    masks_np, valid_np, targets_np = build_supervision_for_frame(
        frame_entry, new_lmr, centroids,
    )
    print(f"  valid: {valid_np.tolist()}, mask sizes: {[int(m.sum()) for m in masks_np]}")

    # Run the loss with a fixed random hidden_state to confirm the wiring
    torch.manual_seed(42)
    prefix_len = 200
    hidden = torch.randn(1, prefix_len, SMOLLM2_HIDDEN_SIZE, requires_grad=True)
    masks = torch.from_numpy(masks_np[None]).to(torch.bool)
    valid = torch.from_numpy(valid_np[None]).to(torch.bool)
    targets = torch.from_numpy(targets_np[None]).to(torch.float32)
    proj = FrozenProjector()
    res = m2_align_loss(hidden, masks, valid, targets, proj)
    print(f"  loss = {res.loss.item():+.4f}, n_valid = {res.n_valid}")
    res.loss.backward()
    grad_nz = (hidden.grad.abs() > 0).any().item()
    print(f"  hidden.grad non-zero: {grad_nz}")

    # Visual gate: overlay the bboxes and patch grid on the camera1 frame.
    mp4 = AUG_ROOT / fl["representative_variant"] / "videos/observation.images.camera1/chunk-000/file-000.mp4"
    cap = cv2.VideoCapture(str(mp4))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"  WARN: failed to decode frame 0 of {mp4} — skipping visual gate")
        return grad_nz

    # Compose: original frame (480×640) + the resized-with-pad 512×512 + overlay patches.
    # Mirror LeRobot's modeling_smolvla.py:134-152 exactly:
    #   ratio = max(cur_w/W, cur_h/H); resized to (cur/ratio); pad LEFT+TOP only.
    target_hw = (512, 512)
    orig_h, orig_w = frame.shape[:2]
    ratio = max(orig_w / target_hw[1], orig_h / target_hw[0])
    new_w = int(orig_w / ratio)
    new_h = int(orig_h / ratio)
    pad_x = max(0, target_hw[1] - new_w)         # all on LEFT
    pad_y = max(0, target_hw[0] - new_h)         # all on TOP

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_hw[0], target_hw[1], 3), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(canvas_rgb)
    # Draw the 8×8 patch grid
    px = target_hw[1] / CAMERA1_PATCH_GRID
    py = target_hw[0] / CAMERA1_PATCH_GRID
    for i in range(CAMERA1_PATCH_GRID + 1):
        ax.axhline(i * py, color="white", linewidth=0.5, alpha=0.4)
        ax.axvline(i * px, color="white", linewidth=0.5, alpha=0.4)

    slot_colors = ["#00cc00", "#cc8800", "#cc0000"]  # L=green, M=orange, R=red
    for slot in range(3):
        if not valid_np[slot]:
            continue
        bb = frame_entry["bboxes"][slot]
        # Original-frame bbox → target-frame bbox (left+top padding, divide by ratio).
        rx1 = bb["x1"] / ratio + pad_x
        ry1 = bb["y1"] / ratio + pad_y
        rx2 = bb["x2"] / ratio + pad_x
        ry2 = bb["y2"] / ratio + pad_y
        # Bbox rectangle
        ax.add_patch(plt.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1,
                                    fill=False, edgecolor=slot_colors[slot],
                                    linewidth=2))
        # Highlight the active patches
        grid = masks_np[slot].reshape(CAMERA1_PATCH_GRID, CAMERA1_PATCH_GRID)
        for r in range(CAMERA1_PATCH_GRID):
            for c in range(CAMERA1_PATCH_GRID):
                if grid[r, c]:
                    ax.add_patch(plt.Rectangle((c * px, r * py), px, py,
                                                fill=True, alpha=0.30,
                                                color=slot_colors[slot]))
        ax.text(rx1, max(ry1 - 5, 12),
                f"{['L','M','R'][slot]}: {celebs[slot]}  ({int(masks_np[slot].sum())} patches)",
                color=slot_colors[slot], fontsize=9, weight="bold",
                bbox=dict(facecolor="white", alpha=0.7, pad=2))
    ax.set_xlim(0, target_hw[1])
    ax.set_ylim(target_hw[0], 0)
    ax.set_xticks(np.arange(0, target_hw[1] + 1, px))
    ax.set_yticks(np.arange(0, target_hw[0] + 1, py))
    ax.set_title(f"{src}  frame 0 — bbox → 8×8 patch mask overlay\n"
                 f"(LeRobot resize_with_pad: 640×480 → {new_w}×{new_h} at top-left, "
                 f"pad_x={pad_x} left, pad_y={pad_y} top)",
                 fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / f"{src}__frame0_bbox_to_patch.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [png] {out}")
    return grad_nz


def main() -> int:
    print("=" * 70)
    print("M2 alignment smoke test")
    print("=" * 70)
    results = [
        test_geometry(),
        test_loss_forward_backward(),
        test_all_invalid(),
        test_real_frame(),
    ]
    print("\n" + "=" * 70)
    if all(results):
        print(f"ALL {len(results)}/{len(results)} CHECKS PASSED")
        return 0
    print(f"{sum(results)}/{len(results)} CHECKS PASSED — failures: "
          f"{[i for i, r in enumerate(results) if not r]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

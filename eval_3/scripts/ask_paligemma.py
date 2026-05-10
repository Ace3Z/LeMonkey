#!/usr/bin/env python3
"""Interactive zero-shot probe for google/paligemma-3b-pt-224.

Modes:

  # one-shot, file
  python ask_paligemma.py /path/to/img.png "Who is this person?"

  # REPL — model stays loaded, fire many queries fast
  python ask_paligemma.py

  # CAMERA — snap a fresh frame from /dev/video0 before each query
  python ask_paligemma.py --camera
  python ask_paligemma.py --camera --device-path /dev/video1

  # CAMERA, one-shot — snap once, ask once, exit
  python ask_paligemma.py --camera --once "Who is this person?"

  # SWEEP — same prompt over every image in a directory
  python ask_paligemma.py --sweep "Who is this person?" /path/to/dir/

In REPL / camera-REPL mode you'll be prompted for an image path and a question
(in camera mode the frame comes from the camera; in REPL mode you type a path).
Type 'quit' or hit ENTER on an empty image path to exit.

Each captured frame is saved to /tmp/paligemma_last_frame.jpg for inspection.

Loads from the HF cache populated by probe_paligemma.py — no re-download.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

MODEL_ID = "google/paligemma-3b-pt-224"


def load(device: str, dtype: torch.dtype):
    print(f"loading {MODEL_ID} on {device} ({dtype})...", flush=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = (
        PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype)
        .to(device)
        .eval()
    )
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
    return proc, model


@torch.inference_mode()
def ask(proc, model, image: Image.Image, prompt: str, device: str, dtype: torch.dtype, max_new_tokens: int = 30) -> str:
    """PaliGemma's prompt convention: 'answer en <question>' for VQA."""
    full_prompt = prompt if prompt.strip().startswith("answer ") else f"answer en {prompt}"
    inputs = proc(text=full_prompt, images=image.convert("RGB"), return_tensors="pt").to(device, dtype)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="Image path (omit for REPL or --camera)")
    ap.add_argument("prompt", nargs="?", default="Who is this person?", help="Question (default: 'Who is this person?')")
    ap.add_argument("--sweep", metavar="PROMPT",
                    help="Sweep mode: ask the same prompt of every image in a directory. "
                         "Usage: --sweep 'prompt' DIR  (positional 'image' becomes the dir)")
    ap.add_argument("--camera", action="store_true",
                    help="Camera mode: capture a fresh frame from --device-path before each query")
    ap.add_argument("--device-path", default="/dev/video0",
                    help="V4L2 device path for --camera (default /dev/video0)")
    ap.add_argument("--once", metavar="PROMPT",
                    help="With --camera: snap one frame, ask this prompt, exit")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="Rotate captured frame by N degrees CCW before inference / saving")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, help="float16/bfloat16/float32; default: float16 on cuda")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype:
        dtype = getattr(torch, args.dtype)
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32

    proc, model = load(device, dtype)

    # Camera modes
    if args.camera:
        import cv2
        FRAME_OUT = Path("/tmp/paligemma_last_frame.jpg")
        cap = cv2.VideoCapture(args.device_path, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            print(f"cannot open {args.device_path}", file=sys.stderr)
            return 1
        # warm up auto-exposure
        for _ in range(10):
            cap.read()

        rot_map = {
            0: None,
            90: cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE,
        }
        rot_code = rot_map[args.rotate]

        def snap():
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                return None
            if rot_code is not None:
                frame_bgr = cv2.rotate(frame_bgr, rot_code)
            cv2.imwrite(str(FRAME_OUT), frame_bgr)
            return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        try:
            if args.once:
                img = snap()
                if img is None:
                    print("camera read failed", file=sys.stderr)
                    return 1
                t0 = time.time()
                ans = ask(proc, model, img, args.once, device, dtype, args.max_new_tokens)
                print(f"\nframe saved to: {FRAME_OUT}")
                print(f"  Q: {args.once}")
                print(f"  A: {ans}    ({time.time()-t0:.1f}s)")
                return 0

            print(f"\nCAMERA REPL — capturing from {args.device_path}, frames saved to {FRAME_OUT}")
            print("Empty question or 'quit' to exit. ENTER alone uses the previous question.\n")
            last_q = "Who is this person?"
            while True:
                try:
                    q = input(f"question (default {last_q!r}): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if q.lower() in {"q", "quit", "exit"}:
                    break
                if not q:
                    q = last_q
                last_q = q
                img = snap()
                if img is None:
                    print("  ! camera read failed")
                    continue
                t0 = time.time()
                ans = ask(proc, model, img, q, device, dtype, args.max_new_tokens)
                print(f"  → {ans}    ({time.time()-t0:.1f}s)   [frame: {FRAME_OUT}]\n")
        finally:
            cap.release()
        return 0

    # Sweep mode: --sweep "prompt"  + positional dir
    if args.sweep:
        sweep_prompt = args.sweep
        sweep_dir = Path(args.image) if args.image else None
        if sweep_dir is None or not sweep_dir.is_dir():
            print(f"--sweep needs a directory as the first positional arg (got {sweep_dir})", file=sys.stderr)
            return 1
        files = sorted(p for p in sweep_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        print(f"\nsweeping {len(files)} images in {sweep_dir} with prompt: {sweep_prompt!r}\n")
        for p in files:
            try:
                img = Image.open(p)
            except Exception as e:
                print(f"  ! {p.name}: cannot open ({e})")
                continue
            ans = ask(proc, model, img, sweep_prompt, device, dtype, args.max_new_tokens)
            print(f"  {p.name:30s}  {ans!r}")
        return 0

    # One-shot
    if args.image:
        img_path = Path(args.image)
        if not img_path.is_file():
            print(f"file not found: {img_path}", file=sys.stderr)
            return 1
        ans = ask(proc, model, Image.open(img_path), args.prompt, device, dtype, args.max_new_tokens)
        print(f"\n{img_path}\n  Q: {args.prompt}\n  A: {ans}")
        return 0

    # REPL
    print("\nREPL mode. Empty 'image' or 'quit' to exit.\n")
    while True:
        try:
            img_path = input("image:    ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not img_path or img_path.lower() in {"q", "quit", "exit"}:
            break
        p = Path(img_path).expanduser()
        if not p.is_file():
            print(f"  ! not a file: {p}")
            continue
        try:
            prompt = input("question: ").strip() or "Who is this person?"
        except (EOFError, KeyboardInterrupt):
            print()
            break
        try:
            img = Image.open(p)
        except Exception as e:
            print(f"  ! cannot open {p}: {e}")
            continue
        t0 = time.time()
        ans = ask(proc, model, img, prompt, device, dtype, args.max_new_tokens)
        print(f"  → {ans}    ({time.time()-t0:.1f}s)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

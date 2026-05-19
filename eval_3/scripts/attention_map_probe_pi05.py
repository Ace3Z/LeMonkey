#!/usr/bin/env python3
"""Gating experiment G1: vanilla Pi0.5 attention probe.

Loads `lerobot/pi05_base` (NO fine-tuning) and visualises the
celeb-name-token → image-patch self-attention at several PaliGemma
layers. The point is to detect whether vanilla PaliGemma already
exhibits the (1,7)-style "sink" pattern we saw on SmolVLM2; if it
does, M2 + KLAL training won't fix the architecture and we'd need
a different plan.

Differences vs eval_3/scripts/attention_map_probe.py (SmolVLA):
- Patch grid: 16x16 = 256 (PaliGemma SigLIP, patch_size=14, image=224)
- Image preprocessing: 480x640 → 224x224 with CENTER padding (28 px
  top + 28 px bottom). vs SmolVLA's left+top.
- Prefix layout: [img_0 (256), img_1 (256), ..., lang_tokens (L)]. No
  state token in prefix (Pi0.5 puts state in suffix via adarms).
- Attention is prefix-LM BIDIRECTIONAL (Pi0.5 sets all prefix tokens
  to attend everywhere within prefix; see modeling_pi05.py:679).
  No causal mask → no architectural reason for sinks.
- Hook target: paligemma.model.language_model.layers[N].self_attn.{q,k}_proj

Usage:
    python eval_3/scripts/attention_map_probe_pi05.py \\
        --repo lerobot/pi05_base \\
        --layers 6 10 14 17 --out /tmp/pi05_attn_probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Pi0.5 + transformers 4.55 compat patch — must run before PI05Policy import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aug"))
from pi05_inference_patch import apply as _apply_pi05_patch  # noqa: E402
_apply_pi05_patch()


PROMPTS = {
    "swift": "Place the coke on Taylor Swift.",
    "obama": "Place the coke on Barack Obama.",
    "lecun": "Place the coke on Yann LeCun.",
}
NAME_PHRASE = {
    "swift": "Taylor Swift",
    "obama": "Barack Obama",
    "lecun": "Yann LeCun",
}

GRID = 16              # PaliGemma 16x16 patches
NUM_PATCHES = GRID * GRID  # 256
IMG_H_RAW, IMG_W_RAW = 480, 640
IMG_HW_TARGET = 224


def _find_token_positions(tokenizer, prompt: str, name_phrase: str):
    full = tokenizer.encode(prompt, add_special_tokens=False)
    for variant in (name_phrase, " " + name_phrase):
        nm = tokenizer.encode(variant, add_special_tokens=False)
        for i in range(len(full) - len(nm) + 1):
            if full[i : i + len(nm)] == nm:
                return i, i + len(nm), len(full)
    raise ValueError(f"Could not locate {name_phrase!r} in {prompt!r}")


def _resize_center_pad(image_chw_01: torch.Tensor) -> torch.Tensor:
    """Mirror Pi0.5's resize_with_pad_torch (modeling_pi05.py:204-216).

    Center-pads 480x640 → 224x224.
    """
    image = image_chw_01.unsqueeze(0)  # (1, 3, 480, 640)
    _, _, h, w = image.shape
    ratio = max(w / IMG_HW_TARGET, h / IMG_HW_TARGET)
    rh, rw = int(h / ratio), int(w / ratio)
    image = F.interpolate(image, size=(rh, rw), mode="bilinear", align_corners=False)
    image = image.clamp(0.0, 1.0)
    pad_h0, rem_h = divmod(IMG_HW_TARGET - rh, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(IMG_HW_TARGET - rw, 2)
    pad_w1 = pad_w0 + rem_w
    image = F.pad(image, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=0.0)
    return image[0]


def _grab_real_frame(dataset_root: Path, ep_idx: int, frame_idx: int = 0):
    import datasets, av
    eps = sorted(str(p) for p in (dataset_root / "meta/episodes").rglob("*.parquet"))
    ds = datasets.load_dataset("parquet", data_files=eps, split="train")
    row = ds[ep_idx]
    chunk = row["videos/observation.images.camera1/chunk_index"]
    fi = row["videos/observation.images.camera1/file_index"]
    vp = dataset_root / f"videos/observation.images.camera1/chunk-{chunk:03d}/file-{fi:03d}.mp4"
    container = av.open(str(vp))
    stream = container.streams.video[0]
    decoded = None
    for i, frame in enumerate(container.decode(stream)):
        if i == frame_idx:
            decoded = frame
            break
    container.close()
    if decoded is None:
        raise IndexError(f"no frame {frame_idx} in {vp}")
    arr = decoded.to_ndarray(format="rgb24")
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def _heatmap_overlay(image_chw_01_224: torch.Tensor, heatmap_NxN: np.ndarray,
                     alpha: float = 0.45) -> Image.Image:
    img = (image_chw_01_224.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hm = heatmap_NxN.astype(np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    hm_up = torch.from_numpy(hm)[None, None].float()
    hm_up = F.interpolate(hm_up, size=img.shape[:2], mode="bilinear",
                          align_corners=False)[0, 0].numpy()
    color = np.stack([hm_up, hm_up * 0.4, np.zeros_like(hm_up)], axis=-1)
    color = (color * 255).astype(np.uint8)
    blended = (img * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="lerobot/pi05_base")
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset-root", default=str(Path.home() / ".cache/huggingface/lerobot/HBOrtiz/so101_eval3_track3_v3_baseline"))
    p.add_argument("--image-path", default=None,
                   help="Bypass dataset lookup: load this PNG/JPG directly (must be 640x480 RGB).")
    p.add_argument("--layers", nargs="+", type=int, default=[6, 10, 14, 17])
    p.add_argument("--episode", type=int, default=100)
    p.add_argument("--frame", type=int, default=10)
    p.add_argument("--out", default=str(Path.home() / "pi05_attn_probe"))
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load Pi0.5 policy. compile_model=False is REQUIRED (graph break under hook).
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor.pipeline import DataProcessorPipeline
    print(f"[load] {args.repo}{'@'+args.revision if args.revision else ''}", flush=True)
    kw = {"revision": args.revision} if args.revision else {}
    policy = PI05Policy.from_pretrained(args.repo, **kw).to(args.device).eval()
    preprocessor = DataProcessorPipeline.from_pretrained(
        args.repo, config_filename="policy_preprocessor.json", **kw
    )
    tokenizer = None
    for step in preprocessor.steps:
        if getattr(step, "input_tokenizer", None) is not None:
            tokenizer = step.input_tokenizer
            break
    assert tokenizer is not None, "no tokenizer in preprocessor steps"
    print(f"[load] tokenizer: {type(tokenizer).__name__}", flush=True)

    # Locate PaliGemma text model + verify shape.
    paligemma = policy.model.paligemma_with_expert.paligemma
    text_model = paligemma.model.language_model
    n_layers = len(text_model.layers)
    n_heads = text_model.layers[0].self_attn.config.num_attention_heads
    n_kv_heads = text_model.layers[0].self_attn.config.num_key_value_heads
    head_dim = text_model.layers[0].self_attn.head_dim
    print(f"[load] text_model.layers={n_layers}  n_heads={n_heads}  "
          f"n_kv_heads={n_kv_heads}  head_dim={head_dim}", flush=True)

    # Real frame → 224x224 center-padded.
    if args.image_path:
        from PIL import Image as _PIL
        _img = _PIL.open(args.image_path).convert("RGB")
        if _img.size != (640, 480):
            _img = _img.resize((640, 480))
        import numpy as _np
        frame_raw = torch.from_numpy(_np.array(_img)).permute(2, 0, 1).float() / 255.0
        print(f"[frame] loaded {args.image_path}", flush=True)
    else:
        frame_raw = _grab_real_frame(Path(args.dataset_root), args.episode, args.frame)
    frame_224 = _resize_center_pad(frame_raw)
    Image.fromarray((frame_224.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(outdir / "input.png")
    print(f"[frame] saved 224x224 input → {outdir/'input.png'}", flush=True)

    # Hooks on q_proj, k_proj per target layer.
    captures: dict[int, dict] = {n: {} for n in args.layers}
    handles = []
    for n in args.layers:
        attn = text_model.layers[n].self_attn
        def mk_q(n=n):
            return lambda mod, inp, out: captures[n].__setitem__("q", out.detach())
        def mk_k(n=n):
            return lambda mod, inp, out: captures[n].__setitem__("k", out.detach())
        handles.append(attn.q_proj.register_forward_hook(mk_q()))
        handles.append(attn.k_proj.register_forward_hook(mk_k()))

    # Pi0.5 needs all image features the policy expects + a state vector.
    # For the bare attention probe we use the same camera frame for all
    # camera slots (the prefix will have N×256 image patches where N is
    # the number of image features expected by the policy).
    img_feature_keys = list(policy.config.image_features.keys())
    state_dim = policy.config.max_state_dim
    print(f"[load] image_features={img_feature_keys}", flush=True)
    print(f"[load] state_dim={state_dim} (padded to max_state_dim)", flush=True)

    summary = []
    for short, prompt in PROMPTS.items():
        policy.reset()
        for n in args.layers:
            captures[n].clear()

        batch = {
            "observation.state": torch.zeros(state_dim, dtype=torch.float32, device=args.device).unsqueeze(0),
            "task": prompt,
        }
        # Use the same 224×224 frame for every camera slot the policy expects.
        for k in img_feature_keys:
            batch[k] = frame_224.to(args.device).unsqueeze(0)
        batch = preprocessor(batch)

        with torch.inference_mode():
            _ = policy.select_action(batch)

        # Locate name tokens in the language sub-sequence.
        ids_s, ids_e, n_real = _find_token_positions(tokenizer, prompt, NAME_PHRASE[short])
        attn_mask = batch["observation.language.attention_mask"][0].cpu().bool().tolist()
        n_used = sum(attn_mask)
        pad_left = 0
        if n_used != n_real:
            for i, a in enumerate(attn_mask):
                if a:
                    pad_left = i
                    break

        # The Pi0.5 prefix = [img patches × #cameras, lang tokens]. camera1
        # is first in `image_features`, so its patches are positions 0..255.
        n_imgs = len(img_feature_keys)
        lang_start = n_imgs * NUM_PATCHES
        name_idx_start = lang_start + pad_left + ids_s
        name_idx_end = lang_start + pad_left + ids_e

        for n in args.layers:
            q = captures[n].get("q")
            k = captures[n].get("k")
            if q is None or k is None:
                print(f"[warn] layer {n}: hook didn't fire for {short}", flush=True)
                continue
            B, L, _ = q.shape
            q = q.float().view(B, L, n_heads, head_dim).transpose(1, 2)
            k = k.float().view(B, L, n_kv_heads, head_dim).transpose(1, 2)
            if n_kv_heads != n_heads:
                k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            attn_avg = attn.mean(dim=1)  # (B, L, L)

            # Attention from name-token rows to camera1 patch cols (first 256).
            row = attn_avg[0, name_idx_start:name_idx_end, :NUM_PATCHES].mean(dim=0)
            grid = row.cpu().numpy().reshape(GRID, GRID)
            ar, ac = divmod(int(row.argmax().item()), GRID)
            entropy = float(-(row * row.clamp(min=1e-12).log()).sum().item())
            # Save heatmap + overlay.
            hm_t = torch.from_numpy(grid)[None, None].float()
            hm_up = F.interpolate(hm_t, size=(IMG_HW_TARGET, IMG_HW_TARGET),
                                  mode="bilinear", align_corners=False)[0, 0].numpy()
            hm_norm = (hm_up - hm_up.min()) / (hm_up.max() - hm_up.min() + 1e-9)
            Image.fromarray((hm_norm * 255).astype(np.uint8)).save(
                outdir / f"{short}_layer{n:02d}_heatmap.png"
            )
            _heatmap_overlay(frame_224, grid).save(outdir / f"{short}_layer{n:02d}_overlay.png")

            print(f"[probe] {short:5s} layer {n:2d}  argmax=({ar:2d},{ac:2d})  "
                  f"max_attn={grid.max():.4f}  entropy={entropy:.3f}", flush=True)
            summary.append((short, n, ar, ac, float(grid.max()), entropy))

    for h in handles:
        h.remove()

    print("\n=== SUMMARY (vanilla Pi0.5 — no training) ===")
    print(f"{'celeb':<6} {'layer':<6} {'argmax(r,c)':<14} {'max_attn':<10} {'entropy':<8}")
    for short, n, ar, ac, mx, h in summary:
        print(f"{short:<6} {n:<6} ({ar:2d},{ac:2d}){'':<8} {mx:<10.4f} {h:<8.3f}")

    # Gating criterion: does argmax DIFFER across the 3 prompts at any layer?
    # If yes → cross-modal attention is reachable; M2+KLAL plan is viable.
    # If no → vanilla PaliGemma is sink-locked; reconsider.
    print("\n=== GATING DECISION ===")
    all_argmaxes = set((s, n, ar, ac) for s, n, ar, ac, _, _ in summary)
    by_layer = {}
    for short, n, ar, ac, _, _ in summary:
        by_layer.setdefault(n, set()).add((ar, ac))
    layers_with_movement = [n for n, pts in by_layer.items() if len(pts) > 1]
    print(f"Layers where argmax differs across prompts: {sorted(layers_with_movement) or 'NONE'}")
    if layers_with_movement:
        print("PASS — Pi0.5 already has some cross-modal attention to build on.")
        print("Plan is viable: port M2 + KLAL on top.")
    else:
        print("CONCERN — argmax constant across prompts at every layer.")
        print("Vanilla PaliGemma may have the same sink failure as SmolVLM2.")
        print("Reconsider plan: VQA pretraining (M5-style) may be needed first.")
    print(f"\nHeatmaps + overlays saved under: {outdir}")
    print(f"Pull locally with:  scp -r <brev>:{outdir} ./")
    return 0


if __name__ == "__main__":
    sys.exit(main())

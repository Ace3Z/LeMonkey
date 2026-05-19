#!/usr/bin/env python3
"""Visualize the self-attention map from the celeb-name tokens to the
camera1 image patches, at several VLM layers, for each of the 3 celeb
prompts.

Why we need this: sanity_checks.py confirmed the policy reads the prompt
(actions differ per prompt) and Strix rollout still failed, which is the
'language pathway off-axis' failure mode the reviewer flagged: language
IS read but does not steer visual attention to the right face.

Method
------
1. Pick a real frame (with 3 visible printouts).
2. Hook `q_proj` and `k_proj` on each target VLM layer.
3. Forward with prompt `Place the coke on <celeb>.` once per celeb
   (calling `policy.reset()` between prompts to bypass SmolVLA's
   action-chunk cache).
4. Compute attention scores manually:
        A = softmax( Q @ K.T / sqrt(d_head) )   # without RoPE
   Note we omit RoPE — the relative positions of name token vs image
   patches are large (~130 tokens apart) so RoPE adds noise to the
   content-based attention we want to inspect. Without RoPE we see the
   pure semantic attention.
5. Average over heads, slice rows = name-token positions, columns =
   first 64 camera1 patches, reshape to 8x8.
6. Save:
   - {outdir}/input.png — the camera1 frame
   - {outdir}/{prompt}_layer{N}_heatmap.png — 8x8 upsampled
   - {outdir}/{prompt}_layer{N}_overlay.png — heatmap blended on input

Usage:
    python eval_3/scripts/attention_map_probe.py --revision step-10000 \\
        --layers 9 11 13 15 --out /tmp/attn_probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aug"))
from smolvlm_inference_patch import apply as _apply_smolvlm_patch  # noqa: E402
_apply_smolvlm_patch()


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
GRID_H = GRID_W = 8
IMG_H, IMG_W = 480, 640


def _find_token_positions(tokenizer, full_prompt: str, name_phrase: str):
    full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    for variant in (name_phrase, " " + name_phrase):
        name_ids = tokenizer.encode(variant, add_special_tokens=False)
        for i in range(len(full_ids) - len(name_ids) + 1):
            if full_ids[i : i + len(name_ids)] == name_ids:
                return i, i + len(name_ids), len(full_ids)
    raise ValueError(f"Could not locate {name_phrase!r} in tokenized {full_prompt!r}")


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


def _heatmap_overlay(image_chw_01: torch.Tensor, heatmap_8x8: np.ndarray,
                     alpha: float = 0.45) -> Image.Image:
    """Overlay a heatmap on an image. heatmap normalized to [0,1] internally."""
    img_np = (image_chw_01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hm = heatmap_8x8.astype(np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    # Upsample 8x8 → 480x640 with bilinear.
    hm_up = torch.from_numpy(hm)[None, None].float()
    hm_up = F.interpolate(hm_up, size=(IMG_H, IMG_W), mode="bilinear",
                          align_corners=False)[0, 0].numpy()
    # Make a red-orange heatmap (R=hm, G=hm*0.5, B=0).
    color = np.stack([hm_up, hm_up * 0.4, np.zeros_like(hm_up)], axis=-1)
    color = (color * 255).astype(np.uint8)
    blended = (img_np * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="HBOrtiz/smolvla_eval3_track_D_m2_mahbod")
    p.add_argument("--revision", default="step-10000")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset-root", default=str(Path.home() / ".cache/huggingface/lerobot/HBOrtiz/so101_eval3_track3_v3_baseline"))
    p.add_argument("--layers", nargs="+", type=int, default=[9, 11, 13, 15])
    p.add_argument("--episode", type=int, default=100, help="dataset episode index for the test frame")
    p.add_argument("--frame", type=int, default=10, help="frame within the episode")
    p.add_argument("--out", default=str(Path.home() / "eval3_attention_probe"))
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load policy + preprocessor.
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline
    print(f"[load] {args.repo}@{args.revision}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.repo, revision=args.revision).to(args.device).eval()
    preprocessor = DataProcessorPipeline.from_pretrained(
        args.repo, config_filename="policy_preprocessor.json", revision=args.revision
    )
    tokenizer = None
    for step in preprocessor.steps:
        if getattr(step, "input_tokenizer", None) is not None:
            tokenizer = step.input_tokenizer
            break
    assert tokenizer is not None

    # Locate VLM text_model.
    vlm_root = policy.model.vlm_with_expert.vlm.model
    text_model = vlm_root.text_model if hasattr(vlm_root, "text_model") else vlm_root
    n_layers = len(text_model.layers)
    print(f"[load] text_model has {n_layers} layers; probing layers {args.layers}", flush=True)

    # Test frame.
    image = _grab_real_frame(Path(args.dataset_root), args.episode, args.frame)
    state = torch.zeros(6, dtype=torch.float32)
    Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(outdir / "input.png")
    print(f"[frame] saved input frame → {outdir/'input.png'}", flush=True)

    # Per-layer hooks on q_proj and k_proj. They emit (B, L, n_heads*head_dim).
    captures: dict[int, dict] = {n: {} for n in args.layers}
    handles = []
    for n in args.layers:
        layer = text_model.layers[n]
        attn = layer.self_attn
        def mk_q(n=n):
            def hook(mod, inp, out):
                captures[n]["q"] = out.detach()
            return hook
        def mk_k(n=n):
            def hook(mod, inp, out):
                captures[n]["k"] = out.detach()
            return hook
        handles.append(attn.q_proj.register_forward_hook(mk_q()))
        handles.append(attn.k_proj.register_forward_hook(mk_k()))

    head_dim = text_model.layers[args.layers[0]].self_attn.head_dim
    n_heads = text_model.layers[args.layers[0]].self_attn.config.num_attention_heads
    n_kv_heads = text_model.layers[args.layers[0]].self_attn.config.num_key_value_heads
    print(f"[load] head_dim={head_dim} n_heads={n_heads} n_kv_heads={n_kv_heads}", flush=True)

    # For each prompt, forward + compute attention.
    summary = []
    for short, prompt in PROMPTS.items():
        policy.reset()
        for n in args.layers:
            captures[n].clear()

        batch = {
            "observation.images.camera1": image.to(args.device).unsqueeze(0),
            "observation.state": state.to(args.device).unsqueeze(0),
            "task": prompt,
        }
        batch = preprocessor(batch)
        with torch.inference_mode():
            _ = policy.select_action(batch)

        # Locate name tokens.
        ids_s, ids_e, n_real = _find_token_positions(tokenizer, prompt, NAME_PHRASE[short])
        attn_mask = batch["observation.language.attention_mask"][0].cpu().bool().tolist()
        n_used = sum(attn_mask)
        pad_left = 0
        if n_used != n_real:
            for i, a in enumerate(attn_mask):
                if a:
                    pad_left = i
                    break
        L_lang = batch["observation.language.tokens"].shape[-1]
        lang_start = 128  # 64 cam1 + 64 empty_cam patches
        name_idx_start = lang_start + pad_left + ids_s
        name_idx_end = lang_start + pad_left + ids_e

        for n in args.layers:
            q = captures[n].get("q")
            k = captures[n].get("k")
            if q is None or k is None:
                print(f"[warn] layer {n}: hook didn't capture for prompt {short}; skipping", flush=True)
                continue
            B, L, _ = q.shape
            q = q.float().view(B, L, n_heads, head_dim).transpose(1, 2)  # (B, H, L, D)
            k = k.float().view(B, L, n_kv_heads, head_dim).transpose(1, 2)
            if n_kv_heads != n_heads:
                k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # (B,H,L,L)
            attn = torch.softmax(scores, dim=-1)  # (B,H,L,L)
            attn_avg = attn.mean(dim=1)  # (B,L,L)

            # Rows for name tokens; columns for camera1 patches.
            attn_name = attn_avg[0, name_idx_start:name_idx_end, :64].mean(dim=0)  # (64,)
            grid = attn_name.cpu().numpy().reshape(GRID_H, GRID_W)

            # Save raw heatmap (8x8 upsampled bilinear).
            hm_t = torch.from_numpy(grid)[None, None].float()
            hm_up = F.interpolate(hm_t, size=(IMG_H, IMG_W), mode="bilinear",
                                  align_corners=False)[0, 0].numpy()
            hm_norm = (hm_up - hm_up.min()) / (hm_up.max() - hm_up.min() + 1e-9)
            Image.fromarray((hm_norm * 255).astype(np.uint8)).save(
                outdir / f"{short}_layer{n:02d}_heatmap.png"
            )
            # Save overlay.
            _heatmap_overlay(image, grid).save(outdir / f"{short}_layer{n:02d}_overlay.png")

            # Quantitative summary.
            argmax_flat = int(grid.argmax())
            ar, ac = divmod(argmax_flat, GRID_W)
            entropy = float(-(attn_name * (attn_name.clamp(min=1e-12)).log()).sum().item())
            summary.append((short, n, ar, ac, float(grid.max()), entropy))
            print(f"[probe] {short:5s} layer {n:2d}  argmax=({ar},{ac})  "
                  f"max_attn={grid.max():.4f}  entropy={entropy:.3f}", flush=True)

    for h in handles:
        h.remove()

    # Concise summary table.
    print("\n=== SUMMARY (attention from celeb-name tokens to camera1 patches) ===")
    print(f"{'celeb':<6} {'layer':<6} {'argmax(r,c)':<12} {'max_attn':<10} {'entropy':<8}")
    for short, n, ar, ac, mx, h in summary:
        print(f"{short:<6} {n:<6} ({ar},{ac}){'':<6} {mx:<10.4f} {h:<8.3f}")

    print(f"\nHeatmaps + overlays saved under: {outdir}")
    print("Pull them locally with:")
    print(f"    scp -r time2sleep:{outdir}  ./")
    return 0


if __name__ == "__main__":
    sys.exit(main())

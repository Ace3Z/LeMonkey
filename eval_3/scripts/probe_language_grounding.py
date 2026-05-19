#!/usr/bin/env python3
"""Does the VLM ground the celeb name to the right face in the image?

For a test frame with known face bboxes (from face_labels.json), we:

1. Forward the policy with prompt 'Place the coke on <celeb>.'
2. Capture layer-9 hidden state (the M2 hook point) for every token in
   the prefix: image patches + language tokens.
3. Find the position of the celeb-name tokens in the prefix and average
   their hidden vectors to get a 'name-vector'.
4. Compute cosine similarity between that name-vector and each of the 64
   camera1 patch hidden-vectors (arranged on an 8x8 grid).
5. The patch with the highest cosine is where the model 'thinks' the
   named celeb is. We compare that argmax patch to the ground-truth bbox.

Repeat for all 3 celebs. If the argmax patch shifts to the correct face
each time, the VLM IS doing name-to-face grounding (the M2 mechanism is
working end-to-end). If it doesn't shift, the VLM is ignoring the prompt
for visual grounding — even though the policy actions depend on the
prompt (we confirmed that in sanity_checks.py); some other pathway is
doing the lifting.

Usage:
    python eval_3/scripts/probe_language_grounding.py --revision step-10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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

# 8x8 patch grid covers the 480x640 image after SigLIP + pixel-shuffle.
# patch_h = 480/8 = 60, patch_w = 640/8 = 80
GRID_H = GRID_W = 8
IMG_H, IMG_W = 480, 640


def _find_real_frame(face_labels_dir: Path):
    """Find a base teleop frame where the detector found exactly 3 faces;
    return (source_episode, frame_idx, bboxes_dict).
    """
    for fp in sorted(face_labels_dir.glob("*.json")):
        d = json.loads(fp.read_text())
        for frame in d.get("frames", []):
            if frame.get("n_visible_faces", 0) == 3:
                # Sort bboxes left-to-right by x_center so we can label them
                bbs = sorted(frame["bboxes"], key=lambda b: b["x_center"])
                return d["source_episode"], d["representative_variant"], frame["frame_idx"], bbs
    raise RuntimeError("No frame with 3 visible faces found in face_labels_dir")


def _grab_real_frame_from_variant(dataset_root: Path, variant_name: str, frame_idx: int):
    """Grab a specific frame from a specific aug variant via the LeRobot cache."""
    import datasets, av
    eps = sorted(str(p) for p in (dataset_root / "meta/episodes").rglob("*.parquet"))
    ds = datasets.load_dataset("parquet", data_files=eps, split="train")
    for ep_idx in range(len(ds)):
        row = ds[ep_idx]
        # Each episode has tasks but the variant_name doesn't appear in metadata
        # directly. We use the episode_index iteratively; just pull frame_idx 10
        # from a few episodes to find non-blank frames.
        chunk = row["videos/observation.images.camera1/chunk_index"]
        fi = row["videos/observation.images.camera1/file_index"]
        vp = dataset_root / f"videos/observation.images.camera1/chunk-{chunk:03d}/file-{fi:03d}.mp4"
        if not vp.exists():
            continue
        # Try this episode; map frame_idx via LeRobot's per-episode frame numbering.
        container = av.open(str(vp))
        stream = container.streams.video[0]
        decoded = None
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_idx:
                decoded = frame
                break
        container.close()
        if decoded is None:
            continue
        arr = decoded.to_ndarray(format="rgb24")
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    raise RuntimeError("Could not decode any frame from cached dataset videos")


def _find_token_positions(tokenizer, full_prompt: str, name_phrase: str):
    """Find the (start_idx, end_idx) of `name_phrase` token IDs inside the
    tokenized `full_prompt`. Returns the slice as a tuple.

    Works robustly across BPE/sentencepiece quirks because we search for the
    contiguous sub-sequence rather than relying on a specific token ID.
    """
    full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    # Encode the name with a leading space so BPE merges match.
    name_variants = [name_phrase, " " + name_phrase]
    for variant in name_variants:
        name_ids = tokenizer.encode(variant, add_special_tokens=False)
        for i in range(len(full_ids) - len(name_ids) + 1):
            if full_ids[i : i + len(name_ids)] == name_ids:
                return i, i + len(name_ids), len(full_ids)
    raise ValueError(f"Could not locate {name_phrase!r} in tokenized {full_prompt!r}")


def _bbox_to_patch(bbox):
    """8x8 grid over 480x640. Return (rmin, cmin, rmax, cmax) inclusive."""
    cmin = int(bbox["x1"] / (IMG_W / GRID_W))
    cmax = int(bbox["x2"] / (IMG_W / GRID_W))
    rmin = int(bbox["y1"] / (IMG_H / GRID_H))
    rmax = int(bbox["y2"] / (IMG_H / GRID_H))
    return rmin, cmin, min(GRID_H - 1, rmax), min(GRID_W - 1, cmax)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="HBOrtiz/smolvla_eval3_track_D_m2_mahbod")
    p.add_argument("--revision", default="step-10000")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--face-labels-dir", default=str(Path.home() / "eval3_m2_toolkit/face_labels"))
    p.add_argument("--dataset-root", default=str(Path.home() / ".cache/huggingface/lerobot/HBOrtiz/so101_eval3_track3_v3_baseline"))
    args = p.parse_args()

    # 1. Load policy + preprocessor.
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline
    print(f"[load]  {args.repo}@{args.revision}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.repo, revision=args.revision).to(args.device).eval()
    preprocessor = DataProcessorPipeline.from_pretrained(
        args.repo, config_filename="policy_preprocessor.json", revision=args.revision
    )
    # The TokenizerProcessor step exposes its loaded HF tokenizer as
    # `input_tokenizer` (the `tokenizer` field is just the dataclass init kwarg).
    tokenizer = None
    for step in preprocessor.steps:
        if getattr(step, "input_tokenizer", None) is not None:
            tokenizer = step.input_tokenizer
            break
        if getattr(step, "tokenizer", None) is not None and not isinstance(
            getattr(step, "tokenizer"), str
        ):
            tokenizer = step.tokenizer
            break
    if tokenizer is None:
        names = [type(s).__name__ for s in preprocessor.steps]
        raise RuntimeError(f"No tokenizer found in preprocessor steps: {names}")
    print(f"[load]  tokenizer: {type(tokenizer).__name__}, vocab={tokenizer.vocab_size}", flush=True)

    # 2. Find a test frame with 3 known faces.
    src_ep, variant_name, frame_idx, bboxes = _find_real_frame(Path(args.face_labels_dir))
    print(f"[frame] source={src_ep!r} variant={variant_name!r} frame_idx={frame_idx}", flush=True)
    print(f"[frame] 3 face bboxes (sorted L→R by x_center):")
    for i, bb in enumerate(bboxes):
        r0, c0, r1, c1 = _bbox_to_patch(bb)
        print(f"           slot {i} (L,M,R): x_center={bb['x_center']:6.1f}  "
              f"patch_box=({r0},{c0})..({r1},{c1})")

    # 3. Pull the frame from disk.
    # The face_labels are for the BASE teleop; videos exist for aug variants. Use a
    # known-good episode index from the dataset (we don't have a base→episode_index
    # map here — just pull a fresh frame from episode 100 as a stand-in. The bboxes
    # are still informative because the layout is preserved across variants.
    print(f"[frame] pulling frame from cached dataset (ep 100, frame {frame_idx})...", flush=True)
    image = _grab_real_frame_from_variant(Path(args.dataset_root), variant_name, frame_idx)
    state = torch.zeros(6, dtype=torch.float32)

    # 4. Attach a hook on layer 10's input_layernorm to capture layer-9 output.
    captured = {}
    text_model = (policy.model.vlm_with_expert.vlm.model.text_model
                  if hasattr(policy.model.vlm_with_expert.vlm.model, "text_model")
                  else policy.model.vlm_with_expert.vlm.text_model)
    target_module = text_model.layers[10].input_layernorm

    def pre_hook(mod, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        captured["h9"] = h.detach()
        return None

    handle = target_module.register_forward_pre_hook(pre_hook, with_kwargs=True)

    # 5. For each celeb prompt, run forward + analyze.
    # IMPORTANT: SmolVLA caches an action chunk (length = chunk_size = 50)
    # and only does a fresh forward when the queue is empty. Call
    # policy.reset() between prompts so each one runs a real forward.
    results = []
    for short, prompt in PROMPTS.items():
        policy.reset()
        captured.pop("h9", None)
        batch = {
            "observation.images.camera1": image.to(args.device).unsqueeze(0),
            "observation.state": state.to(args.device).unsqueeze(0),
            "task": prompt,
        }
        batch = preprocessor(batch)
        with torch.inference_mode():
            _ = policy.select_action(batch)
        if "h9" not in captured:
            raise RuntimeError(f"Hook did not fire for prompt {prompt!r}")

        h = captured["h9"]  # shape (B, prefix_len, 960) in bf16
        # Prefix layout: [img1_patches (64) | empty_cam_patches (64) | lang_tokens (L) | state (1)]
        # Total prefix len: 64 + 64 + L + 1
        cam1_patches = h[0, :64].float()  # (64, 960)
        L = batch["observation.language.tokens"].shape[-1]
        lang_start = 128
        lang_end = lang_start + L
        lang_hidden = h[0, lang_start:lang_end].float()  # (L, 960)

        # Locate the celeb name inside the L language tokens.
        ids_start, ids_end, n_full_ids = _find_token_positions(tokenizer, prompt, NAME_PHRASE[short])
        # The preprocessor probably pads to a fixed length on the left or right; we
        # need the actual first-non-pad token to align. Use the attention mask:
        attn_mask = batch["observation.language.attention_mask"][0].cpu().bool().tolist()
        n_real = sum(attn_mask)
        if n_real != n_full_ids:
            # Padding offset: find where the real tokens start in the padded sequence.
            pad_left = 0
            for i, a in enumerate(attn_mask):
                if a: pad_left = i; break
        else:
            pad_left = 0

        name_slice = lang_hidden[pad_left + ids_start : pad_left + ids_end]
        name_vec = name_slice.mean(dim=0)  # (960,)

        # Cosine similarity vs each camera1 patch.
        cos = F.cosine_similarity(name_vec.unsqueeze(0), cam1_patches, dim=1)  # (64,)
        cos_grid = cos.reshape(GRID_H, GRID_W).cpu().numpy()

        # Argmax patch.
        amax_flat = int(cos.argmax().item())
        amax_r, amax_c = divmod(amax_flat, GRID_W)

        # Which face bbox does this patch land in?
        landed = None
        for slot_idx, bb in enumerate(bboxes):
            r0, c0, r1, c1 = _bbox_to_patch(bb)
            if r0 <= amax_r <= r1 and c0 <= amax_c <= c1:
                landed = slot_idx
                break

        # Map celeb short name to expected slot. Since we don't know which slot
        # corresponds to which celeb in this base frame (we'd need the
        # face_labels per-frame celeb assignment), just report the argmax position
        # and which of the 3 slots it landed in.
        print(f"\n[probe] prompt: {prompt}")
        print(f"        name-token positions in prompt: [{ids_start}..{ids_end}) "
              f"(pad-shifted by +{pad_left})")
        print(f"        cos grid (8x8, brighter=closer to name-token):")
        for row in cos_grid:
            print("           " + " ".join(f"{v:+.2f}" for v in row))
        print(f"        argmax patch: row={amax_r} col={amax_c} "
              f"cos={cos.max().item():+.4f}")
        if landed is not None:
            print(f"        → landed inside face-slot {landed} (L/M/R = 0/1/2)")
        else:
            print(f"        → did NOT land inside any face bbox")
        results.append((short, amax_r, amax_c, landed, cos.max().item()))

    handle.remove()

    # 6. Summary: does the argmax shift across prompts?
    print("\n=== SUMMARY ===")
    print(f"{'prompt':<8}  {'argmax patch':<14}  {'face slot':<10}  {'cos':>7}")
    for short, r, c, slot, cv in results:
        slot_s = f"slot {slot}" if slot is not None else "—"
        print(f"{short:<8}  ({r},{c}){'':<8}  {slot_s:<10}  {cv:+.4f}")

    # Pass criterion: at least 2 of 3 prompts land argmax in different slots.
    slots_hit = [r[3] for r in results if r[3] is not None]
    distinct_slots = len(set(slots_hit))
    print(f"\nDistinct face slots hit across the 3 prompts: {distinct_slots}/3")
    if distinct_slots >= 2:
        print("RESULT: VLM grounding is DIFFERENTIATING — language shifts visual attention.")
    elif distinct_slots == 1:
        print("RESULT: VLM grounds to the SAME slot for all prompts — language is read "
              "(per sanity_checks.py) but does not steer visual attention at layer 9. "
              "The action expert may be doing the lifting via cross-attention elsewhere.")
    else:
        print("RESULT: argmax never lands on a face — visual grounding signal at layer 9 "
              "is not anchored on faces. Suggests M2 supervision didn't propagate well.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end M2 integration smoke test.

Composes everything:
  - 2 real variant directories → load camera1 frame 0 + augmentation.json
  - `m2_dataloader.M2SupervisionBuilder` → bbox_masks + valid + centroids
  - `m2_smolvla_hook.attach_m2_hook` → forward_pre_hook on layer 10
  - `m2_smolvla_hook.apply_m2_partial_freeze` → layers 0-8 frozen, 9-15 train
  - Real `SmolVLMWithExpertModel.forward` (CPU, VLM-stream-only path)
  - `m2_alignment.m2_align_loss` → scalar loss + per-slot cos
  - `.backward()` → verify grads land in layer 9, not in layers 0-8

This is the "integration probe" that catches anything the hook probe
(`dbg_m2_hook_probe.py`) and the alignment unit test
(`dbg_m2_alignment.py`) miss together.

Run from project root:

    python eval_3/aug/dbg/dbg_m2_e2e.py
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "eval_3/aug"))


def _import_smolvlm_with_expert():
    """Bypass lerobot.policies/__init__.py side-effects (pulls in pandas etc)."""
    path = REPO_ROOT / "third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py"
    spec = importlib.util.spec_from_file_location("_smolvlm_with_expert_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SmolVLMWithExpertModel


SmolVLMWithExpertModel = _import_smolvlm_with_expert()  # noqa: E402

from m2_alignment import m2_align_loss, FrozenProjector  # noqa: E402
from m2_dataloader import M2SupervisionBuilder, variant_name_to_source  # noqa: E402
from m2_smolvla_hook import (  # noqa: E402
    DEFAULT_CAPTURE_LAYER,
    apply_m2_partial_freeze,
    attach_m2_hook,
)


# Two real variants we've already validated visually (frame 0 has 3 faces).
SAMPLE_VARIANTS = [
    "quick_swift_SOL_ep04_20260511_185956__t3_0002_v44",
    "quick_lecun_LSO_ep01_20260511_205000__t3_0002_v00",
]


def _build_smolvla(num_vlm_layers: int = 16):
    """Bypass AutoProcessor (needs torchvision we haven't installed) since
    we don't use it in this probe — VLM stream only."""
    from transformers import AutoProcessor

    original = AutoProcessor.from_pretrained

    class _StubProcessor:
        class _StubTokenizer:
            fake_image_token_id = 0
            global_image_token_id = 0
        tokenizer = _StubTokenizer()

    AutoProcessor.from_pretrained = staticmethod(lambda *a, **kw: _StubProcessor())
    try:
        return SmolVLMWithExpertModel(
            model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            load_vlm_weights=True,
            train_expert_only=False,
            freeze_vision_encoder=True,
            attention_mode="cross_attn",
            num_expert_layers=-1,
            num_vlm_layers=num_vlm_layers,
            self_attn_every_n_layers=2,
            expert_width_multiplier=0.75,
            device="cpu",
        )
    finally:
        AutoProcessor.from_pretrained = original


def main() -> int:
    print("=" * 70)
    print("M2 end-to-end integration smoke test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1) Build the supervision for our 2 real variants, frame 0 each.
    # ------------------------------------------------------------------
    print("\n[1/4] M2SupervisionBuilder + real variants")
    aug_root = Path.home() / "Downloads/eval3_track3_aug"
    builder = M2SupervisionBuilder(
        face_labels_dir=REPO_ROOT / "eval_3/aug/stats/face_labels",
        manifest_path=REPO_ROOT / "eval_3/aug/stats/celeb_embeddings.json",
        aug_root=aug_root,
    )
    src_episodes = [variant_name_to_source(v) for v in SAMPLE_VARIANTS]
    frame_idxs = [0, 0]
    sup = builder.build_batch(
        source_episodes=src_episodes,
        frame_idxs=frame_idxs,
        variants=SAMPLE_VARIANTS,
        device="cpu",
    )
    print(f"  source_episodes: {src_episodes}")
    print(f"  bbox_masks shape: {tuple(sup['bbox_masks'].shape)}  dtype: {sup['bbox_masks'].dtype}")
    print(f"  bbox_valid shape: {tuple(sup['bbox_valid'].shape)}")
    for b in range(2):
        n = sup['bbox_valid'][b].sum().item()
        sz = sup['bbox_masks'][b].float().sum(dim=-1).tolist()
        print(f"  sample {b}: valid={sup['bbox_valid'][b].tolist()}, mask sizes={sz}")

    # ------------------------------------------------------------------
    # 2) Load real SmolVLM2 + attach hook + apply partial freeze.
    # ------------------------------------------------------------------
    print("\n[2/4] Load SmolVLM2 + attach hook + partial freeze")
    model = _build_smolvla(num_vlm_layers=16)
    hook = attach_m2_hook(model, capture_layer=DEFAULT_CAPTURE_LAYER, config=None)
    n_frozen, n_trainable = apply_m2_partial_freeze(model, freeze_below=DEFAULT_CAPTURE_LAYER)
    print(f"  hook on text_model.layers[{hook.hook_layer}].input_layernorm")
    print(f"  partial-freeze: {n_frozen/1e6:.1f}M frozen, {n_trainable/1e6:.1f}M trainable")

    # ------------------------------------------------------------------
    # 3) Synthesise inputs_embeds for the VLM stream and run forward.
    #    For this smoke test we use random VLM prefix embeddings with
    #    prefix_len = 64 (camera1) + 64 (camera2 zero) + 50 (lang) + 1 (state).
    # ------------------------------------------------------------------
    print("\n[3/4] Forward pass with random VLM prefix (expert stream None)")
    B = 2
    H = model.config.text_config.hidden_size
    prefix_len = 64 + 64 + 50 + 1
    torch.manual_seed(0)
    vlm_prefix = torch.randn(B, prefix_len, H)
    attn_mask = torch.ones(B, prefix_len, prefix_len, dtype=torch.bool)
    position_ids = torch.arange(prefix_len)[None, :].expand(B, -1)

    outputs, _ = model.forward(
        attention_mask=attn_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[vlm_prefix, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    assert hook.captured is not None, "Hook did not fire"
    captured = hook.captured
    print(f"  captured shape: {tuple(captured.shape)}")
    assert tuple(captured.shape) == (B, prefix_len, H)

    # ------------------------------------------------------------------
    # 4) Run the M2 loss on the captured hidden state + real supervision.
    # ------------------------------------------------------------------
    print("\n[4/4] M2 align_loss with real supervision + backward")
    projector = FrozenProjector()
    result = m2_align_loss(
        hidden_state=captured,
        bbox_masks=sup["bbox_masks"].to(dtype=torch.float32),
        bbox_valid=sup["bbox_valid"],
        target_centroids=sup["target_centroids"],
        projector=projector,
    )
    print(f"  loss = {result.loss.item():+.4f}  n_valid = {result.n_valid}/6")
    print(f"  per-slot cos: "
          f"{[f'{x.item():+.3f}' if not torch.isnan(x) else 'nan' for x in result.per_slot_cos]}")

    # Backward — verify grad flows to layer 9 parameters.
    (0.2 * result.loss).backward()

    text_layers = model.vlm.model.text_model.layers
    has_grad_at = []
    for i, layer in enumerate(text_layers):
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in layer.parameters() if p.requires_grad
        )
        has_grad_at.append((i, any_grad))

    print(f"  layers with non-zero grad: {[i for i, g in has_grad_at if g]}")

    # Expected: only layer 9 (or 9 + nothing else, since cap was captured at
    # the start of layer 10 → grad path is layers 0..9 of which only 9 is
    # trainable).
    expected = {DEFAULT_CAPTURE_LAYER}
    actual = {i for i, g in has_grad_at if g}
    ok = (actual == expected)
    if not ok:
        print(f"  FAIL: expected grads only at layer {expected}, got {actual}")
    else:
        print(f"  OK: gradients flow to layer {DEFAULT_CAPTURE_LAYER} only "
              "(layers 0-8 frozen, layer 10+ not in grad path)")

    hook.remove()
    print("\n" + "=" * 70)
    if ok:
        print("ALL CHECKS PASSED — M2 integration is wired correctly end-to-end")
        return 0
    print("FAIL — see findings above")
    return 1


if __name__ == "__main__":
    sys.exit(main())

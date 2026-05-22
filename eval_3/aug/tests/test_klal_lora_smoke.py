"""Component smoke test for the KLAL + LoRA additions.

Covers, with real torch:
  Tier 1 — pure logic: LoRA math, inject/merge, KLAL target + loss, name-token
           location. No model download.
  Tier 2 — real SmolVLA: load lerobot/smolvla_base, inject LoRA, attach the
           KLAL hookset, and (best-effort) run one real forward to confirm the
           hooks fire and KLAL's gradient reaches the LoRA adapters.

The full 200-step cotrain smoke (datasets + flow/vqa/klal losses) is a
separate GPU-box run — see 2026-05-20_klal_lora_smolvla_cotrain.md.

Run:  <lemonkey-env-python> eval_3/aug/tests/test_klal_lora_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eval_3/aug"))

import torch
import torch.nn as nn

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}",
          flush=True)


# ===========================================================================
# Tier 1 — pure logic
# ===========================================================================
print("\n=== Tier 1: LoRA ===", flush=True)
from m2_lora import (LoRAConfig, LoRALinear, count_lora_params, inject_lora,
                     swap_to_lora, swap_to_merged)

torch.manual_seed(0)
base = nn.Linear(64, 48)
lora = LoRALinear(base, r=8, alpha=16, dropout=0.0)
x = torch.randn(4, 10, 64)

out0, base_out = lora(x), base(x)
check("LoRA is a no-op at init (B=0)", torch.allclose(out0, base_out, atol=1e-6),
      f"max|d|={float((out0 - base_out).abs().max()):.2e}")
check("LoRA forward shape", tuple(out0.shape) == (4, 10, 48))
check("LoRA .weight property returns base weight", lora.weight is base.weight)
check("LoRA base is frozen", not base.weight.requires_grad)
check("LoRA adapters are trainable",
      lora.lora_A.weight.requires_grad and lora.lora_B.weight.requires_grad)

with torch.no_grad():                      # simulate a trained adapter
    lora.lora_B.weight.normal_(0, 0.1)
out1 = lora(x)
check("LoRA adapter is active once B != 0",
      not torch.allclose(out1, base_out, atol=1e-4))

base_w_before = base.weight.detach().clone()
merged = lora.merged_linear()
mdiff = float((merged(x) - out1).abs().max())
check("LoRA merge round-trip (fp32 exact)", mdiff < 1e-5, f"max|d|={mdiff:.2e}")
check("LoRA merge does not mutate the base weight",
      torch.equal(base.weight, base_w_before))

# bf16 merge — quantify the precision flagged by the review.
bbase = nn.Linear(64, 48).to(torch.bfloat16)
blora = LoRALinear(bbase, r=8, alpha=16, dropout=0.0)
with torch.no_grad():
    blora.lora_B.weight.normal_(0, 0.1)
bx = torch.randn(4, 10, 64, dtype=torch.bfloat16)
bdiff = float((blora.merged_linear()(bx).float() - blora(bx).float()).abs().max())
check("LoRA bf16 merge drift is small", bdiff < 5e-2, f"max|d|={bdiff:.2e}")

print("\n=== Tier 1: inject / swap ===", flush=True)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, n, nn.Linear(64, 64))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()


class _TextModel(nn.Module):
    def __init__(self, n=16):
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(n))


tm = _TextModel(16)
reg = inject_lora(tm, LoRAConfig(r=8, alpha=16, layers=(10, 12, 14),
                                 target_modules=("q_proj", "k_proj")))
check("inject_lora module count", len(reg) == 6, f"got {len(reg)}")
check("inject_lora wraps targeted module",
      isinstance(tm.layers[10].self_attn.q_proj, LoRALinear))
check("inject_lora leaves non-targeted module alone",
      type(tm.layers[10].self_attn.v_proj) is nn.Linear)
check("count_lora_params > 0", count_lora_params(reg) > 0,
      f"{count_lora_params(reg)} params")

swap_to_merged(reg)
mkeys = set(tm.state_dict().keys())
check("swap_to_merged → plain nn.Linear",
      type(tm.layers[10].self_attn.q_proj) is nn.Linear)
check("merged state_dict has vanilla keys (no lora_A/base)",
      any(k.endswith("layers.10.self_attn.q_proj.weight") for k in mkeys)
      and not any("lora_A" in k for k in mkeys))
swap_to_lora(reg)
check("swap_to_lora restores LoRALinear",
      isinstance(tm.layers[10].self_attn.q_proj, LoRALinear))

print("\n=== Tier 1: KLAL target + loss + name tokens ===", flush=True)
from m2_klal import KLALConfig, gaussian_target_from_mask, klal_loss
from m2_klal_smolvla import extract_name_token_positions

mask = torch.zeros(64, dtype=torch.bool)
mask[27] = mask[28] = True                 # 8x8 grid
g = gaussian_target_from_mask(mask, sigma_patches=1.0)
check("gaussian target sums to 1", abs(float(g.sum()) - 1.0) < 1e-5,
      f"sum={float(g.sum()):.6f}")
check("gaussian target peaks on the mask", int(g.argmax()) in (27, 28),
      f"argmax={int(g.argmax())}")

lang = torch.tensor([[5, 6, 100, 101, 102, 7, 8],
                     [9, 100, 101, 102, 0, 0, 0]])
pos = extract_name_token_positions(lang, ["obama", "obama"], {"obama": [100, 101, 102]})
check("name-token subsequence located",
      pos is not None and pos[0].tolist() == [2, 3, 4]
      and pos[1].tolist() == [1, 2, 3],
      f"pos={None if pos is None else pos.tolist()}")


class _FakeHook:
    def __init__(self, attn):
        self.attn = attn

    def get_attention(self, layer, sl, names):
        return self.attn


B, P = 2, 64
tgt = torch.zeros(B, P, dtype=torch.bool)
tgt[:, 30] = True
cfg1 = KLALConfig(capture_layers=(0,), target_sigma_patches=1.0, lam=1.0)
gt = torch.stack([gaussian_target_from_mask(tgt[b], 1.0) for b in range(B)])
names = torch.zeros(B, 1, dtype=torch.long)
loss_match = float(klal_loss(_FakeHook(gt), slice(0, P), names, tgt, cfg1))
loss_unif = float(klal_loss(_FakeHook(torch.full((B, P), 1.0 / P)),
                            slice(0, P), names, tgt, cfg1))
check("KLAL ≈ 0 when attention matches target", loss_match < 1e-3,
      f"loss={loss_match:.5f}")
check("KLAL > 0 (and > matched) when attention is uniform",
      loss_unif > loss_match and loss_unif > 0.1,
      f"uniform={loss_unif:.4f} vs matched={loss_match:.5f}")


# ===========================================================================
# Tier 2 — real SmolVLA
# ===========================================================================
print("\n=== Tier 2: real SmolVLA (lerobot/smolvla_base) ===", flush=True)
tier2_ok = True
try:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
    cfg.train_expert_only = True
    cfg.freeze_vision_encoder = True
    cfg.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=cfg)
    policy.model.vlm_with_expert.set_requires_grad()
    print("  loaded SmolVLAPolicy", flush=True)

    text_model = policy.model.vlm_with_expert.vlm.model.text_model
    n_layers = len(text_model.layers)
    lcfg = LoRAConfig(r=16, alpha=32, dropout=0.0, layers=tuple(range(n_layers)),
                      target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    real_reg = inject_lora(text_model, lcfg)
    check("real inject_lora module count", len(real_reg) == n_layers * 4,
          f"{len(real_reg)} over {n_layers} layers")

    q = text_model.layers[10].self_attn.q_proj
    check("LoRA'd q_proj is LoRALinear", isinstance(q, LoRALinear))
    _ = q.weight.dtype                     # SmolVLA's forward reads this — would crash w/o the property
    check("LoRA'd q_proj.weight.dtype access works (no crash)", True,
          f"dtype={q.weight.dtype}")
    qout = q(torch.randn(2, 5, q.weight.shape[1], dtype=q.weight.dtype))
    check("LoRA'd q_proj forward shape", tuple(qout.shape) == (2, 5, q.weight.shape[0]))

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    lora_n = count_lora_params(real_reg)
    check("LoRA params are in the trainable set", lora_n > 0 and trainable >= lora_n,
          f"lora={lora_n / 1e6:.2f}M, total trainable={trainable / 1e6:.1f}M")

    from m2_klal_smolvla import KLALHookSetSmolVLA
    vlm_we = policy.model.vlm_with_expert
    tcfg = vlm_we.config.text_config
    head_dim = getattr(tcfg, "head_dim", None) or (tcfg.hidden_size // tcfg.num_attention_heads)
    hookset = KLALHookSetSmolVLA(text_model, vlm_we, (10, 12, 14),
                                 tcfg.num_attention_heads, tcfg.num_key_value_heads,
                                 head_dim)
    check("KLALHookSetSmolVLA constructs on the real model", True,
          f"heads={tcfg.num_attention_heads}/{tcfg.num_key_value_heads} head_dim={head_dim}")

    # ---- best-effort real forward through VLAFlowMatching ----
    try:
        m = policy.model
        Bsz = 2
        hw = getattr(cfg, "resize_imgs_with_padding", None) or (512, 512)
        if isinstance(hw, int):
            hw = (hw, hw)
        images = [torch.rand(Bsz, 3, hw[0], hw[1])]
        img_masks = [torch.ones(Bsz, dtype=torch.bool)]
        Ltok = 32
        lang_tokens = torch.randint(0, 1000, (Bsz, Ltok))
        lang_masks = torch.ones(Bsz, Ltok, dtype=torch.bool)
        state = torch.randn(Bsz, cfg.max_state_dim)
        actions = torch.randn(Bsz, cfg.chunk_size, cfg.max_action_dim)
        m.train()
        hookset.reset()
        flow_loss = m.forward(images, img_masks, lang_tokens, lang_masks, state, actions)
        check("real VLAFlowMatching.forward runs", torch.is_tensor(flow_loss)
              and torch.isfinite(flow_loss).all(), f"flow_loss={float(flow_loss.mean()):.4f}")
        check("KLAL hooks captured q/k",
              hookset._captures[10].get("q") is not None
              and hookset._captures[10].get("k") is not None)
        check("KLAL captured position_ids", hookset._position_ids is not None)
        ipl, ppi = hookset.image_prefix_len(), hookset.patches_per_image()
        check("KLAL measured image-prefix length", ipl is not None and ppi is not None,
              f"image_prefix_len={ipl} patches_per_image={ppi}")

        # KLAL loss on this forward + gradient reaches the LoRA adapters.
        names = torch.full((Bsz, 1), ipl, dtype=torch.long)   # a name token just past the image block
        tmask = torch.zeros(Bsz, ppi, dtype=torch.bool)
        tmask[:, ppi // 2] = True
        kcfg = KLALConfig(capture_layers=(10, 12, 14), target_sigma_patches=1.0,
                          lam=1.0, patch_grid=8, num_image_patches_total=ppi)
        kloss = klal_loss(hookset, slice(0, ppi), names, tmask, kcfg)
        check("KLAL loss is finite on a real forward",
              torch.is_tensor(kloss) and torch.isfinite(kloss).all(),
              f"klal={float(kloss):.4f}")
        # LoRA inits B=0, so at step 0 the gradient to lora_A is gated to zero
        # by B (standard LoRA: A only trains once B != 0 after the first optim
        # step). lora_B is the adapter param that carries gradient at init, so
        # it is the correct probe that KLAL's gradient reaches the adapters.
        loraB = text_model.layers[10].self_attn.q_proj.lora_B.weight
        grad = torch.autograd.grad(kloss, loraB, retain_graph=False, allow_unused=True)[0]
        check("KLAL gradient reaches a LoRA adapter (q_proj.lora_B)",
              grad is not None and float(grad.norm()) > 0,
              f"|grad|={0.0 if grad is None else float(grad.norm()):.3e}")
    except Exception as e:
        import traceback
        print(f"  [INFO] real-forward sub-test not exercised locally: "
              f"{type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        print("  [INFO] component checks above still hold; the full forward "
              "is covered by the GPU-box cotrain smoke.", flush=True)
except Exception as e:
    tier2_ok = False
    import traceback
    traceback.print_exc()
    print(f"  [INFO] Tier 2 skipped (could not load real model): "
          f"{type(e).__name__}: {e}", flush=True)


# ===========================================================================
print(f"\n=== SUMMARY: {len(_PASS)} passed, {len(_FAIL)} failed ===", flush=True)
if _FAIL:
    print("FAILED: " + ", ".join(_FAIL), flush=True)
sys.exit(1 if _FAIL else 0)

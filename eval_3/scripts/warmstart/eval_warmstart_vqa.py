#!/usr/bin/env python3
"""Validate the PaliGemma VQA warm-start via N-way teacher-forced discrimination.

WHY NOT GENERATION
==================
lerobot's Pi0.5 port (`PaliGemmaForConditionalGenerationWithPiGemma` / `PiGemmaModel`)
was built for flow-matching ACTION inference. Its autoregressive `.generate()` text
path is untested and produces garbage tokens. So we do NOT call `.generate()`.

Instead we use TEACHER-FORCED SCORING — the exact forward path training uses
(verified working: the warm-start smoke test + 5.6h run both ran it). For an
(image, candidate_name) pair we feed `<image><prompt><name>` with `suffix=name`,
read the cross-entropy on just the name tokens. Lower CE = the model finds that
name more likely for that face.

THE TEST
========
N-way discrimination, which mirrors our actual eval task (closed-set: given a
name + visible portraits, pick the match):

  For each test face:
    - candidates = [true_name] + (N-1) random distractor names
    - score every candidate with the model (teacher-forced CE)
    - the model "picks" the lowest-CE candidate
    - hit = picked == true

  TEST A — VGGFace2 (identities the warm-start trained on)
  TEST B — our 8 eval celebs (Swift/Obama/LeCun/Federer/Bezos/Musk/Messi/Ronaldo;
           NOT in VGGFace2 — a true skill-transfer probe)

  Compare BASELINE (lerobot/pi05_base) vs WARMED (HBOrtiz/pi05_paligemma_celeb_warm).
  Random-guess accuracy = 1/N. If WARMED >> BASELINE >~ random, the warm-start
  taught face discrimination.

USAGE
    python eval_3/scripts/warmstart/eval_warmstart_vqa.py --n-way 5 --n-vggface2 40
"""
from __future__ import annotations
import argparse, csv, random, sys
from pathlib import Path

import torch
from PIL import Image

PROMPT = "<image>Who is the person in this image?\n"
EVAL_CELEBS = [
    "taylor_swift", "barack_obama", "yann_lecun", "roger_federer",
    "jeff_bezos", "elon_musk", "lionel_messi", "cristiano_ronaldo",
]


def load_paligemma(pi05_repo: str, device: str):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(pi05_repo)
    policy.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    return policy.model.paligemma_with_expert.paligemma.to(device).eval()


@torch.no_grad()
def score_candidates(pg, processor, image: Image.Image, candidates: list[str],
                      device: str) -> list[float]:
    """Teacher-forced CE for each candidate name given the image. Lower = better.

    One processor call per candidate (batched would need equal-length suffixes;
    per-candidate keeps it simple and the candidate count is small).
    """
    losses = []
    for name in candidates:
        inputs = processor(text=[PROMPT], images=[image], suffix=[name],
                           return_tensors="pt", padding="longest",
                           truncation="only_first", max_length=384)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = pg(**inputs)
        # out.loss is mean CE over the (non -100) suffix tokens
        losses.append(float(out.loss))
    return losses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="lerobot/pi05_base")
    ap.add_argument("--warm-model", default="HBOrtiz/pi05_paligemma_celeb_warm")
    ap.add_argument("--processor-name", default="google/paligemma-3b-pt-224",
                     help="PG1 processor — pi05 uses google/paligemma-3b-pt-224. "
                          "(PG1/PG2 tokenizers are byte-identical; image proc same.)")
    ap.add_argument("--scraped-bank", type=Path,
                     default=Path.home() / "LeMonkey/datasets/eval3_celebs/scraped")
    ap.add_argument("--vggface2-dir", type=Path,
                     default=Path.home() / "face_data/cropped-vggface2-224/data")
    ap.add_argument("--identity-meta-csv", type=Path, default=None)
    ap.add_argument("--n-vggface2", type=int, default=40)
    ap.add_argument("--n-way", type=int, default=5,
                     help="Candidates per face (1 true + N-1 distractors). "
                          "Random-guess accuracy = 1/N.")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.processor_name)

    # id → name map (auto-discover in HF cache)
    if args.identity_meta_csv is None:
        cands = list(Path.home().glob(".cache/huggingface/hub/**/identity_meta.csv"))
        args.identity_meta_csv = cands[0] if cands else None
    id_to_name = {}
    if args.identity_meta_csv:
        with open(args.identity_meta_csv, newline="") as f:
            for row in csv.DictReader(f):
                clean = {}
                for k, v in row.items():
                    if k is None:
                        continue
                    if isinstance(v, list):
                        v = ",".join(v) if v else ""
                    clean[k.strip()] = (v or "").strip().strip('"')
                cid, nm = clean.get("Class_ID"), clean.get("Name")
                if cid and nm:
                    id_to_name[cid] = nm.replace("_", " ")
    all_vgg_names = sorted(set(id_to_name.values()))

    # ── Build test sets: each item = (PIL image, true_name, [candidates]) ──
    from datasets import load_dataset, disable_progress_bar
    disable_progress_bar()
    shard0 = sorted(str(p) for p in args.vggface2_dir.rglob("*.parquet"))[:1]
    ds = load_dataset("parquet", data_files=shard0, split="train")
    label_feat = ds.features["label"]

    test_a = []
    for _ in range(args.n_vggface2):
        i = rng.randrange(len(ds))
        row = ds[i]
        nm = id_to_name.get(label_feat.int2str(row["label"]))
        if not nm:
            continue
        distractors = rng.sample([n for n in all_vgg_names if n != nm], args.n_way - 1)
        cands = [nm] + distractors
        rng.shuffle(cands)
        test_a.append((row["image"].convert("RGB"), nm, cands))

    # Test B: our eval celebs. Candidates = all 8 eval-celeb names (so N-way = 8,
    # exactly the closed-set our real task faces).
    eval_names = [" ".join(w.capitalize() for w in s.replace("-", " ").split("_"))
                  for s in EVAL_CELEBS]
    test_b = []
    for slug, name in zip(EVAL_CELEBS, eval_names):
        d = args.scraped_bank / slug
        if not d.is_dir():
            print(f"[WARN] missing eval celeb: {d}", flush=True)
            continue
        photos = sorted(p for p in d.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if photos:
            img = Image.open(rng.choice(photos)).convert("RGB")
            test_b.append((img, name, list(eval_names)))

    print(f"TEST A — VGGFace2 in-dist : {len(test_a)} faces, {args.n_way}-way "
          f"(random={100/args.n_way:.0f}%)")
    print(f"TEST B — our eval celebs  : {len(test_b)} faces, {len(eval_names)}-way "
          f"(random={100/len(eval_names):.0f}%)")
    print()

    # ── Run both models ─────────────────────────────────────────────────
    results = {}
    for tag, repo in [("BASELINE", args.base_model), ("WARMED", args.warm_model)]:
        print(f"==> {tag}: {repo}", flush=True)
        pg = load_paligemma(repo, device)
        rec = {"A": [], "B": []}
        for img, truth, cands in test_a:
            losses = score_candidates(pg, processor, img, cands, device)
            pick = cands[min(range(len(cands)), key=lambda j: losses[j])]
            rec["A"].append((truth, pick, pick == truth))
        for img, truth, cands in test_b:
            losses = score_candidates(pg, processor, img, cands, device)
            pick = cands[min(range(len(cands)), key=lambda j: losses[j])]
            margin = sorted(losses)[1] - min(losses)   # confidence gap
            rec["B"].append((truth, pick, pick == truth, margin))
        results[tag] = rec
        del pg
        torch.cuda.empty_cache()

    # ── Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"TEST A — VGGFace2 {args.n_way}-way discrimination")
    print("=" * 70)
    for tag in ("BASELINE", "WARMED"):
        a = results[tag]["A"]
        hit = sum(1 for *_, ok in a if ok)
        print(f"  {tag:<9}: {hit}/{len(a)} correct  ({100*hit/max(len(a),1):.0f}%)  "
              f"[random {100/args.n_way:.0f}%]")

    print("\n" + "=" * 70)
    print(f"TEST B — our 8 eval celebs, 8-way (= the real closed-set task)")
    print("=" * 70)
    for tag in ("BASELINE", "WARMED"):
        b = results[tag]["B"]
        # B tuples are (truth, pick, ok, margin) — index [2] is the bool.
        hit = sum(1 for row in b if row[2])
        print(f"  {tag:<9}: {hit}/{len(b)} correct  ({100*hit/max(len(b),1):.0f}%)  "
              f"[random 12%]")
    print("\n  per-celeb (truth → BASELINE pick | WARMED pick):")
    for i in range(len(results["BASELINE"]["B"])):
        truth, bp, bok, _ = results["BASELINE"]["B"][i]
        _, wp, wok, wmargin = results["WARMED"]["B"][i]
        bf = "✓" if bok else " "
        wf = "✓" if wok else " "
        print(f"    {truth:<20} → {bf} {bp[:18]:<18} | {wf} {wp[:18]:<18} "
              f"(margin {wmargin:.3f})")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    a_b = sum(1 for *_, ok in results["BASELINE"]["A"] if ok) / max(len(test_a), 1)
    a_w = sum(1 for *_, ok in results["WARMED"]["A"] if ok) / max(len(test_a), 1)
    b_b = sum(1 for row in results["BASELINE"]["B"] if row[2]) / max(len(test_b), 1)
    b_w = sum(1 for row in results["WARMED"]["B"] if row[2]) / max(len(test_b), 1)
    print(f"  VGGFace2 in-dist : BASELINE {a_b*100:.0f}% → WARMED {a_w*100:.0f}%  "
          f"(Δ {(a_w-a_b)*100:+.0f} pts)")
    print(f"  eval celebs      : BASELINE {b_b*100:.0f}% → WARMED {b_w*100:.0f}%  "
          f"(Δ {(b_w-b_b)*100:+.0f} pts)")
    print()
    # Compare against RANDOM chance, not against baseline — a model that
    # collapses to one guess can "beat baseline" while being at chance.
    rand_a = 1.0 / args.n_way
    rand_b = 1.0 / max(len(test_b), 1)
    if a_w > rand_a + 0.10:
        print(f"  ✓ Warm-start improved in-distribution discrimination "
              f"({a_w*100:.0f}% vs {rand_a*100:.0f}% random).")
    else:
        print(f"  ✗ In-distribution still near random ({a_w*100:.0f}% vs "
              f"{rand_a*100:.0f}%).")
    if b_w > rand_b + 0.12:
        print(f"  ✓ Real transfer to our eval celebs ({b_w*100:.0f}% vs "
              f"{rand_b*100:.0f}% random).")
    else:
        # Detect single-answer collapse
        warm_picks = [row[1] for row in results["WARMED"]["B"]]
        from collections import Counter
        top_pick, top_n = Counter(warm_picks).most_common(1)[0]
        print(f"  ✗ No usable transfer to eval celebs ({b_w*100:.0f}% ≈ "
              f"{rand_b*100:.0f}% random).")
        if top_n >= len(test_b) * 0.5:
            print(f"    WARMED collapsed to '{top_pick}' for {top_n}/{len(test_b)} "
                  f"celebs — not discriminating, just guessing one name.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Patches applied on top of `third_party/lerobot`

The submodule pin (currently `v0.5.1-104-g81948979`, upstream
huggingface/lerobot main) has two upstream bugs that block
`lerobot-train` on the dependency versions our environment actually
installs (`huggingface_hub==1.14.0`, `transformers==5.3.0`,
`torch==2.10`). These patches restore a working import + run path.

Apply them with:

```bash
bash third_party/lerobot_patches/apply.sh
```

`apply.sh` is idempotent (it checks for the patched markers before
applying). The Eval 3 setup scripts under `eval_3/scripts/training_vm/` and
`eval_3/scripts/smolvla_cotrain/setup_env.sh` invoke it automatically
after `pip install -e third_party/lerobot[...]`. (The shared
`scripts/setup_smolvla_env.sh` installs lerobot from PyPI and does
NOT need these patches.)

## Patch 1: `01-groot-skip-strict.patch`

`policies/groot/groot_n1.py` decorates `GR00TN15Config` with
`@strict` from `huggingface_hub.dataclasses`. On `huggingface_hub>=1.0`:
- `@strict` requires `@dataclass` first (the class lacked it), and
- `@strict` scans inherited methods for `validate_*` and rejects
  `validate_rope` (inherited from transformers `PretrainedConfig`)
  because it takes more than just `self`.

The patch drops `@strict` and adds `@dataclass`. PretrainedConfig
validation is unaffected; the only thing lost is the redundant
hf-hub-level field-type check.

## Patch 2: `02-untagged-dataset-fallback.patch`

`datasets/utils.py:get_safe_version` raises `RevisionNotFoundError`
for datasets without version tags, but `huggingface_hub>=1.0`'s
`RevisionNotFoundError.__init__` requires
`response: httpx.Response` as a keyword argument, which the call site
has no way to provide. The patch replaces the raise with a `[WARN]` +
fall-back to the `main` branch (per CLAUDE.md §5 no-silent-fallbacks
rule). The right long-term fix is to tag the HF datasets:

```python
from huggingface_hub import HfApi
HfApi().create_tag("HBOrtiz/so101_eval2", tag="v3.0", repo_type="dataset")
```

(See `docs/experiments/2026-05-23_training_smoke_audit.md` for the
full list of affected datasets and which trainers exercise them.)

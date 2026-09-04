# Model assets

This directory is a **local mount point**, not a place to commit model weights.

Leviathan's Git history should contain:

- model metadata;
- immutable revision identifiers;
- architecture/config snapshots when licensing permits;
- evaluation fingerprints;
- adapter/module deltas created by Leviathan when appropriately licensed;
- scripts required to reproduce asset acquisition.

It should not contain multi-GB/TB upstream checkpoints.

## Registry

The canonical model list is:

`spec/model-registry.toml`

The architecture/transplant plan is:

`spec/omega-transplant.toml`

The canonical DeepSeek V4 Mixture-of-Parameters contract is:

`spec/deepseek-v4-mop.toml`

## Recommended local layout

```text
models/
  README.md
  checkpoints/              # gitignored
    deepseek-v4-pro-base/    # canonical Leviathan pretrained core
    qwen3-30b-a3b-base/      # cheaper development/control checkpoint
    olmo3-32b-base/          # scientific control
    ...
  adapters/                 # gitignored by default; publish deliberately
  fingerprints/             # small reproducibility metadata may be committed later
```

## List registered assets

```bash
python scripts/fetch_model_assets.py --list
```

## Fetch canonical V4 metadata only

Metadata-only is the default, but frontier entries still require explicit acknowledgement:

```bash
python scripts/fetch_model_assets.py \
  deepseek-v4-pro-base \
  --allow-disabled
```

After metadata/config acquisition, validate the V4 architecture fingerprint without claiming the weights are present:

```bash
python scripts/prepare_deepseek_v4_mop.py --config-only
```

## Fetch the full canonical checkpoint

Install the optional downloader:

```bash
python -m pip install -e '.[models]'
```

The full DeepSeek-V4-Pro-Base checkpoint is multi-terabyte class, so requesting weights requires explicit opt-in. Pin an immutable upstream revision:

```bash
python scripts/fetch_model_assets.py \
  deepseek-v4-pro-base \
  --weights \
  --revision <hugging-face-commit-sha> \
  --allow-disabled
```

After acquisition, require all 64 canonical shards and emit the Leviathan MoP manifest:

```bash
python scripts/prepare_deepseek_v4_mop.py \
  --output runs/deepseek-v4-mop-manifest.json
```

The manifest's `full_checkpoint_verified` field must be true before a run may be reported as the canonical full-V4 experiment.

Qwen remains available as a smaller development/regression control:

```bash
python scripts/fetch_model_assets.py \
  qwen3-30b-a3b-base \
  --weights \
  --revision <hugging-face-commit-sha>
```

Before any large download, independently verify:

1. model-card/license terms;
2. current repository ID;
3. storage requirements;
4. network bandwidth/quota;
5. accelerator-memory requirements;
6. serving/training implementation support;
7. the exact immutable revision used by the experiment.

## Environment variables

- `HF_TOKEN` — optional Hugging Face access token for gated/private assets.
- `LEVIATHAN_MODEL_DIR` — override checkpoint destination.
- `LEVIATHAN_DEEPSEEK_V4_DIR` — override the V4 directory used by the MoP preparation script.
- `HF_HOME` — optional Hugging Face cache location.

See `.env.example`.

## Reproducibility rule

A benchmark or training result is not considered reproducible unless it records at least:

- registry model ID;
- exact upstream repository ID;
- immutable revision/commit;
- tokenizer revision;
- model config hash;
- serving/training code revision;
- quantization/precision;
- hardware topology;
- important runtime flags.

For canonical V4 results also record:

- whether all 64 weight shards were verified;
- MoP tile width;
- active tiles/token;
- router revision;
- parity result against the locked pretrained baseline.

## Weight-soup warning

Leviathan is an **architecture soup**, not a naive tensor soup.

Do not average unrelated checkpoints from DeepSeek, Kimi, Qwen, GLM, MiMo or Mistral merely because they are all large language models. Transfer capability by compatible block reuse, gated grafts, projection bridges or distillation as described in `docs/13-weight-transplantation.md`.

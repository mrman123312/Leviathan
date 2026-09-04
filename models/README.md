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

## Recommended local layout

```text
models/
  README.md
  checkpoints/              # gitignored
    qwen3-30b-a3b-base/
    olmo3-32b-base/
    deepseek-v4-pro-base/
    ...
  adapters/                 # gitignored by default; publish deliberately
  fingerprints/             # small reproducibility metadata may be committed later
```

## List registered assets

```bash
python scripts/fetch_model_assets.py --list
```

## Fetch metadata only

Metadata-only is the default and is intentionally safe:

```bash
python scripts/fetch_model_assets.py qwen3-30b-a3b-base
```

## Fetch model weights

Install the optional downloader:

```bash
python -m pip install -e '.[models]'
```

Then explicitly request weights and preferably pin an immutable upstream revision:

```bash
python scripts/fetch_model_assets.py \
  qwen3-30b-a3b-base \
  --weights \
  --revision <hugging-face-commit-sha>
```

Frontier entries are disabled for automatic download in the registry. A giant download therefore requires both the model ID and an explicit override:

```bash
python scripts/fetch_model_assets.py \
  deepseek-v4-pro-base \
  --weights \
  --revision <commit-sha> \
  --allow-disabled
```

Before doing this, independently verify:

1. model-card/license terms;
2. current repository ID;
3. storage requirements;
4. network bandwidth/quota;
5. accelerator-memory requirements;
6. serving implementation support;
7. the exact immutable revision used by the experiment.

## Environment variables

- `HF_TOKEN` — optional Hugging Face access token for gated/private assets.
- `LEVIATHAN_MODEL_DIR` — override checkpoint destination.
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

## Weight-soup warning

Leviathan is an **architecture soup**, not a naive tensor soup.

Do not average unrelated checkpoints from DeepSeek, Kimi, Qwen, GLM, MiMo or Mistral merely because they are all large language models. Transfer capability by compatible block reuse, gated grafts, projection bridges or distillation as described in `docs/13-weight-transplantation.md`.

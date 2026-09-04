# 17 — Prompt runtime and executable MoP-0 parity

Leviathan R4 is no longer only a model/tiling specification. This document defines the first executable prompt paths.

## Two prompt paths

### 1. Served original DeepSeek V4

For ordinary prompt testing, serve the canonical checkpoint with an inference engine that exposes an OpenAI-compatible API, then point Leviathan at that endpoint.

The base checkpoint is pretrained rather than instruction-tuned, so raw text completions are the default:

```bash
python scripts/run_prompt.py \
  --backend endpoint \
  --base-url http://127.0.0.1:8000 \
  --model deepseek-ai/DeepSeek-V4-Pro-Base \
  --prompt "The capital of France is"
```

Interactive shell:

```bash
python scripts/run_prompt.py --backend endpoint
```

Environment equivalents:

```text
LEVIATHAN_INFERENCE_URL=http://127.0.0.1:8000
LEVIATHAN_SERVED_MODEL=deepseek-ai/DeepSeek-V4-Pro-Base
LEVIATHAN_INFERENCE_API_KEY=
```

`--chat` is available for a checkpoint/server with a valid chat template, but should not be used to pretend the raw Base model is instruction tuned.

## 2. Local original or MoP-0 reference execution

Install the heavyweight local runtime only on a machine intended to load the checkpoint:

```bash
python -m pip install -e '.[inference]'
```

Run the original local checkpoint:

```bash
python scripts/run_prompt.py \
  --backend transformers \
  --model-dir /models/deepseek-v4-pro-base \
  --prompt "The capital of France is"
```

Run the same model with routed experts replaced by Leviathan's exact 128-channel MoP-0 reference reconstruction:

```bash
python scripts/run_prompt.py \
  --backend transformers \
  --model-dir /models/deepseek-v4-pro-base \
  --mop0-reference \
  --prompt "The capital of France is"
```

When `--mop0-reference` is enabled, Leviathan loads Transformers with `experts_implementation="eager"`. The point is to keep expert arithmetic inspectable and prevent a grouped/deep-GEMM backend change from being confused with MoP behavior. Fast expert kernels return later, after parity is proven.

The reference executor supports both public V4 layouts:

- the standalone DeepSeek-style routed experts exposing `w1`, `w2`, `w3`;
- Hugging Face's packed `DeepseekV4Experts`, where `gate_up_proj` has shape `[experts, 2 * intermediate, hidden]` and `down_proj` has shape `[experts, hidden, intermediate]`.

For the packed Transformers path, the original router and packed gate/up projection are unchanged. The activated 3,072-wide intermediate state is divided into 128-channel slices, the matching `down_proj[:, start:stop]` columns are evaluated, and all 24 tile contributions are summed before the original routing weight is applied. The shared expert remains untouched.

For a standalone expert object whose quantized `w2` cannot safely be sliced, the fallback reference masks the intermediate activation to one tile at a time and calls the unchanged donor `w2`, then sums the contributions.

Both paths are intentionally **correctness paths, not optimized serving paths**.

## Prompt-level MoP-0 parity test

The scientific R4 test is:

```bash
python scripts/check_mop0_prompt_parity.py \
  --model-dir /models/deepseek-v4-pro-base \
  --prompt "The capital of France is" \
  --require-argmax-match \
  --output runs/mop0-parity-france.json
```

The command:

1. loads one local V4 checkpoint using the eager expert backend;
2. runs the prompt with the original routed experts;
3. installs MoP-0 reference wrappers around routed experts only;
4. runs the exact same prompt again;
5. restores the original experts;
6. reports max/mean/RMS logit drift, relative L2 drift, and whether the last-token argmax matches.

The wrapper deliberately does **not** alter the shared expert or the V4 router. The original top-6 expert decision remains the routing decision at MoP-0.

## Why this reference is useful

The optimized tile kernel is a later implementation. We should not optimize before we know the mathematical transplant survives the real checkpoint.

This reference gives us an executable ladder:

```text
original V4 prompt
      ↓
same prompt through exact tile reconstruction
      ↓
measure logit drift
      ↓
prove prompt/benchmark parity
      ↓
build fused tile-aware expert kernel
      ↓
measure real speed
      ↓
only then train independent cross-expert tile routing
```

If the reference reconstruction cannot preserve V4 behavior, the problem is in the conversion or runtime and the independent MoP router must not be trained.

## Important limitation

These scripts do not make a 1.6T checkpoint fit onto ordinary hardware. They provide the execution contract once suitable V4-capable compute/storage is available.

The endpoint prompt path can talk to any machine or cluster already serving the model. The local `transformers` path requires the checkpoint and hardware/runtime capable of loading it.

A CPU unit test proves the tile decomposition itself on small tensors. It does **not** substitute for a real full-checkpoint FP8 parity run. That full run is the next empirical gate.

## R4 prompt test order

The first real run should be deliberately small:

1. checkpoint preflight with `prepare_deepseek_v4_mop.py`;
2. one raw baseline completion;
3. one short prompt parity run;
4. several deterministic prompt parity runs;
5. held-out WikiText loss parity;
6. ARC-Easy canary baseline/parity;
7. protected benchmark matrix;
8. only then kernel optimization and active-tile reduction.

Every result should keep the immutable checkpoint revision, tokenizer revision, runtime revision, hardware topology, precision and prompt/evaluator fingerprint.

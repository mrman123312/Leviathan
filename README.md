# Leviathan

One evolving model with expressive parameter cells, recurrent computation and evidence-gated learning. This is an executable research project, not an AGI or speedup claim.

## Current models

**Qwen/Qwen3.8-27B** is the canonical target at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its official release is **post-trained**, not raw base. The verified implementation is Qwen3_5ForConditionalGeneration: 5120-wide language states, 64 hybrid DeltaNet/attention layers and 17408-wide SwiGLU FFNs.

**Qwen/Qwen3-1.7B-Base**, revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, is the genuine pretraining-only control and default RTX 3060 test model. Small-model engineering tests are not evidence for giant-model MoP economics.

Nominal 27B four-bit weights alone exceed 12 GiB: this is not a fully resident RTX 3060 12 GB model. Nominal 1.7B weights are approximately 0.79 GiB at four bits or 3.17 GiB at BF16, before cache, CUDA, quantization metadata, unquantized tensors, adapters and workspace. Actual 3060 peak memory remains unmeasured.

DeepSeek code/tests are retained. Original README and architecture graph are archived as `docs/historical-deepseek-readme.md` and `spec/architecture-legacy.yaml`.

## Run prompts

Install a compatible CUDA PyTorch build, then:

```bash
python -m pip install -e '.[consumer]'
python scripts/run_prompt.py --prompt "The capital of France is"
```

The default loads the pinned 1.7B base through bitsandbytes NF4/double quantization. Initial download is the original checkpoint, not an already-compressed NF4 file. Run without `--prompt` for the interactive shell.

CPU baseline:

```bash
python scripts/run_prompt.py --device cpu --quantization none --prompt "The capital of France is"
```

Observe a zero-gated recurrent adapter on the final decoder FFN:

```bash
python scripts/run_prompt.py --nrdf --observe-at-zero --loops 4 --prompt "The capital of France is" --report runs/prompt.json
```

Execute actual ancestral cells with a float donor:

```bash
python scripts/run_prompt.py --quantization none --nrdf --cells --observe-at-zero --loops 4 --prompt "The capital of France is"
```

**NF4 ancestral slicing is not implemented.** NF4 supports the recurrent adapter, while ancestral cell tests support float matrices and the separate portable packed-INT4 reference. AWQ/GPTQ/NF4 packed tensors cannot be sliced as normal weights. No fused sparse GPU speed is claimed.

The larger target requires more memory and explicit opt-in:

```bash
python scripts/run_prompt.py --profile qwen27b --allow-large-model --prompt "Explain why ice floats."
```

This runner uses raw text, not an instruction/multimodal chat template. Base models produce completions. The 27B vision tower is not silently removed.

## What NRDF implements

`unchanged donor FFN + zero-gated recurrent residual`

The new path is a **small recurrent Transformer adapter**, not a conversion of the complete pretrained Qwen backbone into a looped model. It includes parallel token-local slots, shared attention/MLP steps, input injection, optional step conditioning, variable depth, bounded low-rank fast state and an optional halting head. Adaptive evaluation physically compacts continuing rows.

Ancestral cells can execute their real gate/up/down slices within each recurrence, exchange messages, recruit more compatible same-layer cells, carry differentiable depth-local state and merge their final proposals. No state is shared across independent requests. Donor causal attention and DeltaNet caches remain unchanged.

PulseBridge optionally reuses the same embedding and LM head for discrete token checkpoints and supervised alignment. It is not yet useful learned CoT. Slot meanings, confidence, abstention, routing and halting require training and calibration.

Zero-gate evaluation may bypass the new path. Training preserves its graph so the outer gate receives gradients; inner modules still require auxiliary training or gate warmup. Nonfinite proposals fail loudly.

## Executed evidence

An actual pinned **1.7B base** CPU BF16 smoke run generated `Paris. Paris is the capital of France. Paris is the` from `The capital of France is`. With the zero-gated recurrent/cell path executing, the prompt's maximum logit difference was **0.0**.

The first eight ARC-Easy test examples scored **4/8** for both donor and zero-graft, with identical predictions. This custom completion-scoring smoke test is **not** full ARC-Easy or a published lm-eval-equivalent result. No learned capability gain is inferred. Run artifacts record precise scope and versions.

```bash
python scripts/validate_model_registry.py
python -m unittest discover -s tests -v
python scripts/benchmark_consumer_reference.py
python scripts/check_consumer_hf.py
python scripts/check_consumer_hf.py --actual-base --arc-limit 8 --output evidence/consumer/pretrained-smoke.json
```

Native tiny-Qwen integration also checks opened-graft backpropagation, donor gradient isolation, incremental-cache parity, exact graft save/reload and same-model greedy speculative equivalence. These tiny native tests use random weights; they are not pretrained benchmark scores.

## Train a candidate

Supply disjoint JSONL datasets with a `text` field:

```bash
python scripts/train_consumer_graft.py --train data/train.jsonl --replay data/replay.jsonl --heldout data/heldout.jsonl --output runs/nrdf-candidate --steps 100
python scripts/run_prompt.py --graft runs/nrdf-candidate --allow-experimental --prompt "The capital of France is"
```

The trainer freezes inherited parameters, mixes replay, samples depths, measures heldout loss and saves only graft tensors. Passing one loss gate never promotes a model. Broad capability, calibration, safety and wall-clock gates remain mandatory. Exact text hashes prove exclusion from this optimizer/replay run, not absence from inherited pretraining.

## Efficiency results and boundaries

The CPU reference measured exact-input reuse at about **4x faster for one larger synthetic FFN**, but **slower for a tiny FFN**. All tested tiled FFN references were slower than dense execution. Six recurrent iterations performed worse than two on the synthetic optimization test. These findings reject universal cache/depth/sparsity claims.

Implemented components include bounded hot-tensor storage, route-margin checks, stable-body coefficient deltas and greedy-logit margin checks. They are falsifiable adaptations of existing mathematical ideas, not claims of unprecedented invention.

Same-model shallow/deep speculative generation has executable greedy and corrected rejection-sampling references. It replays full prefixes and makes no throughput claim. Native MTP kernels, hybrid-cache transactions, modified-model paged serving, predictive prefetch, NF4 sparse kernels and fused coalitions remain pending. Stock vLLM does not automatically run these Python grafts.

## Research and remaining work

- `docs/19-consumer-nrdf.md`: implementation and limitations.
- `docs/20-recurrent-research.md`: primary-source audit.
- `docs/21-steering-ledger.md`: failures and corrective rules.
- `spec/consumer-substrate.toml`: pins and invariants.
- `spec/consumer-checklist.json`: full pending/partial mechanism checklist.
- `spec/architecture-maturity.toml`: L0–L10 five-gate ledger.
- `src/leviathan/consumer/`: new executable neural/runtime code.

Legacy belief/memory/controller primitives remain available, but are not yet a learned end-to-end autonomous system connected to this prompt runner. The standard remains **Specification → Executable → Integrated → Learned → Demonstrated**.

# Leviathan — Frozen Bedrock

One pretrained semantic model, reused through different computational routes. **No training or new neural parameters in the Bedrock path.** This is an executable architecture experiment, not a demonstrated AGI or language-quality improvement.

## Current implementation

`BedrockRuntime` extends the existing cognitive runtime rather than adding another model. It connects frozen Qwen layer-band recurrence, bounded activation state, real ancestral parameter-cell communication/recruitment, host verification, belief revisions, persistent executable skills and a live dependency graph.

The existing `Qwen/Qwen3-1.7B-Base` control and pinned revision remain unchanged. The larger Qwen27B target stays in the historical registry. **The new frozen-band adapter supports Qwen2/Qwen3; the hybrid 27B port is pending.** Earlier random NRDF adapters and training scripts remain historical research and are not invoked by this path.

Read [the mechanism/proof/gap report](docs/22-frozen-bedrock.md) and [machine-readable policy](spec/bedrock.toml).

## Run without reinstalling

On the already-working RTX 3060 Windows setup, double-click **`RUN_FROZEN_BEDROCK.bat`**. It reuses `C:\LeviathanBenchmarkCache\.venv-v7\Scripts\python.exe` and the existing Qwen cache. No drive scanning, CUDA installation, model download, CPU fallback or training. It opens `results/RESULTS.html` and preserves raw results in JSON.

This is a **feature lab**, not another full ARC/WikiText run. The new Windows/CUDA path has not been executed in the local Linux CPU environment.

For existing development environments:

```bash
python scripts/validate_model_registry.py
python -m unittest discover -s tests -v
python scripts/run_bedrock_lab.py --mechanisms-only
python scripts/check_bedrock_hf.py
```

The last command needs Transformers and tests a tiny native Qwen with random frozen weights, not a pretrained checkpoint.

## Evidence and limits

The local full suite passes **147 tests**, including the original 97. A separate no-training finite-grammar experiment solved 36 declared deterministic worlds, validated on fresh observations, persisted programs, and answered **137 previously unqueried domain inputs correctly after reload**. Boolean-circuit discovery averaged 4.58 queries versus 7.67 for fixed order, before two additional validation observations.

Those are bounded algorithm/mechanism results. They are **not Qwen reasoning scores or general unknown-world competence**. No new ARC-Easy, WikiText, GPU-speed or 27B result is claimed. The donor remains the production default. Nonneutral recurrence, sparse routing and fast state can still reduce quality or increase latency.

The proof report separates real-arithmetic invariants from floating-point bounds and empirical quality. It also identifies unfinished general theory induction, calibrated uncertainty, useful automatic latent reasoning, hybrid caches and fused sparse GPU execution.

## History is preserved

See [the pre-Bedrock README](docs/pre-bedrock-readme.md), [earlier DeepSeek README](docs/historical-deepseek-readme.md), and the existing architecture documents and tests. R0–R9 is not restarted. R3's sparse-speed failure and the user's prior zero-graft retention result remain historical evidence, not reasons to force MoP onto a tiny model.

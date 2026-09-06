# Leviathan — Frozen Bedrock

One pretrained semantic model, reused through different computational routes. **No training or new neural parameters in the Bedrock path.** This is an executable architecture experiment, not a demonstrated AGI or language-quality improvement.

## Current implementation

`BedrockRuntime` extends the existing cognitive runtime rather than adding another model. It connects frozen Qwen computation, bounded activation state, real ancestral parameter-cell communication/recruitment, host verification, belief revisions, persistent executable skills and a live dependency graph.

The existing `Qwen/Qwen3-1.7B-Base` control and pinned revision remain unchanged. The larger Qwen27B target stays in the historical registry. The hybrid 27B port remains pending. Earlier random NRDF adapters and training scripts remain historical research and are not invoked by this path.

### RTX result changed the recurrence design

The first real RTX 3060 FP16 feature-lab run falsified naive raw band loopback as a safe default: the untouched donor was healthy, but feeding a late-band output directly back into an earlier pretrained band produced a non-finite recurrent state.

The experimental lab now uses **transported frozen recurrence** instead. Each recurrent probe is moved only a bounded distance from the actual donor band input, the resulting innovation is applied under a second output trust region, and any non-finite replay falls back to the untouched donor rather than aborting the run. See `docs/23-transport-recurrence.md`.

Read [the main mechanism/proof/gap report](docs/22-frozen-bedrock.md) and [machine-readable policy](spec/bedrock.toml).

## Run without reinstalling

On the already-working RTX 3060 Windows setup, double-click **`RUN_FROZEN_BEDROCK.bat`**. It reuses `C:\LeviathanBenchmarkCache\.venv-v7\Scripts\python.exe` and the existing Qwen cache. No drive scanning, CUDA installation, model download, CPU fallback or training.

The current lab warms CUDA before timing, isolates failures per experimental route, and records route status, non-finite fallback count, re-entry radius, peak VRAM and logit change. It opens `results/RESULTS.html` and preserves raw results in JSON.

This is a **feature lab**, not another full ARC/WikiText run.

For existing development environments:

```bash
python scripts/validate_model_registry.py
python -m unittest discover -s tests -v
python scripts/run_bedrock_lab.py --mechanisms-only
python scripts/check_bedrock_hf.py
```

The last command uses a tiny randomly initialized native Qwen fixture, not a pretrained checkpoint.

## Evidence and limits

Current branch CI is green across `validate-model-registry`, `consumer-reference` and `bedrock-frozen`.

The full no-training branch suite passes **151 tests**, including the original 97 project tests and the transported-recurrence stability tests. The native tiny-Qwen integration retains exact neutral logits, bytewise-equal donor tensors and same-model greedy speculative equivalence.

A separate no-training finite-grammar experiment solved 36 declared deterministic worlds, validated on fresh observations, persisted programs, and answered **137 previously unqueried domain inputs correctly after reload**. Boolean-circuit discovery averaged 4.58 queries versus 7.67 for fixed order, before two additional validation observations.

Those are bounded algorithm/mechanism results. They are **not Qwen reasoning scores or general unknown-world competence**. No new ARC-Easy, WikiText, general language-quality or 27B result is claimed. The real RTX run supplied an important negative result for raw recurrence; the transported repair still needs its own real RTX evidence before any quality or speed claim.

The donor remains the fallback. Nonneutral recurrence, sparse routing and fast state can still reduce quality or increase latency.

## History is preserved

See [the pre-Bedrock README](docs/pre-bedrock-readme.md), [earlier DeepSeek README](docs/historical-deepseek-readme.md), and the existing architecture documents and tests. R0–R9 is not restarted. R3's sparse-speed failure and the user's prior zero-graft retention result remain historical evidence, not reasons to force MoP onto a tiny model.

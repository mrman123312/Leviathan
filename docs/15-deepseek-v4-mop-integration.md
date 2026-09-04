# DeepSeek V4 -> Leviathan Mixture-of-Parameters integration

This document defines the current canonical neural-substrate experiment for Leviathan.

The goal is **not** merely to rename DeepSeek V4's Mixture-of-Experts routing. The goal is to inherit the complete pretrained DeepSeek-V4-Pro-Base function, expose its learned expert capacity at a finer parameter granularity, and then test whether parameter-tile composition can improve capability-per-active-compute without damaging the pretrained model.

This is still only the neural substrate of Leviathan. Persistent beliefs, metacognitive routing, verification, memory, causal credit and transactional learning remain separate architectural layers around the same semantic core.

## 1. Canonical source

Registry ID:

`deepseek-v4-pro-base`

Upstream:

`deepseek-ai/DeepSeek-V4-Pro-Base`

Leviathan uses the **Base** checkpoint because the architecture surgery should begin from the pretrained semantic model rather than depending on a post-trained chat policy.

The current source fingerprint enforced in code is:

- architecture: `DeepseekV4ForCausalLM`
- model type: `deepseek_v4`
- hidden layers: 61
- hidden size: 7168
- routed experts: 384
- shared experts: 1
- routed experts per token: 6
- MoE intermediate width: 3072
- configured maximum positions: 1,048,576
- weight shards: 64

`src/leviathan/deepseek_v4.py` rejects a config that differs from this fingerprint. The full-checkpoint gate additionally requires every expected safetensors shard and `model.safetensors.index.json`.

The purpose of the fingerprint is to stop a reduced one-layer test, a Flash checkpoint, or another V4 variant from accidentally being reported as the canonical Leviathan experiment.

## 2. Why parameter tiles instead of scalar-parameter routing

Routing individual scalar weights would create catastrophic indexing, memory-movement and kernel-launch overhead. R3 already taught the project that mathematical sparsity does not guarantee real speed.

Leviathan therefore routes **contiguous hardware-friendly parameter tiles**.

The initial tile axis is the expert SwiGLU intermediate dimension. For tile width 128:

`3072 / 128 = 24 tiles per expert`

Across 384 routed experts:

`384 * 24 = 9,216 routed tiles per layer`

The inherited DeepSeek route selects 6 experts, so the exact parity route contains:

`6 * 24 = 144 routed tiles per token`

This does **not** initially save compute. That is intentional. The first requirement is preservation, not sparsity.

## 3. Exact function-preserving decomposition

For a routed expert with SwiGLU-style intermediate channels, divide the intermediate dimension into contiguous blocks. Each tile owns the corresponding slices of the gate projection, up projection and down projection.

Conceptually:

`E(x) = sum_j Tile_j(x)`

where the sum is over every tile belonging to the original expert.

At the parity stage, if the original router selects experts:

`{e1, e2, e3, e4, e5, e6}`

Leviathan selects:

`all tiles(e1) + all tiles(e2) + ... + all tiles(e6)`

The original route therefore remains exactly reconstructable before the system learns any cross-expert parameter composition.

Important: this claim depends on implementing the tensor slicing and accumulation consistently with the actual upstream expert equations and quantization representation. The current repository implements the architecture/fingerprint/route manifest and invariants, not a claim that 1.6 TB of weights have already been rewritten and numerically verified.

## 4. Migration phases

### V4-M0 — baseline lock

Record:

- immutable upstream revision,
- tokenizer revision,
- model config hash,
- exact weight manifest,
- inference/training engine revision,
- precision/quantization,
- hardware topology,
- baseline benchmarks,
- baseline calibration,
- baseline latency/throughput/memory metrics.

No architecture change is accepted without this frozen reference.

### V4-M1 — full-checkpoint validation

Require the canonical 61-layer config and all 64 weight shards.

A partial checkpoint may be used for engineering experiments, but it must never be labeled a canonical Leviathan result.

### V4-M2 — inert expert-channel tiling

Represent each routed expert as 24 contiguous 128-channel tiles while retaining the original expert routing decision.

No independent tile routing.

No reduction in active tiles.

No pretrained-weight update.

### V4-M3 — parity proof

Compare the original model against the tiled representation on identical inputs.

Required checks:

- logits,
- selected hidden states,
- expert/router outputs,
- language-model loss,
- benchmark outputs where deterministic evaluation is available.

Drift must be numerical/kernel-precision scale or explicitly explained. Benchmark regression is not permitted at this stage.

### V4-M4 — tile-router distillation

Train a tile-level router while keeping the pretrained semantic weights frozen.

The original expert route acts as a strong teacher/constraint. Early tile routes should remain close to the exact expert reconstruction until they demonstrate parity.

### V4-M5 — cross-expert parameter composition

Only after parity is stable may the router select tiles independently across different original experts.

This is the point where the architecture becomes meaningfully different from conventional expert routing.

The experiment asks whether useful computations are distributed across expert subspaces such that a token benefits from, for example, several tiles from one expert, a few from another and none from a third rather than paying for entire expert MLPs.

### V4-M6 — active-tile reduction

Reduce active tile count gradually.

The first candidate should be conservative. Do not jump directly from 144 routed tiles to an arbitrarily tiny number.

For each active-tile budget record:

- capability,
- held-out language loss,
- calibration,
- router entropy/confidence,
- active parameters,
- HBM traffic,
- communication time,
- router overhead,
- single-stream throughput,
- aggregate throughput,
- wall-clock task success.

A candidate with fewer active parameters but worse wall-clock performance fails.

### V4-M7 — selective unfreeze

Only if routing alone cannot recover/improve capability should the smallest necessary pretrained parameter groups be unfrozen.

Use:

- continued-pretraining objective,
- replay,
- stability regularization,
- protected capability suites,
- independent evaluation.

### V4-M8 — Leviathan hooks

Once the semantic substrate is stable, integrate zero-gated hooks for:

- persistent belief state,
- memory context,
- metacognitive state,
- routing uncertainty,
- future-state/action prediction,
- verifier-outcome prediction.

These hooks must begin with zero or identity influence. DeepSeek V4 remains the inherited semantic function while the new pathways learn.

## 5. Protected benchmark gate

The MoP experiment should not optimize one headline benchmark while degrading broad competence.

Protected suite currently includes:

- ARC-Easy,
- ARC-Challenge,
- MMLU,
- MMLU-Pro,
- HellaSwag,
- WinoGrande,
- GSM8K,
- HumanEval,
- AGIEval,
- LongBench-V2,
- PIQA,
- OpenBookQA,
- BoolQ,
- BBH,
- held-out WikiText loss.

### ARC-Easy canary

ARC-Easy is kept as an explicit canary because earlier small-model experiments showed disproportionate weakness there.

Do not dismiss an ARC-Easy drop as an unimportant benchmark quirk. Diagnose whether it comes from:

- token/probability formatting,
- few-shot or prompt mismatch,
- loss of elementary factual associations,
- router collapse,
- capacity starvation,
- calibration/choice-scoring error,
- benchmark harness differences.

The reason matters because an architecture that improves difficult reasoning while damaging simple reliable recall may be allocating compute incorrectly.

### WikiText gate

The earlier hard retention boundary remains:

`held-out WikiText loss increase <= 2%`

For the giant V4 migration, the target is much stricter:

`held-out WikiText loss increase <= 0.25%`

The 2% value is a rejection ceiling, not a desired result.

## 6. Efficiency gate

R3 demonstrated the difference between theoretical and real efficiency.

Every MoP result must report at least:

- active parameters/token,
- active tiles/token,
- HBM bytes/token,
- router overhead,
- expert/tile communication time,
- GPU utilization,
- time to first token,
- inter-token latency,
- single-stream tokens/sec,
- aggregate tokens/sec,
- wall-clock seconds per successful task.

No "speedup" may be claimed from active-parameter count alone.

## 7. One-model invariant

Leviathan's cognitive core remains one model.

The teacher ensemble in the repository is an **offline training/evaluation resource**, not a runtime civilization of models pretending to be one agent.

The canonical deployed semantic path is one DeepSeek-V4-Pro-Base-derived Leviathan checkpoint. External deterministic tools, verifiers, sensors and memory services remain external resources, just as compilers/databases are external resources, but they do not replace the single cognitive model.

## 8. Relationship to the larger Leviathan architecture

Mixture-of-Parameters addresses only one question:

> Can a huge learned parameter reservoir be composed more finely and efficiently than fixed whole-expert selection?

It does not solve:

- persistent beliefs,
- epistemic uncertainty,
- contradiction handling,
- adaptive recursive thinking,
- few-shot rule induction,
- long-term memory,
- causal credit,
- verifier independence,
- safe continual learning,
- goal/governance stability.

Those capabilities are layered around the same semantic substrate according to `docs/03-cognitive-architecture.md`, `docs/04-meta-controller.md`, `docs/05-world-belief-model.md`, `docs/06-memory-and-continual-learning.md` and `docs/07-verification-and-credit.md`.

## 9. Executable preparation

The repository does not store giant weights.

Fetch metadata:

```bash
python scripts/fetch_model_assets.py deepseek-v4-pro-base --allow-disabled
```

Validate only the downloaded config:

```bash
python scripts/prepare_deepseek_v4_mop.py --config-only
```

After the complete checkpoint has been acquired, validate all shards and emit a manifest:

```bash
python scripts/prepare_deepseek_v4_mop.py \
  --output runs/deepseek-v4-mop-manifest.json
```

The manifest is the handoff contract for the future tensor/kernel implementation. It records the canonical model identity, source fingerprint, full-checkpoint status and MoP tile geometry.

## 10. Current implementation boundary

What is implemented now:

- canonical V4 registry selection,
- exact source fingerprint validation,
- full 64-shard presence validation,
- MoP tile geometry,
- exact expert-route -> tile-route expansion,
- machine-readable migration phases/gates,
- unit tests,
- CI validation.

What is **not yet implemented**:

- rewriting/dequantizing/repacking the 1.6 TB V4 weights into tile-aware kernel storage,
- a fused tile router/kernel,
- parity measurements on the real checkpoint,
- full benchmark results,
- training independent tile routing,
- measured speedup.

Those are the next engineering steps. The repository deliberately separates "architecture is specified and validated" from "the giant model has actually been converted and benchmarked" so experimental claims remain honest.

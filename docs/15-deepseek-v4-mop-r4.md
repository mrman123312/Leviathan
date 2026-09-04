# R4 — DeepSeek V4 Mixture-of-Parameters transplant

## Decision

R4 no longer uses a small Qwen checkpoint as the canonical sequence-model experiment.

The canonical R4 substrate is now:

`deepseek-ai/DeepSeek-V4-Pro-Base`

The Qwen base entry remains in the registry only as a cheaper regression/development
fallback. It is not the model that defines the final Leviathan parameter substrate.

This change follows the project goal: prove the architecture on the actual class of
model for which Mixture-of-Parameters is intended, while retaining the ability to use
smaller models for debugging.

## Why this is not a model-name swap

Leviathan's end goal is a single pretrained semantic core that can later participate in:

- adaptive parameter/compute routing,
- persistent beliefs and calibrated uncertainty,
- recursive/adaptive deliberation,
- few-experience rule acquisition,
- verified memory and continual learning,
- procedural compilation,
- independent verification and causal credit,
- native action heads and world-model interaction.

R4 modifies only the neural capacity substrate. It must not simultaneously rewrite the
attention system, mHC residual topology, tokenizer, belief architecture or learning
governance. Those changes have separate gates.

The rule is:

`inherit pretrained function -> prove parity -> introduce new routing -> measure -> keep or reject`

## Upstream architecture contract

The R4 contract is machine-readable in `spec/deepseek-v4-mop.toml`.

The verified V4-Pro-Base configuration used to design the transplant has:

- 61 decoder layers,
- hidden size 7,168,
- 384 routed experts,
- 1 shared expert,
- 6 routed experts selected per token,
- routed-expert SwiGLU intermediate size 3,072,
- FP8 expert weights with 128 x 128 weight blocks,
- 128 attention heads and one shared KV head,
- 1,048,576 maximum positions,
- mHC residual connectivity,
- one next-token-prediction auxiliary layer.

Do not trust these numbers merely because they are written here. Every actual run must
pin an immutable upstream revision and pass the config validator before weights are
loaded.

## MoP-0: exact expert decomposition

DeepSeek's routed expert can be written schematically as:

`E(x) = W_down [ SiLU(W_gate x) * (W_up x) ]`

Let the 3,072 intermediate channels be divided into disjoint sets `S_j`.

Because the down projection is linear over intermediate channels:

`E(x) = sum_j W_down[:, S_j] [ SiLU(W_gate[S_j] x) * W_up[S_j] x ]`

Therefore an expert can be represented as independent channel tiles without changing
the function, as long as:

1. every tile uses the matching gate rows,
2. every tile uses the matching up rows,
3. every tile uses the matching down-projection columns,
4. all tiles of an originally selected expert are active,
5. every tile inherits the original expert's route weight,
6. tile outputs are summed before the routed contribution is combined with the shared expert.

No averaging, distillation or approximation is required for MoP-0.

### Default tile geometry

The initial tile width is 128 channels because it matches the checkpoint's 128 x 128
FP8 expert weight block.

For each routed expert:

`3072 / 128 = 24 parameter tiles`

For each layer:

`384 * 24 = 9,216 routed parameter tiles`

The original router activates 6 experts, so MoP-0 expands that route to:

`6 * 24 = 144 routed tiles per token`

This does **not** reduce active compute yet. It changes the representation of routing
while preserving the pretrained function. Any claimed MoP speedup begins only after
the tile router can safely choose fewer than the equivalent original set.

The shared expert remains untouched in R4.

## MoP-1: learn the tile router

After MoP-0 parity:

1. freeze the DeepSeek donor path,
2. create the tile router with the original expert route as its teacher/reference,
3. begin with the exact 144-tile route,
4. train only the new routing parameters,
5. measure routing calibration and load balance,
6. keep the original expert route available as the rollback path.

The tile router is not allowed to become "smarter" by silently changing the model
function before the retention suite is running.

## MoP-2: reduce active tiles

Only after routing is stable do we test whether finer routing is useful.

Candidate tile widths can include 256, 128 and 64, but theoretical sparsity does not
decide the winner. For every candidate record:

- active parameters/token,
- active tiles/token,
- single-stream tokens/s,
- aggregate tokens/s,
- time to first token,
- inter-token latency,
- HBM bytes/token,
- HBM bandwidth utilization,
- GPU utilization,
- routing/load-balance overhead,
- task success and protected benchmark scores.

A configuration is rejected if it activates fewer parameters but runs slower in real
wall-clock execution. This directly carries forward the failed R3 speed lesson.

Do not route scalar parameters. The routing granularity must remain large enough for
efficient grouped/fused kernels.

## MoP-3: retire the old expert route

The original expert-level router is redundant only after the independent tile route
passes every required suite:

- capability,
- retention,
- calibration,
- safety,
- adversarial robustness,
- efficiency,
- rollback restoration.

Until then, the old route remains the reference behavior.

If MoP never beats the original V4 MoE on quality per real compute, Leviathan keeps the
original V4 MoE. Mixture-of-Parameters is a hypothesis, not a required branding feature.

## R4 retention gate

### Public language preservation

Use 64 held-out WikiText passages that are never used for:

- router training,
- replay,
- adapter training,
- hyperparameter selection.

Hard rejection boundary:

`relative held-out public-language loss increase <= 2%`

Target:

`<= 0.25%`

The target is intentionally much stricter than the hard boundary.

### Protected benchmark matrix

ARC-Easy is a named canary because earlier small-scale work struggled there. It must be
reported separately for every candidate rather than hidden inside an average.

Protected set:

- ARC-Easy,
- ARC-Challenge,
- HellaSwag,
- PIQA,
- BoolQ,
- OpenBookQA,
- MMLU,
- GSM8K,
- HumanEval.

The exact evaluator version, prompt formatting, sample count and random seed belong in
the run fingerprint.

When a score drops, the report should say in plain language what kind of ability likely
moved and which architecture change caused it. Do not keep a change merely because a
different benchmark rose.

## Baseline fingerprint

Before any MoP training, lock:

- model registry ID,
- Hugging Face repository ID,
- immutable upstream revision,
- config SHA-256,
- tokenizer revision,
- tokenizer/config hashes,
- inference/training code revision,
- precision/quantization,
- serving engine revision,
- kernel revisions,
- accelerator type/count/topology,
- tensor/expert/pipeline parallel layout,
- context length and batch sizes,
- benchmark harness revisions.

`python scripts/prepare_deepseek_v4_mop.py` performs the config/manifest portion of this
preflight without loading the checkpoint weights.

Example:

```bash
python scripts/prepare_deepseek_v4_mop.py \
  --config /models/deepseek-v4-pro-base/config.json \
  --revision <immutable-hugging-face-commit> \
  --output models/fingerprints/deepseek-v4-r4.json
```

Model weights remain outside Git.

## Kernel direction

The first production MoP kernel should work with contiguous expert-channel tiles and
group work by destination expert/tile so GEMMs stay large.

Existing vendored projects relevant to this work include DeepGEMM, TileKernels, Triton,
CUTLASS, DeepEP, SGLang and vLLM.

The research order is:

1. exact reference implementation,
2. numerical parity,
3. grouped/fused kernel,
4. throughput/HBM benchmark,
5. only then sparsity sweep.

Do not optimize an incorrect implementation.

## Relationship to the rest of Leviathan

R4 is only the nonlinear sequence/parameter substrate.

After R4, the same single model still has to acquire the later Leviathan mechanisms:

- R5 adaptive recursive thinking,
- R6 beliefs, uncertainty and contradiction handling,
- R7 unfamiliar-rule learning from few experiences,
- R8 verified long-term memory and safe learning,
- R9 canonical integrated Leviathan.

Mixture-of-Parameters therefore must expose route uncertainty, active-compute cost and
hardware cost to the later metacognitive controller. It must not become an isolated
serving trick.

The eventual controller should be able to ask two different questions:

1. **How should I think?** direct, recall, reason, search, simulate, experiment, tool...
2. **How much of the neural parameter substrate should this state activate?**

Those controls can interact, but they are not the same decision.

## R4 success definition

R4 passes only when all of these are true:

1. the full V4 base checkpoint can be fingerprinted reproducibly;
2. MoP-0 reproduces the original model within the declared numerical tolerance;
3. tile routing can be trained without an unexplained protected-capability regression;
4. held-out language preservation passes;
5. ARC-Easy no longer shows an unexplained architecture-specific collapse;
6. at least one MoP setting produces a real efficiency advantage or a capability gain
   with no efficiency loss;
7. the original checkpoint remains restorable.

If item 6 never happens, R4 can still teach us that V4's original MoE is the better
substrate. In that case, reject MoP at this scale and continue Leviathan on the original
pretrained V4 architecture.

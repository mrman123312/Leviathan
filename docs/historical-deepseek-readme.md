# Leviathan

**Leviathan** is a research and engineering blueprint for a general-purpose cognitive architecture assembled from the strongest ideas we studied across open, open-weight, partially open, and closed frontier AI systems.

This repository is not a claim that AGI has been solved. It is a structured synthesis of what current systems demonstrate, what design lessons can be extracted from them, and what still has to be invented before a stable, continually learning general agent is plausible.

## Core thesis

The important unit of optimization is no longer the next token. It is the **successful cognitive trajectory**.

A capable system should learn:

1. how finely to represent information,
2. how much neural compute to spend,
3. whether to recall, reason, search, simulate, experiment, parallelize, or execute a known skill,
4. how to represent actions in their native space,
5. how to verify outcomes against reality or trusted formal systems,
6. how to assign credit and blame across long trajectories,
7. what deserves to be remembered,
8. what deserves to be compiled into a reusable skill,
9. what deserves to change plastic parameters,
10. and what is trustworthy enough to enter slow core-weight consolidation.

The resulting system looks less like one giant chatbot and more like a **hierarchical cognitive operating system** built around one evolving semantic model.

## One-model invariant

Leviathan means **one cognitive model**, not a committee of LLMs pretending to be one agent.

Parameterized cells, cognitive operators, memories, hypotheses and verifiers are components/state of the same system. The canonical architecture keeps:

- one global cognitive state,
- one semantic model identity,
- one parameter ownership system,
- one learning/promotion boundary,
- one final output/action stream.

The repository validator rejects parameter-cell or cognitive-kernel specs that introduce an independent subagent committee.

## Three-loop architecture

Leviathan separates cognition into three timescales.

### 1. Fast cognitive loop

`perception -> state update -> compute routing -> prediction -> action`

### 2. Deliberative loop

`belief state -> hypotheses -> plan/search/simulate -> action -> verification -> evidence update -> replan`

### 3. Consolidation loop

`experience -> verify -> episodic memory -> semantic/procedural abstraction -> plastic candidate -> protected consolidation`

## Leviathan Ω model layer

Leviathan is an **architecture soup, not a naive weight soup**. Unrelated giant-model tensors are not averaged together. Pretrained capability is transferred by direct compatible reuse, zero-gated grafts, latent projection bridges and distillation.

### Canonical pretrained core

**DeepSeek-V4-Pro-Base is the canonical Leviathan semantic substrate.** Qwen3-30B-A3B-Base remains a cheaper development/regression control, but it no longer defines the primary architecture experiment.

The canonical V4 integration is intentionally the **full checkpoint**, not a reduced layer sample:

- 61 hidden layers,
- hidden size 7168,
- 384 routed experts plus 1 shared expert,
- 6 routed experts active per token in the inherited MoE route,
- MoE intermediate width 3072,
- 1,048,576-token configured maximum context,
- 64 safetensors weight shards.

`src/leviathan/deepseek_v4.py` rejects a config that does not match the canonical fingerprint. A run may claim `full_checkpoint_verified=true` only after all 64 local shards are present **and** `model.safetensors.index.json` references the complete canonical shard set. `src/leviathan/deepseek_v4_mop.py` adds the stricter R4 architecture/FP8/revision contract.

### Mixture-of-Parameters migration

The first V4 -> Leviathan MoP transformation is function-preserving.

With 128 intermediate channels per tile:

- 3072 / 128 = **24 tiles per routed expert**,
- 384 x 24 = **9,216 routed parameter tiles per layer**,
- the inherited 6-expert route expands to **144 tiles per token** at the parity stage.

At initialization, every selected expert is still reconstructed from **all** of its 24 tiles. Independent cross-expert tile routing is disabled until logit/hidden-state parity and protected benchmark gates pass.

Mathematical sparsity alone is not success. A lower-active-tile candidate is accepted only if **measured wall-clock efficiency** improves without capability, retention, calibration or safety loss.

See `spec/deepseek-v4-mop.toml`, `docs/15-deepseek-v4-mop-r4.md` and `docs/16-deepseek-v4-mop-integration.md`.

### Mixture of Parameterized Cells — L1.5

The 128-channel tile is now treated as an **ancestral cell body**, not the final primitive.

The safe insertion rule is:

`Cell_i(h) = Tile_i(h) + alpha_i * Refine_i(h, control_i)`, with `alpha_i = 0` at insertion.

`src/leviathan/parameter_cells.py` adds an executable reference membrane around packed V4 expert tiles. A cell can emit:

- multidimensional confidence,
- abstention probability,
- a low-dimensional proposal message,
- an associative recruitment query,
- a low-rank refinement proposal.

Those signals are observational at insertion. The original V4 route and shared expert remain authoritative until later gates are trained and demonstrated.

The machine-readable progression in `spec/parameter-cells.toml` is:

`MoP-0 exact tiles -> MoP-1 independent tile routing -> MoP-2 confidence/abstention -> MoP-3 proposal messages -> MoP-4 one communication round -> MoP-5 disagreement recruitment -> MoP-6 local state -> MoP-7 verified coalitions -> MoP-8 transactional local plasticity -> MoP-9 grow/split/merge/prune`.

Sparse communication and associative recruitment have executable reference primitives. Persistent local state, local plasticity and cell lifecycle remain gated future stages; they are not falsely marked complete.

See `docs/18-parameter-ecology-and-embodiment.md`.

### Single-model cognitive kernel

`src/leviathan/cognitive_kernel.py` turns the high-level architecture into an explicit reference execution structure:

`Representation Compiler -> Cognitive Program Compiler -> Dynamic Cognitive Graph -> Hypothesis/Prediction -> Evidence Update -> Learning Router -> Cognitive Compilation`

The reference kernel currently provides:

- problem-dependent representation plans (symbolic, procedural, causal, spatial, concept/event, graph),
- explicit cognitive instructions rather than hiding state in prompts,
- bounded acyclic dependency graphs,
- hypothesis and prediction records before outcomes,
- independence-discounted evidence updates,
- structured goal state,
- conservative learning destinations,
- protected core-consolidation gates,
- repeated-verified-trajectory skill compilation,
- append-only cognitive events and causal-accountability records.

It owns exactly one semantic model id. It is a transparent baseline architecture that later learned policies must beat, not a claim that the learned representation compiler, world model or metacognitive policy already exists.

`src/leviathan/memory_ecology.py` adds an executable L5 baseline: append-only persistent memory journal, separate current belief state vs history, contradiction-aware revisions, episodic/semantic/procedural tiers, and verification-gated promotion.

### Architecture embodiment gates

Every L0-L10 layer is tracked separately through:

`Specification -> Executable -> Integrated -> Learned -> Demonstrated`

A layer is not called achieved until all five gates pass. The ledger is `spec/architecture-maturity.toml`; `scripts/show_architecture_status.py` prints the live status.

Current development order is deliberately leverage-first rather than numerically bottom-to-top:

`L1 -> L1.5 -> L2 -> L5 -> L8 -> L6 -> L7 -> L9 -> L4 -> L3 -> L0 -> L10`.

L10 remains last because moving primary cognition into a new canonical latent begins crossing the pretrained-function-preservation wall.

### Prompt runtime

R4 has an executable prompt path rather than only a transplant specification.

Prompt an already-served OpenAI-compatible V4 instance:

```bash
python scripts/run_prompt.py \
  --backend endpoint \
  --base-url http://127.0.0.1:8000 \
  --prompt "The capital of France is"
```

Or open the interactive shell:

```bash
python scripts/run_prompt.py --backend endpoint
```

On a machine capable of loading the local full checkpoint, install the inference extras and run the donor directly:

```bash
python -m pip install -e '.[inference]'
python scripts/run_prompt.py \
  --backend transformers \
  --model-dir /models/deepseek-v4-pro-base \
  --prompt "The capital of France is"
```

The same local runner can install the deliberately slow **MoP-0 reference executor**:

```bash
python scripts/run_prompt.py \
  --backend transformers \
  --model-dir /models/deepseek-v4-pro-base \
  --mop0-reference \
  --prompt "The capital of France is"
```

Prompt-level parity is measured with:

```bash
python scripts/check_mop0_prompt_parity.py \
  --model-dir /models/deepseek-v4-pro-base \
  --prompt "The capital of France is" \
  --require-argmax-match
```

The reference executor is a correctness oracle, not a speed claim: it reconstructs each selected routed expert from all 24 tiles using the unchanged donor projections. See `docs/17-prompt-and-mop0-runtime.md`.

### Other model roles

- **Qwen3-30B-A3B-Base** — development/regression control.
- **OLMo 3 32B Base** — transparent scientific control.
- **MiMo-V2.5-Pro-Base** — frontier efficiency substrate/donor.
- **Mistral Large 3 Base** — multimodal base donor.
- **GLM/Kimi/Step/Qwen-family architectures** — architecture/efficiency lessons and compatible donors where justified.
- post-trained frontier checkpoints — teachers only where a distillation/evaluation experiment explicitly needs them.

Every architectural graft must initially preserve the pretrained function:

`h' = F_pretrained(h) + alpha * G_new(h)`, with `alpha = 0` at insertion.

See `docs/12-omega-model-soup.md` and `docs/13-weight-transplantation.md`.

## Repository map

- `docs/00-source-ledger.md` — source/evidence ledger.
- `docs/01-157-lessons.md` — numbered lesson library.
- `docs/02-master-principles.md` — architecture principles.
- `docs/03-cognitive-architecture.md` — full proposed cognitive architecture.
- `docs/04-meta-controller.md` — metacognitive controller.
- `docs/05-world-belief-model.md` — belief/world-state design.
- `docs/06-memory-and-continual-learning.md` — memory and learning hierarchy.
- `docs/07-verification-and-credit.md` — verification and causal credit.
- `docs/08-efficiency-and-inference.md` — runtime efficiency.
- `docs/09-open-stack-blueprint.md` — open implementation mapping.
- `docs/10-roadmap-and-gaps.md` — roadmap/research blockers.
- `docs/11-failure-modes.md` — failure classes.
- `docs/12-omega-model-soup.md` — donor/substrate/teacher roles.
- `docs/13-weight-transplantation.md` — function-preserving migration.
- `docs/14-omega-source-addendum.md` — source provenance notes.
- `docs/15-deepseek-v4-mop-r4.md` — DeepSeek V4 MoP protocol.
- `docs/16-deepseek-v4-mop-integration.md` — V4 integration boundary.
- `docs/17-prompt-and-mop0-runtime.md` — prompt/parity runtime.
- `docs/18-parameter-ecology-and-embodiment.md` — parameterized-cell architecture and embodiment gates.
- `spec/architecture.yaml` — machine-readable cognitive module graph and trust rules.
- `spec/interfaces.md` — shared data contracts.
- `spec/model-registry.toml` — model/checkpoint metadata.
- `spec/omega-transplant.toml` — transplantation plan.
- `spec/deepseek-v4-mop.toml` — canonical V4 fingerprint/MoP gates.
- `spec/parameter-cells.toml` — MoP-0..MoP-9 cell ecology.
- `spec/cognitive-kernel.toml` — one-model cognitive pipeline/governance invariants.
- `spec/architecture-maturity.toml` — five-gate L0-L10 embodiment ledger.
- `src/leviathan/parameter_cells.py` — zero-gated cell membrane, communication/recruitment and coalition primitives.
- `src/leviathan/cognitive_kernel.py` — compiled one-model cognitive architecture.
- `src/leviathan/memory_ecology.py` — persistent epistemic memory/belief-state baseline.
- `scripts/show_architecture_status.py` — print architecture maturity and MoP roadmap.
- `scripts/fetch_model_assets.py` — guarded model acquisition.
- `scripts/prepare_deepseek_v4_mop.py` — pinned V4 checkpoint preflight/manifest.
- `scripts/run_prompt.py` — raw/interactive V4 prompt runner.
- `scripts/check_mop0_prompt_parity.py` — original V4 vs MoP-0 logit comparison.
- `scripts/validate_model_registry.py` — registry plus architecture invariant validator.
- `models/README.md` — local checkpoint storage/reproducibility rules.

## Inspect the architecture

Validate all machine-readable invariants:

```bash
python scripts/validate_model_registry.py
```

Print the live L0-L10 and MoP-0..9 status:

```bash
python scripts/show_architecture_status.py
```

The status command is intentionally conservative: executable reference code raises an `Executable` gate, but no layer reaches `Demonstrated` without empirical evidence.

## Model assets

Model weights are deliberately **not stored in Git**. The canonical DeepSeek V4 Base checkpoint is multi-terabyte class.

List the registry:

```bash
python scripts/fetch_model_assets.py --list
```

Install optional model-download support:

```bash
python -m pip install -e '.[models]'
```

Fetch V4 metadata only:

```bash
python scripts/fetch_model_assets.py deepseek-v4-pro-base --allow-disabled
```

Validate a pinned downloaded V4 config without claiming the weights are present:

```bash
python scripts/prepare_deepseek_v4_mop.py \
  --config-only \
  --revision <immutable-hugging-face-commit>
```

Validate the complete checkpoint and emit a manifest:

```bash
python scripts/prepare_deepseek_v4_mop.py \
  --revision <immutable-hugging-face-commit> \
  --output runs/deepseek-v4-mop-manifest.json
```

Fetching all weights requires explicit opt-in and should use an immutable upstream revision:

```bash
python scripts/fetch_model_assets.py deepseek-v4-pro-base \
  --weights --allow-disabled --revision <immutable-hugging-face-commit>
```

For local prompt/parity execution install the separate inference extra:

```bash
python -m pip install -e '.[inference]'
```

## Evidence labels

Throughout the repository:

- **OPEN** — usable code plus meaningful model/training artifacts are public.
- **OPEN-WEIGHT / PARTIAL** — weights or substantial code are public, but the complete original training system is not reproducible.
- **OPEN FRAMEWORK** — the main useful contribution is infrastructure rather than a standalone foundation model.
- **CLOSED / LESSON ONLY** — the core system is proprietary; only publicly disclosed behavior or architecture lessons are used.
- **SYNTHESIS** — an architectural inference created by combining lessons across projects. It is not attributed to a single source.

## Central design rule

> **Do not spend the same kind of computation everywhere.**

Representation resolution, active parameters/cells, reasoning depth, search breadth, memory mechanism, learning mechanism, verification strength, precision and hardware path should be selected according to uncertainty, expected value, risk, cost and evidence.

## Non-claims

Leviathan does **not** claim that converting V4 MoE routing into parameter tiles/cells creates AGI. Executable reference architecture is not the same thing as learned capability. Major open work still includes learned independent cell routing, calibrated disagreement, integrated persistent memory, grounded theory-building world models, long-horizon causal credit, learned metacognitive algorithm synthesis, safe lifelong plasticity, real wall-clock sparse-cell efficiency, and the later canonical latent migration.

## Origin

This repository consolidates the architecture discussion and source inventory developed through September 2026 into one research specification and executable scaffold.

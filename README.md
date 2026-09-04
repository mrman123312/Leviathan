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

The resulting system looks less like one giant chatbot and more like a **hierarchical cognitive operating system**.

## Three-loop architecture

Leviathan separates cognition into three timescales.

### 1. Fast cognitive loop

`perception -> state update -> compute routing -> prediction -> action`

This layer borrows ideas from BLT, LongCat, Step 3.5 Flash, Qwen3-Omni, GUI-Actor, ShowUI, pi/openpi, RDT and related systems.

### 2. Deliberative loop

`belief state -> hypotheses -> plan/search/simulate -> tool or environment action -> verification -> replan`

This layer borrows from reasoning-RL systems, world models, AlphaEvolve-style population search, MiniMax-style trajectory optimization, external tools, formal verifiers and persistent agent state.

### 3. Consolidation loop

`experience -> verify -> rank -> episodic memory -> semantic abstraction -> procedural skill -> plastic update -> slow core consolidation`

This layer borrows from Letta, Mem0, Voyager, Absolute Zero Reasoner and continual-learning research.

## Leviathan Ω model layer

Leviathan is an **architecture soup, not a naive weight soup**. Unrelated giant-model tensors are not averaged together. Pretrained capability is transferred by direct compatible reuse, zero-gated grafts, latent projection bridges and distillation.

### Canonical pretrained core

**DeepSeek-V4-Pro-Base is now the canonical Leviathan semantic substrate.** Qwen3-30B-A3B-Base remains in the project as a cheaper development/regression control, but it is no longer the model that defines the primary architecture experiment.

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

At initialization, every selected expert is still reconstructed from **all** of its 24 tiles. Independent cross-expert tile routing is disabled until logit/hidden-state parity and protected benchmark gates pass. Only after parity may the router learn finer parameter composition and attempt to reduce active tiles.

Mathematical sparsity alone is not success. A lower-active-tile candidate is accepted only if **measured wall-clock efficiency** improves without capability, retention, calibration or safety loss.

See `spec/deepseek-v4-mop.toml`, `docs/15-deepseek-v4-mop-r4.md` and `docs/16-deepseek-v4-mop-integration.md`.

### Prompt runtime

R4 now has an executable prompt path rather than only a transplant specification.

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
- **GLM-5.3-Flash / Kimi K3 / Step / Qwen-family architectures** — efficiency and sequence-compute donors.
- **Kimi K3 / GLM-5.3 / Qwen3.8 / DeepSeek post-trained checkpoints** — teacher ensemble for trajectory distillation and disagreement-driven curriculum.

Every architectural graft must initially preserve the pretrained function:

`h' = F_pretrained(h) + alpha * G_new(h)`, with `alpha = 0` at insertion.

See `docs/12-omega-model-soup.md` and `docs/13-weight-transplantation.md`.

## Repository map

- `docs/00-source-ledger.md` — every system we learned from, with openness status and the lesson extracted.
- `docs/01-157-lessons.md` — the full numbered lesson library from the conversation corpus.
- `docs/02-master-principles.md` — the lessons compressed into reusable architecture principles.
- `docs/03-cognitive-architecture.md` — the full proposed Leviathan architecture.
- `docs/04-meta-controller.md` — the learned controller that decides *how to think*.
- `docs/05-world-belief-model.md` — persistent belief state, uncertainty, dynamics, provenance and active experimentation.
- `docs/06-memory-and-continual-learning.md` — working/episodic/semantic/procedural/parametric memory and safe consolidation.
- `docs/07-verification-and-credit.md` — verifier hierarchy, prediction error, causal credit assignment and anti-reward-hacking design.
- `docs/08-efficiency-and-inference.md` — token, model, context, trajectory and serving efficiency.
- `docs/09-open-stack-blueprint.md` — practical mapping from Leviathan modules to available open/open-weight projects.
- `docs/10-roadmap-and-gaps.md` — staged implementation path and remaining research blockers.
- `docs/11-failure-modes.md` — drift, forgetting, simulator bias, verifier corruption, goal drift and other failure classes.
- `docs/12-omega-model-soup.md` — substrate/donor/teacher roles and the target Leviathan Ω neural stack.
- `docs/13-weight-transplantation.md` — compatibility classes and function-preserving architecture migration.
- `docs/14-omega-source-addendum.md` — model-source provenance and role notes.
- `docs/15-deepseek-v4-mop-r4.md` — R4 execution protocol, MoP-0 parity, benchmark/efficiency gates and success definition.
- `docs/16-deepseek-v4-mop-integration.md` — canonical full-V4 integration boundary and handoff to later Leviathan layers.
- `docs/17-prompt-and-mop0-runtime.md` — prompt shell, local reference executor and prompt-level parity procedure.
- `spec/architecture.yaml` — machine-readable cognitive module graph and trust rules.
- `spec/interfaces.md` — proposed data contracts between modules.
- `spec/model-registry.toml` — model/checkpoint metadata and download policy.
- `spec/omega-transplant.toml` — machine-readable Omega transplantation plan.
- `spec/deepseek-v4-mop.toml` — canonical full-V4 fingerprint, MoP phases and acceptance gates.
- `scripts/fetch_model_assets.py` — guarded metadata/checkpoint acquisition utility.
- `scripts/prepare_deepseek_v4_mop.py` — validate a pinned local V4 checkpoint and emit the combined checkpoint/R4 MoP manifest.
- `scripts/run_prompt.py` — raw/interactive V4 prompt runner for endpoint or local Transformers execution.
- `scripts/check_mop0_prompt_parity.py` — run one prompt through original V4 and MoP-0 and measure logit drift.
- `scripts/validate_model_registry.py` — stdlib-only registry/Omega/V4 validator.
- `models/README.md` — local checkpoint storage and reproducibility rules.
- `src/leviathan/` — research scaffold for the controller, trust system, transplant state machine and V4 MoP/runtime plan.
- `vendor/` — pinned upstream Git submodules for the public source projects studied.

## Model assets

Model weights are deliberately **not stored in Git**. The canonical DeepSeek V4 Base checkpoint is multi-terabyte class.

List the registry:

```bash
python scripts/fetch_model_assets.py --list
```

Validate the registry, Omega references and V4 MoP constants:

```bash
python scripts/validate_model_registry.py
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

## The central design rule

> **Do not spend the same kind of computation everywhere.**

Representation resolution, active parameters, reasoning depth, search breadth, tool use, modality, memory mechanism, learning mechanism and verification strength should all be selected according to uncertainty, expected value, risk, cost and available evidence.

## The missing center

The strongest synthesis from all of the systems studied is a learned **metacognitive controller**. Its job is not to solve the task directly. Its job is to choose the cognitive algorithm:

`recall | direct | reason | retrieve | search | simulate | experiment | parallelize | evolve | invoke-skill | ask | act`

A conceptual objective is:

`mode* = argmax(expected success gain + information gain - compute cost - latency - risk)`

DeepSeek V4 supplies the canonical pretrained semantic engine. It does **not** replace the belief state, metacognitive controller, verifier hierarchy, memory system, causal credit assignment or governance boundary. Those remain the mechanisms that turn a foundation model into the larger Leviathan research architecture.

## Non-claims

Leviathan does **not** claim that converting V4 MoE routing into parameter tiles creates AGI. The main unsolved problems remain open-world verification, long-horizon causal credit assignment, stable lifelong belief state, safe parametric consolidation, cross-domain metacognition, simulator grounding, calibration after self-modification and governance of a self-improving learner.

## Origin

This repository consolidates the architecture discussion and source inventory developed through September 2026 into one research specification and executable scaffold.

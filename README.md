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

`perception -> adaptive representation -> parameter/compute routing -> state update -> prediction -> action`

The canonical R4 neural substrate is now the full `DeepSeek-V4-Pro-Base` pretraining checkpoint. Leviathan's Mixture-of-Parameters experiment begins by decomposing its routed experts into function-preserving, hardware-aligned parameter tiles before any attempt to reduce active compute.

### 2. Deliberative loop

`belief state -> hypotheses -> plan/search/simulate -> tool or environment action -> verification -> replan`

This layer borrows from reasoning-RL systems, world models, AlphaEvolve-style population search, MiniMax-style trajectory optimization, external tools, formal verifiers and persistent agent state.

### 3. Consolidation loop

`experience -> verify -> rank -> episodic memory -> semantic abstraction -> procedural skill -> plastic update -> slow core consolidation`

This layer borrows from Letta, Mem0, Voyager, Absolute Zero Reasoner and continual-learning research.

## Leviathan Ω model layer

Leviathan is an **architecture soup, not a naive weight soup**. Unrelated giant-model tensors are not averaged together. Pretrained capability is transferred by direct compatible reuse, zero-gated grafts, latent projection bridges and distillation.

Current role split:

- **DeepSeek-V4-Pro-Base** — canonical R4 and Ω-L pretrained semantic substrate; the full-scale Mixture-of-Parameters target.
- **Qwen3-30B-A3B-Base** — cheaper development/regression fallback, no longer the canonical R4 substrate.
- **OLMo 3 32B Base** — transparent scientific control.
- **MiMo-V2.5-Pro-Base** — frontier efficiency substrate/donor.
- **Mistral Large 3 Base** — multimodal base donor.
- **GLM-5.3-Flash / Kimi K3 / Step / Qwen-family architectures** — efficiency and sequence-compute donors.
- **Kimi K3 / GLM-5.3 / Qwen3.8 / DeepSeek post-trained checkpoints** — teacher ensemble for trajectory distillation and disagreement-driven curriculum.

Every architectural graft must initially preserve the pretrained function:

`h' = F_pretrained(h) + alpha * G_new(h)`, with `alpha = 0` at insertion.

For R4 MoP, the equivalent rule is even stricter: the first parameter-tile representation must exactly reconstruct each originally routed SwiGLU expert before independent tile routing is trained.

See `docs/12-omega-model-soup.md`, `docs/13-weight-transplantation.md` and `docs/15-deepseek-v4-mop-r4.md`.

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
- `docs/15-deepseek-v4-mop-r4.md` — full-scale R4 DeepSeek V4 Mixture-of-Parameters migration and benchmark gates.
- `spec/architecture.yaml` — machine-readable cognitive module graph and trust rules.
- `spec/interfaces.md` — proposed data contracts between modules.
- `spec/model-registry.toml` — model/checkpoint metadata and download policy.
- `spec/omega-transplant.toml` — machine-readable Omega transplantation plan.
- `spec/deepseek-v4-mop.toml` — pinned V4 architecture contract, parameter-tile geometry and R4 rejection gates.
- `scripts/fetch_model_assets.py` — guarded metadata/checkpoint acquisition utility.
- `scripts/prepare_deepseek_v4_mop.py` — validates a pinned V4 config and emits the R4 transplant manifest without loading weights.
- `scripts/validate_model_registry.py` — stdlib-only registry/reference/R4 invariant validator.
- `models/README.md` — local checkpoint storage and reproducibility rules.
- `src/leviathan/deepseek_v4_mop.py` — exact V4 parameter-tile plan and checkpoint-config validation.
- `src/leviathan/` — minimal research scaffold for the meta-controller, trust weighting and shared types.
- `vendor/` — pinned upstream Git submodules for the public source projects studied.

## R4: full DeepSeek V4 Mixture-of-Parameters

The current V4 routed expert has 3,072 SwiGLU intermediate channels. R4 starts with 128-channel tiles, producing 24 tiles per routed expert. Across 384 routed experts this is 9,216 routed tiles per layer. The original six-expert route expands to 144 tiles at MoP-0 and must reproduce the pretrained function before any sparsity is introduced.

The first stage is deliberately **not** a speed claim. It is the exact bridge from pretrained MoE to parameter-level routing. Later stages may select fewer tiles only if protected capability, held-out language loss and real wall-clock/HBM measurements pass.

ARC-Easy is a named canary and is reported separately for every R4 candidate. The 64-passage held-out WikiText gate may not be used for training or hyperparameter selection.

## Model assets

Model weights are deliberately **not stored in Git**. The DeepSeek V4 checkpoint is a frontier-scale external asset and remains disabled for automatic download unless explicitly overridden.

List the registry:

```bash
python scripts/fetch_model_assets.py --list
```

Validate the registry, Omega references and V4 MoP contract:

```bash
python scripts/validate_model_registry.py
```

Validate a pinned local V4 configuration and create the R4 manifest:

```bash
python scripts/prepare_deepseek_v4_mop.py \
  --config /models/deepseek-v4-pro-base/config.json \
  --revision <immutable-hugging-face-commit> \
  --output models/fingerprints/deepseek-v4-r4.json
```

Install optional model-download support:

```bash
python -m pip install -e '.[models]'
```

Metadata-only fetch remains the safe default. Full V4 weights require explicit `--weights`, `--allow-disabled`, sufficient storage/compute, and an immutable revision.

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

For R4 this has an extra systems rule: **mathematical sparsity is not efficiency unless real hardware measurements improve too.** If Mixture-of-Parameters activates fewer weights but makes V4 slower, that candidate is rejected.

## The missing center

The strongest synthesis from all of the systems studied is a learned **metacognitive controller**. Its job is not to solve the task directly. Its job is to choose the cognitive algorithm:

`recall | direct | reason | retrieve | search | simulate | experiment | parallelize | evolve | invoke-skill | ask | act`

A conceptual objective is:

`mode* = argmax(expected success gain + information gain - compute cost - latency - risk)`

The parameter router and the metacognitive controller are related but not identical: one decides how much/which neural capacity to activate, while the other decides what kind of cognition to perform.

## Non-claims

Leviathan does **not** claim that replacing V4's MoE router with Mixture-of-Parameters automatically improves the model, nor that wiring these projects together creates AGI. R4 keeps the original V4 MoE as the fallback and rejects MoP if it fails real capability/efficiency gates.

The main unsolved problems remain open-world verification, long-horizon causal credit assignment, stable lifelong belief state, safe parametric consolidation, cross-domain metacognition, simulator grounding, calibration after self-modification and governance of a self-improving learner.

## Origin

This repository consolidates the architecture discussion and source inventory developed through September 2026 into one research specification and executable scaffold.

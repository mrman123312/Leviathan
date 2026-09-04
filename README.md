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

Current role split:

- **Qwen3-30B-A3B-Base** — primary development-scale architecture-surgery substrate.
- **OLMo 3 32B Base** — transparent scientific control.
- **DeepSeek-V4-Pro-Base** — preferred frontier semantic substrate after smaller-scale mechanisms are proven.
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
- `spec/architecture.yaml` — machine-readable cognitive module graph and trust rules.
- `spec/interfaces.md` — proposed data contracts between modules.
- `spec/model-registry.toml` — model/checkpoint metadata and download policy.
- `spec/omega-transplant.toml` — machine-readable Omega transplantation plan.
- `scripts/fetch_model_assets.py` — guarded metadata/checkpoint acquisition utility.
- `scripts/validate_model_registry.py` — stdlib-only registry/reference validator.
- `models/README.md` — local checkpoint storage and reproducibility rules.
- `src/leviathan/` — minimal research scaffold for the meta-controller, trust weighting and shared types.
- `vendor/` — pinned upstream Git submodules for the public source projects studied.

## Model assets

Model weights are deliberately **not stored in Git**. Some frontier checkpoints are multi-terabyte assets.

List the registry:

```bash
python scripts/fetch_model_assets.py --list
```

Validate the registry and Omega references:

```bash
python scripts/validate_model_registry.py
```

Install optional model-download support:

```bash
python -m pip install -e '.[models]'
```

Metadata-only fetch is the default:

```bash
python scripts/fetch_model_assets.py qwen3-30b-a3b-base
```

Weights require explicit `--weights`; frontier entries additionally require `--allow-disabled`. Pin an immutable upstream revision for reproducible experiments.

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

## Non-claims

Leviathan does **not** claim that simply wiring these projects together creates AGI. The main unsolved problems remain open-world verification, long-horizon causal credit assignment, stable lifelong belief state, safe parametric consolidation, cross-domain metacognition, simulator grounding, calibration after self-modification and governance of a self-improving learner.

## Origin

This repository consolidates the architecture discussion and source inventory developed through September 2026 into one research specification and executable scaffold.

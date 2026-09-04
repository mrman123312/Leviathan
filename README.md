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
- `spec/architecture.yaml` — machine-readable module graph and trust rules.
- `spec/interfaces.md` — proposed data contracts between modules.
- `src/leviathan/` — minimal research scaffold for the meta-controller, trust weighting and shared types.

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

This repository consolidates the complete architecture discussion and source inventory developed through September 2026 into one clean research specification.

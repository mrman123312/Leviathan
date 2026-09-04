# Leviathan Ω — model substrate and teacher architecture

This document turns the open-weight model survey into a concrete design for the neural substrate of Leviathan.

## Objective

Leviathan is not a weight merge. It is a staged architecture-transplantation project whose target is:

`maximum verified capability / (active compute × memory traffic × latency × learning instability)`

The system uses one canonical pretrained cognitive core plus external donors, teachers, tools and verifiers. Donors and teachers do not become a runtime civilization of foundation models pretending to be one model.

## Role A — canonical semantic substrate

### DeepSeek-V4-Pro-Base

**DeepSeek-V4-Pro-Base is the canonical pretrained semantic core for the current Leviathan experiment.**

Role: inherit large-scale language, code, mathematics and semantic representations from a true base checkpoint, then evolve the same model into the Leviathan substrate without discarding its pretrained function.

Why it is preferred:

- true base/pretraining checkpoint rather than only chat behavior;
- 1.6T-class sparse capacity with much lower active compute than total parameters;
- long-context architecture;
- existing sparse routing and future-prediction lineage relevant to Leviathan;
- permissive license relative to many frontier-weight releases;
- enough pretrained knowledge that Leviathan does not need to relearn language and world regularities from scratch;
- expert structure that can be decomposed into finer parameter tiles without first inventing specialization from random initialization.

The core rule remains **inherit first, mutate second**. The initial Leviathan representation must reproduce the base model's function before original paths are relaxed or retired.

The exact canonical V4 fingerprint and Mixture-of-Parameters migration are specified in `spec/deepseek-v4-mop.toml`, `docs/15-deepseek-v4-mop-r4.md` and `docs/16-deepseek-v4-mop-integration.md`.

### Qwen3-30B-A3B-Base — development control

Qwen remains useful, but it is no longer the canonical substrate.

Role:

- cheaper architecture-surgery control;
- fast regression reproduction;
- interface tests;
- zero-gated module tests;
- memory/controller plumbing;
- training-loop debugging before expensive V4 runs.

A Qwen result is not a substitute for a canonical full-V4 result. The V4 experiment has an explicit full-checkpoint fingerprint so reduced models cannot be reported as Leviathan Ω-L.

### Scientific control: OLMo 3 32B Base

Role: transparent control model for measuring architectural effects against a substantially open training lineage.

Use OLMo when the research question is not simply "does this improve the model?" but "where in training should this mechanism be introduced, and what changed?"

## Mixture-of-Parameters as the V4 substrate migration

DeepSeek V4 begins with routed experts. Leviathan does not immediately throw away this routing.

The first conversion decomposes each routed expert's 3072-wide SwiGLU intermediate space into contiguous 128-channel tiles:

- `3072 / 128 = 24` tiles per expert;
- `384 * 24 = 9,216` routed tiles per layer;
- the inherited 6-expert route expands to `6 * 24 = 144` routed tiles per token at the parity stage.

At initialization:

`selected expert -> every constituent tile of that expert`

Therefore the original expert computation remains reconstructable. Independent tile composition across experts is disabled until parity passes.

Only later does the router learn:

`whole-expert routing -> tile routing -> cross-expert tile composition -> reduced active tile budget`

A lower active-parameter count does not count as an efficiency win unless real wall-clock metrics also improve. This directly incorporates the R3 lesson that mathematical sparsity can be slower in practice when routing and memory movement dominate.

## Role B — efficiency-architecture donors

### MiMo-V2.5-Pro-Base

Primary lessons to transplant:

- mostly-local attention with periodic global integration;
- low active/total parameter ratio;
- multi-token prediction;
- long-context KV-cache economy;
- sparse MoE routing.

Leviathan should test whether a mostly-local/periodic-global pattern can replace more expensive attention paths while preserving the inherited V4 function.

### GLM-5.3-Flash

Primary lessons:

- extremely low active compute relative to total capacity;
- hybrid sparse/linear attention;
- mHC-style richer residual connectivity;
- native multimodal efficiency;
- million-token operating regime.

GLM-5.3-Flash is treated primarily as an **architecture and behavior donor**, not as the clean canonical base substrate.

### Kimi K3

Primary lessons:

- KDA/recurrent-efficient sequence processing;
- periodic stronger global attention;
- Attention Residuals across depth;
- Stable LatentMoE;
- quantization-native design;
- long-horizon agent training with persistent state;
- vision-in-the-loop artifact correction.

Kimi's scale is not itself the feature to transplant. Reproduce useful mechanisms against the inherited V4 function and retain them only when quality-per-compute improves.

### Step 3.5 Flash / Qwen-family MTP

Primary lesson: reduce sequential decode dependence by predicting several future symbols/states per expensive representation.

Leviathan extends the idea beyond text:

`h_t -> {future_tokens, future_actions, future_latent_states, future_verifier_outcomes}`

## Role C — multimodal donors

### Mistral Large 3 Base

Role: clean multimodal base donor.

Use the vision encoder or learned representations through a projection bridge rather than attempting incompatible raw weight transplantation.

A generic bridge is:

`z_vision -> P_vision(z_vision) -> z_Leviathan`

where both large pretrained systems are frozen while the projector is initially trained.

### Qwen3-Omni

Role: audio/video/speech architecture donor.

Primary lesson: semantic reasoning and modality realization should be separable. Leviathan should not require its most expensive cognitive core to generate every acoustic or motor detail.

## Role D — post-training teachers

Teacher models are not the deployed cognitive substrate. They are offline training/evaluation resources used to create, compare and verify trajectories.

Candidate teacher ensemble:

- Kimi K3;
- GLM-5.3;
- Qwen3.8-2.4T-A95B;
- DeepSeek-V4-Pro post-trained checkpoints;
- GLM-5.3-Flash for efficient solution proposals.

The ensemble should not be distilled by naive majority vote.

For task `x`, model the target as a task-conditioned mixture:

`P*(y|x) = sum_i w_i(x) P_i(y|x)`

Teacher weights should depend on demonstrated domain competence and verifier evidence.

### Teacher disagreement becomes curriculum

When teachers strongly agree and a trusted verifier agrees, the trajectory is a high-quality candidate for distillation.

When teachers disagree, do **not** train on an arbitrary winner. Route the item into:

1. deterministic/formal verification if available;
2. retrieval of external evidence;
3. environment execution;
4. additional search/experimentation;
5. a curriculum queue if uncertainty remains informative.

High disagreement is a signal that the sample may lie near the current capability frontier. It is not itself truth.

## Heterogeneous experts

The long-term Leviathan routing substrate should not require every routed operation to remain a conventional MLP.

Candidate operation classes:

- semantic FFN computation;
- code computation;
- mathematical computation;
- memory retrieval;
- recurrent world-state update;
- causal-model operation;
- planning operation;
- simulation operation;
- verifier interface.

The router therefore evolves from "which expert MLP processes this token?" toward "which parameter/cognitive operation should process this state?"

This migration must be gradual. Heterogeneous paths begin with zero route probability and do not replace inherited V4 computation until matched tests pass.

## Function-preserving transplantation

Every architectural graft starts with zero or identity influence.

For a new module `G` around a pretrained function `F`:

`h' = F(h) + alpha * G(h)`

Initialize `alpha = 0`.

Required sequence:

1. load and fingerprint the full pretrained V4 substrate;
2. retain a restorable locked baseline;
3. insert new modules with zero/identity effect;
4. train only new parameters where possible;
5. demonstrate logit/hidden-state/perplexity/behavior parity;
6. gradually increase gates or routing freedom;
7. selectively unfreeze the smallest necessary old parameter groups;
8. run continued pretraining/post-training with replay and stability losses;
9. verify capability retention and calibration;
10. shadow-evaluate candidate;
11. only then retire redundant old paths.

For the MoP conversion, expert-channel tiling itself is the function-preserving representation step. Independent cross-expert tile routing is a later learned change.

## Architecture migration, not weight soup

Do **not** numerically average unrelated checkpoints such as DeepSeek, Mistral, Kimi and GLM. Their tensor semantics, widths, attention mechanisms and tokenizers are not generally compatible.

Use one of four transfer methods instead:

1. **direct reuse** — preserve compatible pretrained blocks;
2. **gated graft** — add new modules while preserving old function;
3. **projection bridge** — connect independently pretrained modalities/subsystems in latent space;
4. **distillation** — teach a structurally different Leviathan component to reproduce teacher behavior or representations.

## Target Leviathan Ω stack

```text
multimodal encoders
      |
      v
DeepSeek-V4-derived Leviathan semantic core
      |
      v
shared belief/state space <---- episodic + semantic memory
      |
      v
metacognitive router
      |
      +--> direct sparse/MoP core computation
      +--> procedural skill
      +--> retrieval
      +--> world simulation
      +--> tree/population search
      +--> experiment
      +--> tool/API execution
      |
      v
hierarchical planner
      |
      v
native action heads
      |
      v
environment
      |
      v
independent verifier hierarchy
      |
      v
causal credit assignment
      |
      +--> memory promotion
      +--> procedural compilation
      +--> plastic candidate
      +--> slow verified core consolidation
```

The semantic core remains one model. External tools/verifiers/memory services are resources, not additional hidden cognitive models merged at runtime.

## Scale ladder and current experiment

The scale ladder remains useful for cheap controls, but it no longer defines the canonical checkpoint.

### Ω-S0 — 3B–30B controls

Use Qwen-class models to debug interfaces, training code and safety invariants cheaply.

### Ω-S1 — 100B–300B sparse controls

Use medium sparse models to stress distributed routing, adaptive expert count, recurrent/local/global hybrids, MTP extensions and plastic-module tooling.

### Ω-S2 — ~1T controls

Use MiMo-class or equivalent true bases to test whether architecture effects survive giant sparse representations.

### Ω-L — canonical

Use the **full DeepSeek-V4-Pro-Base** checkpoint as the canonical Leviathan neural-substrate experiment.

Smaller controls remain scientifically valuable, but a mechanism does not need to wait for every smaller rung if we intentionally choose to test it on Ω-L. Conversely, success on Ω-L does not excuse a failure to measure retention, calibration, safety or real efficiency.

## Non-negotiable evaluation dimensions

No architecture change is accepted solely because benchmark accuracy rises or active parameters fall.

Track simultaneously:

- task success;
- calibration;
- retention of old capability;
- held-out language loss;
- ARC-Easy canary behavior;
- active parameters/token;
- active tiles/token;
- HBM bytes/token;
- KV/state bytes/token;
- routing/communication overhead;
- single-stream and aggregate throughput;
- reasoning tokens/success;
- tool calls/success;
- wall-clock time/success;
- energy/compute proxy;
- verifier agreement;
- rollback success;
- catastrophic-forgetting score;
- adversarial robustness;
- simulator-vs-reality gap.

The target is not the largest model. The target is the **highest verified cognitive work per unit of lifetime computation**.

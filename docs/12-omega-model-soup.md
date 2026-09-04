# Leviathan Ω — model substrate and teacher architecture

This document turns the open-weight model survey into a concrete design for the neural substrate of Leviathan.

## Objective

Leviathan is not a weight merge. It is a staged architecture-transplantation project whose target is:

`maximum verified capability / (active compute × memory traffic × latency × learning instability)`

The model stack is therefore split into roles rather than numerically merging unrelated checkpoints.

## Role A — primary semantic substrate

### Canonical R4 / frontier substrate: DeepSeek-V4-Pro-Base

Role: inherit large-scale language, code, mathematics and semantic representations from a true base checkpoint, then grow Leviathan modules around it.

The current R4 decision is to perform the Mixture-of-Parameters experiment directly on the full DeepSeek-V4-Pro-Base class rather than treating a small Qwen model as the canonical test. This exception is specific to the parameter-substrate question: MoP is intended for giant sparse models where routing granularity, HBM traffic and distributed execution economics are materially different from tiny models.

Why DeepSeek V4 is preferred:

- true base/pretraining checkpoint rather than only chat behavior;
- very large sparse MoE capacity with far lower active compute than total parameter count;
- a routed SwiGLU expert structure that can be decomposed into exact channel tiles;
- FP8 expert blocks that give MoP a hardware-aligned starting granularity;
- long-context architecture;
- MTP/sparse-attention lineage relevant to Leviathan;
- enough pretrained knowledge that Leviathan does not need to relearn language and world regularities from scratch.

The core rule remains **inherit first, mutate second**. MoP-0 must reproduce the base model's function before independent parameter-tile routing is permitted.

See `docs/15-deepseek-v4-mop-r4.md` and `spec/deepseek-v4-mop.toml` for the exact R4 contract.

### Development fallback: Qwen3-30B-A3B-Base

Role: cheap implementation debugging, regression reproduction and interface tests.

Qwen can still be used to test:

- zero-gated module insertion;
- expert expansion;
- persistent-state channels;
- memory experts;
- heterogeneous expert routing;
- MTP extensions;
- continual adapters;
- verifier-conditioned learning;
- regression and rollback tooling.

It is **not** the canonical R4 MoP substrate. A result on Qwen does not substitute for the full V4 experiment.

### Scientific control: OLMo 3 32B Base

Role: transparent control model for measuring architectural effects against a substantially open training lineage.

Use OLMo when the research question is not simply "does this improve the model?" but "where in training should this mechanism be introduced, and what changed?"

## Role B — efficiency-architecture donors

### MiMo-V2.5-Pro-Base

Primary lessons to transplant:

- mostly-local attention with periodic global integration;
- low active/total parameter ratio;
- multi-token prediction;
- long-context KV-cache economy;
- sparse MoE routing.

Leviathan should test whether a mostly-local/periodic-global pattern can replace more expensive attention paths while preserving the inherited base function.

### GLM-5.3-Flash

Primary lessons:

- extremely low active compute relative to total capacity;
- hybrid sparse/linear attention;
- mHC-style richer residual connectivity;
- native multimodal efficiency;
- million-token operating regime.

GLM-5.3-Flash is treated primarily as an **architecture and behavior donor**, not as the initial clean base substrate.

### Kimi K3

Primary lessons:

- KDA/recurrent-efficient sequence processing;
- periodic stronger global attention;
- Attention Residuals across depth;
- Stable LatentMoE;
- quantization-native design;
- long-horizon agent training with persistent state;
- vision-in-the-loop artifact correction.

Kimi's scale is not required for every donor experiment. Reproduce donor primitives at the cheapest scale that can answer the actual research question.

### Step 3.5 Flash / Qwen-family MTP

Primary lesson: reduce sequential decode dependence by predicting several future symbols/states per expensive representation.

Leviathan extends the idea beyond text:

`h_t -> {future_tokens, future_actions, future_latent_states, future_verifier_outcomes}`

## Role C — multimodal donors

### Mistral Large 3 Base

Role: clean multimodal base donor.

Use the vision encoder or learned representations through a projection bridge rather than attempting incompatible raw weight transplantation.

A generic bridge is:

`z_vision -> P_vision(z_vision) -> Leviathan belief space`

where both large pretrained systems are frozen while the projector is initially trained.

### Qwen3-Omni

Role: audio/video/speech architecture donor.

Primary lesson: semantic reasoning and modality realization should be separable. Leviathan should not require its most expensive cognitive core to generate every acoustic or motor detail.

## Role D — post-training teachers

Teacher models are not necessarily the substrate. They are used to create, compare and verify trajectories.

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

High disagreement is a signal that the sample lies near the current capability frontier.

## Mixture-of-Parameters in the neural substrate

R4 changes routing granularity inside the existing DeepSeek routed FFN before it changes any other major V4 subsystem.

The initial contract is:

- keep attention unchanged;
- keep mHC/residual topology unchanged;
- keep the shared expert unchanged;
- do not average experts;
- do not route individual scalar parameters;
- split each routed expert's 3,072 intermediate channels into hardware-aligned tiles;
- preserve the original expert router through the early MoP stages;
- reject sparsity that is slower on real hardware.

At the default 128-channel width:

`3,072 / 128 = 24 tiles/expert`

`384 experts × 24 = 9,216 routed tiles/layer`

`6 originally active experts × 24 = 144 tiles/token at MoP-0`

MoP-0 changes representation, not compute. Speed/capacity gains are claimed only after later tile routing activates fewer tiles while retention and wall-clock gates pass.

## Heterogeneous cognitive operations

A long-term Leviathan system should not require every cognitive operation to be an MLP expert.

Candidate operation classes include:

- semantic FFN processing;
- code/math specialization;
- memory retrieval;
- recurrent world-state update;
- causal-model evaluation;
- planning;
- simulation;
- verifier interaction.

This later metacognitive routing problem is distinct from R4's parameter-tile router. One asks **which neural capacity should activate?** The other asks **which cognitive procedure should run?**

## Function-preserving transplantation

Every architectural graft starts with zero or identity influence.

For a new module `G` around a pretrained function `F`:

`h' = F(h) + alpha * G(h)`

Initialize `alpha = 0`.

Required sequence:

1. load and fingerprint the pretrained substrate;
2. insert a zero/identity or exact-equivalent representation;
3. demonstrate behavioral and perplexity parity;
4. train only new parameters;
5. gradually increase new-path authority;
6. selectively unfreeze compatible old parameters only when necessary;
7. run continued pretraining/post-training where justified;
8. verify capability retention and calibration;
9. shadow-evaluate candidate;
10. only then retire redundant old paths.

For MoP specifically, exact channel decomposition replaces the generic zero gate at MoP-0: the tiled representation must reconstruct the original expert function.

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
DeepSeek V4 semantic/parameter substrate <--- hardware-aware MoP routing
      |
      v
shared belief/state space <---- episodic + semantic memory
      |
      v
metacognitive router
      |
      +--> direct neural core
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
      +--> plastic adapter candidate
      +--> slow verified core consolidation
```

## Development scale ladder

The general Leviathan ladder remains useful, but scale must match the hypothesis being tested.

### Ω-S0 — 3B–30B

Use for interfaces, safety invariants, memory/controller logic and cheap architecture debugging.

### Ω-S1 — 100B–300B sparse

Use for distributed sparse-kernel and heterogeneous-routing prototypes where that scale is sufficient.

### Ω-S2 — ~1T

Use for giant-scale transfer checks when a 1T-class donor answers the question more economically.

### Ω-L — 1.6T+

Current R4 MoP target: full DeepSeek-V4-Pro-Base.

This does **not** mean every Leviathan feature skips directly to Ω-L. It means the parameter-routing hypothesis is being tested at the scale where it is intended to matter. If a smaller reproducer can diagnose an implementation bug, use it; if scale changes the economics, the Ω-L result is authoritative.

## Non-negotiable R4 gates

No architecture change is accepted solely because benchmark accuracy rises or theoretical active parameters fall.

Track simultaneously:

- task success;
- ARC-Easy as a named reasoning canary;
- held-out WikiText/public-language loss;
- calibration;
- retention of old capability;
- active parameters/token;
- active parameter tiles/token;
- HBM bytes/token;
- KV/state bytes/token;
- single-stream and aggregate throughput;
- wall-clock time/success;
- routing/kernel overhead;
- rollback success;
- catastrophic-forgetting score;
- adversarial robustness.

The 64 held-out WikiText passages are excluded from training and hyperparameter selection. The hard public-language rejection boundary remains +2% relative loss, with a much stricter +0.25% target.

If MoP activates fewer parameters but makes real execution slower, reject it. If it improves speed but causes an unexplained protected-capability regression, reject it. If no MoP setting beats or matches the original V4 MoE under the complete objective, keep the original V4 MoE and continue the rest of Leviathan.

The target is not the largest model or the sparsest routing pattern. The target is the **highest verified cognitive work per unit of lifetime computation**.

# Leviathan Ω — model substrate and teacher architecture

This document turns the open-weight model survey into a concrete design for the neural substrate of Leviathan.

## Objective

Leviathan is not a weight merge. It is a staged architecture-transplantation project whose target is:

`maximum verified capability / (active compute × memory traffic × latency × learning instability)`

The model stack is therefore split into four roles rather than forcing one checkpoint to do everything.

## Role A — primary semantic substrate

### Preferred frontier substrate: DeepSeek-V4-Pro-Base

Role: inherit large-scale language, code, mathematics and semantic representations from a true base checkpoint, then grow Leviathan modules around it.

Why it is preferred at frontier scale:

- true base/pretraining checkpoint rather than only chat behavior;
- very large sparse MoE capacity with far lower active compute than total parameter count;
- long-context architecture;
- existing MTP/sparse-attention lineage relevant to Leviathan;
- permissive license relative to many frontier-weight releases;
- enough pretrained knowledge that Leviathan does not need to relearn language and world regularities from scratch.

The core rule is **inherit first, mutate second**. The initial Leviathan system must reproduce the base model's function before expensive original paths are retired.

### Preferred experimental substrate: Qwen3-30B-A3B-Base

Role: development-scale surgery.

Use it to test:

- zero-gated module insertion;
- expert expansion;
- persistent-state channels;
- memory experts;
- heterogeneous parameter-basis routing inside one checkpoint;
- MTP extensions;
- continual adapters;
- verifier-conditioned learning;
- regression and rollback tooling.

Architecture experiments should fail here before they are ever attempted on a trillion-parameter checkpoint.

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

Kimi's 2.8T scale is not required to test these lessons. Reproduce the primitives at development scale first.

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

Teacher models are never the deployed agent. They are offline sources used to create
and compare candidate training trajectories. Any retained information must be distilled
or transplanted into one student parameter state before runtime.

Candidate teacher ensemble:

- Kimi K3;
- GLM-5.3;
- Qwen3.8-2.4T-A95B;
- DeepSeek-V4-Pro post-trained checkpoints;
- GLM-5.3-Flash for efficient solution proposals.

The offline set should not be distilled by naive majority vote, and its members must
never co-infer as Leviathan's runtime cognition.

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

## Heterogeneous experts

A long-term Leviathan MoE should not require every expert to be an MLP.

Candidate expert classes:

- semantic FFN expert;
- code expert;
- mathematical expert;
- memory-retrieval expert;
- recurrent world-state expert;
- causal-model expert;
- planning expert;
- simulation expert;
- verifier-interface expert.

The router therefore evolves from "which MLP processes this token?" toward "which cognitive operation should process this state?"

## Function-preserving transplantation

Every architectural graft starts with zero or identity influence.

For a new module `G` around a pretrained function `F`:

`h' = F(h) + alpha * G(h)`

Initialize `alpha = 0`.

Required sequence:

1. load and freeze pretrained substrate;
2. insert new modules with zero/identity effect;
3. train only new parameters;
4. demonstrate behavioral and perplexity parity;
5. gradually increase gates;
6. selectively unfreeze compatible old parameters;
7. run continued pretraining/post-training;
8. verify capability retention and calibration;
9. shadow-evaluate candidate;
10. only then retire redundant old paths.

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
shared belief/state space <---- episodic + semantic memory
      |
      v
metacognitive router
      |
      +--> direct sparse core
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

### Ω-S0 — 3B–30B

Prove interfaces and safety invariants.

Success criteria:

- persistent belief state improves long tasks;
- memory reduces repeated reasoning without accuracy loss;
- meta-controller learns to select cognition modes;
- verifier-aware routing improves calibration;
- new modules can be added without measurable base-model regression.

### Ω-S1 — 100B–300B sparse

Prove:

- heterogeneous experts;
- adaptive expert count;
- recurrent/local/global hybrid sequence processing;
- MTP state/action prediction;
- safe plastic adapters;
- distributed serving efficiency.

### Ω-S2 — ~1T

Use MiMo-class or equivalent true base substrate to test whether architecture gains survive giant-scale sparse pretraining representations.

### Ω-L — 1.6T+

Frontier transplantation into a DeepSeek-V4-Pro-Base-class substrate after the smaller architecture has passed retention, safety, calibration and efficiency gates.

## Non-negotiable evaluation dimensions

No architecture change is accepted solely because benchmark accuracy rises.

Track simultaneously:

- task success;
- calibration;
- retention of old capability;
- active parameters/token;
- HBM bytes/token;
- KV/state bytes/token;
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

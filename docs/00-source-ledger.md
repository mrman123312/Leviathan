# Source Ledger

This document records every named system used in the Leviathan synthesis, what is actually public, and the specific engineering lesson extracted from it. Openness labels are deliberately conservative: a GitHub repository does not automatically mean the full research system is reproducible.

## Legend

- **OPEN** — usable code plus meaningful model/training artifacts.
- **OPEN-WEIGHT / PARTIAL** — public weights or substantial code, incomplete training stack or system.
- **OPEN FRAMEWORK** — reusable infrastructure rather than a single foundation model.
- **CLOSED / LESSON ONLY** — proprietary system; only publicly disclosed lessons are used.

---

## World models and predictive state

### NVIDIA Cosmos / Cosmos Predict
**Status:** OPEN-WEIGHT / PARTIAL  
**Source:** https://github.com/NVIDIA/Cosmos  
**Related:** https://github.com/nvidia-cosmos/cosmos-predict2.5

**Lessons**
- Future-state prediction should be action-conditioned.
- World understanding and world generation can share representations.
- Physical reasoning benefits from explicit predictive dynamics rather than language-only continuation.
- Simulation should support policy/value reasoning, not only photorealistic generation.

### Meta V-JEPA 2
**Status:** OPEN  
**Source:** https://github.com/facebookresearch/vjepa2

**Lessons**
- Predict latent future representations instead of reconstructing every pixel.
- A useful world model can be compact and task-relevant rather than generatively complete.
- Latent predictive representations can support physical reasoning and planning.

### JEPA-WM
**Status:** OPEN  
**Source:** https://github.com/facebookresearch/jepa-wms

**Lessons**
- Planning can occur directly in latent predictive space.
- Counterfactual action evaluation does not require full image generation.

---

## Reasoning and reinforcement learning

### DeepSeek-R1 / R1-Zero
**Status:** OPEN-WEIGHT / PARTIAL  
**Source:** https://github.com/deepseek-ai/DeepSeek-R1

**Lessons**
- Reasoning procedures can emerge from reinforcement learning rather than being fully scripted by supervised traces.
- Pure outcome-driven RL can produce useful reasoning but also repetition, readability issues and other policy pathologies.
- Exploration and behavioral scaffolding are complementary.
- Expensive search discovered during training can later become cheaper learned behavior.

### Qwen3 / Qwen3 Thinking
**Status:** OPEN-WEIGHT / PUBLIC CODE  
**Source:** https://github.com/QwenLM/Qwen3

**Lessons**
- One model can expose multiple reasoning-effort policies.
- Extended reasoning should be optional rather than permanently enabled.
- Compute policy can be conditioned on task difficulty.

### K2 Horizon / IFM
**Status:** OPEN / unusually reproducible release  
**Source organization:** https://github.com/ifm-ai

**Lessons**
- Intermediate checkpoints, logs, data recipes and training configurations are themselves research artifacts.
- Capability emergence is easier to study when the full training trajectory is published.
- A single methodology can span small and frontier-scale models.

### veRL
**Status:** OPEN FRAMEWORK  
**Source:** https://github.com/verl-project/verl

**Lessons**
- Rollout generation, reward computation, distributed execution, model serving and policy updates should be modular.
- The learning loop should support multiple policy-gradient and verifiable-reward methods without being architecturally coupled to one reward.

---

## Agentic multimodal foundation models

### Kimi K3
**Status:** OPEN-WEIGHT / PARTIAL  
**Source:** https://github.com/MoonshotAI/Kimi-K3

**Lessons**
- Sparse MoE separates total knowledge capacity from active compute.
- Latent expert spaces can reduce communication and expert compute.
- Full global attention need not appear at every layer.
- Deep representations can be selectively revisited through attention-residual-style mechanisms.
- Low-precision inference works better when quantization is part of training rather than an afterthought.
- Agentic ability should be trained on full trajectories with tools and environment feedback.
- Reasoning/task state should persist across long tool interactions.
- Vision-in-the-loop enables generate-render-inspect-repair workflows.

### GPT-5.6 Sol / Luna
**Status:** CLOSED / LESSON ONLY

**Lessons**
- Optimize work-per-token, not just next-token quality.
- Reward shorter successful trajectories.
- Let generated programs orchestrate deterministic tool work without returning to the expensive model after every intermediate step.
- Stable context should be cacheable and separable from changing context.

### GPT Astra
**Status:** CLOSED / LESSON ONLY

**Lessons**
- Preserve task/reasoning state across actions.
- Long-lived agents need compact notes plus searchable historical state.
- More reasoning compute can mean more tests, experiments and verification rather than a longer monologue.
- Actions can be chosen partly for information gain.
- Environmental feedback is often stronger than self-judgment.
- Tool use and reasoning should support asynchronous parallelism.
- Reasoning effort can change in the middle of a task.

### Claude Fable 5.1
**Status:** CLOSED / LESSON ONLY

**Lessons**
- Reasoning effort should adapt to difficulty and uncertainty.
- Per-step reasoning budget and whole-task budget are different control variables.
- Prompt/prefix caching is central to long agent loops.
- Tool schemas should be retrieved only when needed.
- Deterministic intermediate work should happen outside the LLM.
- Programmatic tool orchestration can dramatically reduce token traffic.

### Dolphin 3.0 R1 Mistral 24B
**Status:** OPEN-WEIGHT  
**Source:** https://huggingface.co/dphn/Dolphin3.0-R1-Mistral-24B

**Lessons**
- Reasoning behavior can be added through post-training without a novel base architecture.
- Large reasoning-trace datasets can strongly reshape policy.
- Alignment/refusal behavior is separable from raw reasoning architecture.
- GQA and quantization make medium models much more practical.

---

## Efficient architecture and adaptive compute

### Step 3.5 Flash
**Status:** OPEN-WEIGHT / substantial public implementation  
**Source:** https://github.com/stepfun-ai/Step-3.5-Flash

**Lessons**
- Sparse MoE separates total parameter capacity from active parameters per token.
- Multi-token prediction attacks autoregressive serial latency.
- Predicting multiple futures can pressure hidden states to encode short-horizon trajectory information.
- Sparse activation and multi-token prediction are complementary: one reduces compute per pass, the other reduces passes per output.

### LongCat-Flash / LongCat-Flash-Thinking
**Status:** OPEN-WEIGHT / PARTIAL  
**Source:** https://github.com/meituan-longcat/LongCat-Flash-Thinking

**Lessons**
- Neural computation can be dynamically allocated by token/context.
- Easy tokens should be able to select effectively zero/low extra expert computation.
- Dynamic neural routing must be co-designed with load balancing and hardware scheduling.
- Adaptive computation exists below the level of visible reasoning-token count.

### Byte Latent Transformer (BLT)
**Status:** SOURCE-AVAILABLE / non-commercial licensing caveat  
**Source:** https://github.com/facebookresearch/blt

**Lessons**
- Tokenizers are not sacred.
- Raw-byte models avoid fixed vocabulary fragmentation.
- Representation granularity can change dynamically with predictability/entropy.
- High-information regions deserve finer resolution and more computation.

### Mamba / Mamba-2 / Mamba-3
**Status:** OPEN  
**Source:** https://github.com/state-spaces/mamba

**Lessons**
- Long sequence memory need not rely entirely on full attention and ever-growing KV state.
- Recurrent state-space processing can offer linear-time sequence handling.
- Hardware-aware formulation is part of architecture design, not an implementation afterthought.
- Recurrence and attention can be combined.

### RecurrentGemma / Griffin
**Status:** OPEN CODE + OPEN WEIGHTS  
**Source:** https://github.com/google-deepmind/recurrentgemma

**Lessons**
- Local attention and recurrent long-horizon state can coexist.
- Different memory timescales may deserve different mechanisms.

---

## Native multimodality and action

### Qwen3-Omni
**Status:** OPEN / PUBLIC CODE + MODELS  
**Source:** https://github.com/QwenLM/Qwen3-Omni

**Lessons**
- Semantic thinking and speech realization can be separate networks/subsystems.
- Audio, video, images and text should not be forced through a text-only bottleneck.
- Different output modalities can share a common semantic core while using native decoders.

### GUI-Actor
**Status:** OPEN  
**Source:** https://github.com/microsoft/GUI-Actor

**Lessons**
- Spatial actions should not have to be serialized into text coordinates.
- Dedicated action heads can map directly from visual representation to action regions.
- Candidate generation and grounding verification can be separated.
- Small specialized heads can add major capabilities without retraining the entire backbone.

### ShowUI
**Status:** OPEN  
**Source:** https://github.com/showlab/ShowUI

**Lessons**
- Visual compute should be concentrated on task-relevant GUI regions/tokens.
- Small specialized perception-action models can be more efficient than a giant general model for every interaction.

---

## Robotics and embodied action

### openpi: pi0 / pi0-FAST / pi0.5
**Status:** SUBSTANTIALLY OPEN / PARTIAL PAPER REPRODUCTION  
**Source:** https://github.com/Physical-Intelligence/openpi

**Lessons**
- Robot actions should be represented in native motor/action spaces.
- Continuous action generation can use flow matching.
- Specialized action tokenizers are an alternative for autoregressive action generation.
- Semantic knowledge should guide low-level control.
- Long-horizon control benefits from goal -> semantic subtask -> motor trajectory hierarchy.
- Semantic knowledge and fast control may need architectural insulation to reduce destructive interference.

### SmolVLA / LeRobot
**Status:** OPEN  
**Source:** https://github.com/huggingface/lerobot

**Lessons**
- Useful general robot policies do not necessarily require enormous models.
- Action chunking reduces control overhead.
- Small models can be adapted with modest demonstrations.

### RDT-1B
**Status:** OPEN  
**Source:** https://github.com/thu-ml/RoboticsDiffusionTransformer

**Lessons**
- Diffusion can generate full action chunks/trajectories instead of one motor command at a time.
- Multi-robot pretraining can create transferable action representations.

### Octo
**Status:** OPEN  
**Source:** https://github.com/octo-models/octo

**Lessons**
- Heterogeneous robot datasets can produce generalist policies.
- Modularity improves transfer across cameras, embodiments, sensors and action spaces.

---

## Memory and lifelong agents

### Letta / MemGPT lineage
**Status:** OPEN FRAMEWORK  
**Source:** https://github.com/letta-ai/letta

**Lessons**
- Agent state should live independently of the current context window.
- Memory should be actively edited and organized by the agent.
- Persistent agents require state continuity across sessions.
- Memory consolidation can be treated as a separate cognitive process.

### Mem0
**Status:** OPEN FRAMEWORK  
**Source:** https://github.com/mem0ai/mem0

**Lessons**
- Memory storage should be separate from foundation-model weights.
- Extract and rank useful facts/events instead of replaying complete histories.
- Retrieval quality and relevance control are first-class problems.
- Memory can reduce context cost without continual pretraining.

### Voyager
**Status:** OPEN AGENT FRAMEWORK, CLOSED ORIGINAL FOUNDATION MODEL  
**Source:** https://github.com/MineDojo/Voyager

**Lessons**
- Expensive reasoning can be compiled into executable procedural skills.
- Procedural capability can accumulate without updating the base model.
- Automatic curricula can drive exploration toward progressively harder tasks.
- Execution errors are valuable learning signals.
- Skills should be compositional and reusable.

---

## Self-generated experience and curriculum

### Absolute Zero Reasoner (AZR)
**Status:** OPEN  
**Source:** https://github.com/LeapLabTHU/Absolute-Zero-Reasoner

**Lessons**
- The learner can generate its own training problems.
- The same model can alternate between proposer and solver roles.
- Curriculum should target learning progress/learnability rather than random difficulty.
- Self-generated experience is only trustworthy when anchored by external verification.
- Training distribution can become endogenous instead of permanently human-authored.

### SIMA 2
**Status:** CLOSED / LESSON ONLY

**Lessons**
- Agents can generate training experience by inhabiting interactive environments.
- Self-play can extend beyond formal board games.
- Experience from one generation can become data for the next.
- General action policies can span multiple 3D environments.

### Genie 3
**Status:** CLOSED / LESSON ONLY

**Lessons**
- Training environments themselves can be generated.
- World models can become experience factories.
- Scaling can shift from collecting more static internet data toward generating more grounded interaction.

---

## Verification and credit assignment

### DeepSeek-Prover-V2
**Status:** OPEN-WEIGHT / PUBLIC BENCHMARKS  
**Source:** https://github.com/deepseek-ai/DeepSeek-Prover-V2

**Lessons**
- Formal systems provide unusually strong reward signals.
- Large proofs should be decomposed into verifiable subgoals.
- Search + formal verification is stronger than linguistic self-confidence.

### Skywork Reward
**Status:** OPEN MODELS / PARTIAL PIPELINE  
**Source:** https://github.com/SkyworkAI/Skywork-Reward

**Lessons**
- Reward-model quality can depend more on curation than raw dataset size.
- Verifiers can be specialized by domain.
- Learned reward models are useful, but they are not ground truth.

### EVPV-PRM
**Status:** OPEN  
**Source:** https://github.com/Qwen-Applications/EVPV-PRM

**Lessons**
- Intermediate reasoning steps can be scored rather than only final answers.
- Multimodal reasoning should verify that a claimed premise is actually supported by the image/input.
- Process reward reduces the temporal distance between an error and its learning signal.

---

## Agent interfaces and tools

### SWE-agent / mini-SWE-agent
**Status:** OPEN FRAMEWORK  
**Source:** https://github.com/SWE-agent/SWE-agent

**Lessons**
- The agent-computer interface is part of intelligence.
- Structured tools can make the same model substantially more capable.
- Agent controllers can remain small if the environment interface is designed well.

### Gorilla / BFCL
**Status:** OPEN  
**Source:** https://github.com/ShishirPatil/gorilla

**Lessons**
- Tool selection is a learned capability.
- Huge tool libraries require retrieval before invocation.
- Semantic tool choice and syntactic call correctness are distinct requirements.
- Robust agents need recovery behavior when tool calls fail.

### Microsoft UFO / UFO3
**Status:** OPEN FRAMEWORK  
**Source:** https://github.com/microsoft/UFO

**Lessons**
- Complex goals can be represented as dependency DAGs rather than serial lists.
- Independent work should execute in parallel.
- Prefer structured interfaces in the order API > accessibility/UI tree > vision > raw coordinates.

---

## Inference and serving

### vLLM
**Status:** OPEN FRAMEWORK  
**Source:** https://github.com/vllm-project/vllm

### SGLang
**Status:** OPEN FRAMEWORK  
**Source:** https://github.com/sgl-project/sglang

**Lessons from modern serving stacks**
- Continuous batching.
- Paged KV-cache management.
- Prefix caching.
- Speculative decoding.
- Disaggregated prefill/decode where beneficial.
- Expert parallelism for MoE.
- Overlap communication with computation.
- Fused kernels.
- CUDA graphs.
- Low-precision/quantization-aware serving.

These techniques do not make the model intrinsically smarter, but they radically change the amount of useful cognition that can be purchased per unit of latency and hardware.

---

## Population search and automated discovery

### AlphaEvolve
**Status:** CLOSED CORE / public results and evaluators only  
**Public results:** https://github.com/google-deepmind/alphaevolve_results

**Lessons**
- Maintain populations of candidate solutions instead of committing all compute to one chain.
- Cheap models can explore breadth while expensive models improve selected branches.
- Execute and score proposals externally.
- Generator and evaluator should be separate roles.
- Evolutionary accumulation can discover improvements that one model call does not.

---

## Source-boundary rule

The repository uses these systems as evidence for **local design primitives**, not proof that the full Leviathan architecture exists today. When a lesson is a combination of ideas from several systems rather than a direct claim from one project, it is labeled **SYNTHESIS** in the architecture documents.

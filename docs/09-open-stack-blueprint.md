# Open-Stack Blueprint

This file maps Leviathan's conceptual modules to public projects that can be used as
offline donors, research references or isolated experiments. It is **not** a runtime
wiring diagram. Connecting these projects would create a collection of models, not the
one Leviathan model. A mechanism reaches runtime only after it is implemented or
distilled inside one substrate and one checkpoint. Deterministic stores, tools,
executors and measurement systems may remain external under frozen contracts.

## 1. Foundation reasoner

Candidate references:

- DeepSeek-R1 / distilled variants — https://github.com/deepseek-ai/DeepSeek-R1
- Qwen3 — https://github.com/QwenLM/Qwen3
- K2/IFM open releases — https://github.com/ifm-ai
- Step 3.5 Flash — https://github.com/stepfun-ai/Step-3.5-Flash
- LongCat — https://github.com/meituan-longcat/LongCat-Flash-Thinking
- Kimi K3 public/open-weight artifacts — https://github.com/MoonshotAI/Kimi-K3

Desired properties:
- strong reasoning,
- tool calling,
- controllable reasoning effort,
- long context,
- good quantized inference,
- ideally sparse/dynamic compute.

## 2. RL and post-training

Candidate:

- veRL — https://github.com/verl-project/verl

Responsibilities:
- rollout orchestration,
- policy updates,
- process/outcome rewards,
- verifiable reward integration,
- distributed execution,
- multi-turn agent training.

## 3. World model

Candidates:

- NVIDIA Cosmos — https://github.com/NVIDIA/Cosmos
- Cosmos Predict — https://github.com/nvidia-cosmos/cosmos-predict2.5
- V-JEPA 2 — https://github.com/facebookresearch/vjepa2
- JEPA-WM — https://github.com/facebookresearch/jepa-wms

Research split:

- use latent predictors where exact pixels are unnecessary,
- use generative world models when visual/physical detail matters,
- maintain a separate symbolic/semantic belief state regardless.

During research, an external world model is a simulator tool whose outputs remain
labelled hypothetical evidence. It cannot join the deployed cognitive path as a second
stateful model. A promoted world-model mechanism must become a head or state transition
inside the one Leviathan checkpoint.

## 4. Persistent memory

Candidates:

- Letta — https://github.com/letta-ai/letta
- Mem0 — https://github.com/mem0ai/mem0

Required extensions for Leviathan:
- explicit provenance,
- contradiction tracking,
- epistemic status,
- causal links,
- versioned semantic memories,
- promotion/demotion rules,
- skill registry.

## 5. Procedural memory

Reference:

- Voyager — https://github.com/MineDojo/Voyager

Leviathan skill object should support:

- preconditions,
- typed inputs/outputs,
- executable procedure,
- expected verifier,
- failure recovery,
- success statistics,
- provenance,
- version and environment assumptions.

## 6. Self-generated curriculum

Reference:

- Absolute Zero Reasoner — https://github.com/LeapLabTHU/Absolute-Zero-Reasoner

Leviathan extension:

`curriculum_value = learning_progress * usefulness * transfer * verifier_quality / (cost + risk)`

The proposer should not optimize novelty alone.

## 7. Formal verification

Reference:

- DeepSeek-Prover-V2 — https://github.com/deepseek-ai/DeepSeek-Prover-V2

Use when claims can be mapped into a theorem prover. Formal verification should receive high trust because the verifier is structurally independent of linguistic persuasion.

## 8. Learned verifiers and PRMs

References:

- Skywork Reward — https://github.com/SkyworkAI/Skywork-Reward
- EVPV-PRM — https://github.com/Qwen-Applications/EVPV-PRM

Rules:
- learned verifiers require calibration,
- keep them independent from the generator where possible,
- never assign them the same trust as formal/grounded verification.

## 9. GUI action

Candidates:

- GUI-Actor — https://github.com/microsoft/GUI-Actor
- ShowUI — https://github.com/showlab/ShowUI

Preferred hierarchy:

`API > structured UI/accessibility tree > native visual action head > raw coordinate generation`

## 10. Tool/API orchestration

References:

- Gorilla/BFCL — https://github.com/ShishirPatil/gorilla
- SWE-agent — https://github.com/SWE-agent/SWE-agent
- Microsoft UFO — https://github.com/microsoft/UFO

Leviathan requirements:
- tool retrieval before invocation,
- structured schemas,
- error recovery,
- DAG planning,
- parallel execution,
- deterministic filtering outside the LLM.

## 11. Robot action

Candidates:

- openpi — https://github.com/Physical-Intelligence/openpi
- LeRobot/SmolVLA — https://github.com/huggingface/lerobot
- RDT-1B — https://github.com/thu-ml/RoboticsDiffusionTransformer
- Octo — https://github.com/octo-models/octo

Possible action mechanisms:
- continuous flow-matching action head,
- action tokenizer/autoregressive chunks,
- diffusion trajectory generation.

## 12. Multimodal speech/output

Reference:

- Qwen3-Omni — https://github.com/QwenLM/Qwen3-Omni

Lesson:
- share semantic cognition while delegating speech/acoustic realization to a dedicated output subsystem.

## 13. Efficient representation

Reference:

- BLT — https://github.com/facebookresearch/blt

Experiment:
- compare normal tokenization versus entropy-controlled byte patches on code, rare strings, multilingual text and exact symbolic tasks.

## 14. Recurrent/long-state architecture

References:

- Mamba — https://github.com/state-spaces/mamba
- RecurrentGemma — https://github.com/google-deepmind/recurrentgemma

Research question:
- what should live in local attention, recurrent neural state, compressed belief state and external memory respectively?

## 15. Efficient serving

Candidates:

- vLLM — https://github.com/vllm-project/vllm
- SGLang — https://github.com/sgl-project/sglang

Required capabilities:
- continuous batching,
- paged KV,
- prefix caching,
- speculative decoding,
- MoE expert parallelism,
- quantized kernels,
- asynchronous tool/service integration.

## 16. Population search

AlphaEvolve itself is closed. Public results:

- https://github.com/google-deepmind/alphaevolve_results

Open implementation strategy for Leviathan:

```text
candidate archive
  -> cheap model mutations/proposals
  -> optional strong-model refinement
  -> external execution/evaluation
  -> novelty + fitness ranking
  -> archive survivors
  -> repeat
```

The evaluator must be task-specific and external whenever possible.

## 17. Generated environments

SIMA 2 and Genie 3 are lesson-only closed systems in this inventory.

Open Leviathan substitutes can begin with:

- software sandboxes,
- games,
- browser environments,
- code execution,
- robotics simulators,
- procedurally generated formal tasks.

The key research requirement is provenance: simulated success must never silently become real-world truth.

## 18. Minimal first implementation

A practical first research stack should be much smaller than the full architecture and
must still have only one cognitive checkpoint:

```text
one open reasoner checkpoint
+ vLLM/SGLang
+ structured belief state
+ Letta/Mem0-style memory
+ tool registry
+ code execution sandbox
+ verifier interface
+ procedural skill store
+ meta-controller
```

The belief store, memory index, registry, sandbox and verifier interface in this list
are typed infrastructure. They do not infer competing answers. The metacognitive policy
must end as a head of the same reasoner; a temporary deterministic baseline is allowed
only as an auditable governance scaffold.

Do **not** begin with continual core-weight learning. First prove that the system can:

1. maintain stable beliefs,
2. choose useful cognitive modes,
3. compile/reuse skills,
4. verify outcomes,
5. improve task efficiency across repeated experience.

Only after these are measurable should plastic parameter updates be introduced.

## 19. Second implementation layer

Add:

- world-model simulation,
- multi-hypothesis search,
- population search for verifiable optimization tasks,
- self-generated curriculum in code/formal domains,
- process-reward learning,
- uncertainty calibration.

## 20. Third implementation layer

Add carefully:

- plastic adapters/experts,
- transactional continual updates,
- replay and stability constraints,
- shadow deployment,
- cross-domain curriculum,
- physical/GUI native action modules.

## 21. What should remain external

Even in an advanced research system, keep these outside the learner's unilateral control:

- root permissions,
- deployment signing,
- rollback authority,
- safety/regression suites,
- final verifier policy,
- production promotion rules.

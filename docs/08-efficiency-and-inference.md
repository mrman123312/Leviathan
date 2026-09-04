# Efficiency and Inference

Leviathan separates **five kinds of efficiency** because "tokens per second" and "token efficiency" are often conflated.

## 1. Representation efficiency

Question: how much task-relevant information fits into one model step?

Mechanisms:
- better tokenizer design,
- byte-level adaptive patches (BLT),
- selective visual tokens (ShowUI),
- compressed latent representations.

Goal:

`useful_information / representation_unit -> up`

## 2. Model efficiency

Question: how much neural compute is required per processed unit?

Mechanisms:
- sparse MoE,
- dynamic expert activation,
- latent MoE,
- recurrence/state-space layers,
- local/sliding attention,
- MLA/GQA-style KV reduction,
- low precision.

Goal:

`active_parameters / total_capacity -> down`

without materially reducing quality.

## 3. Reasoning efficiency

Question: how much deliberation is needed to reach a correct decision?

Mechanisms:
- RL for direct successful trajectories,
- adaptive reasoning budgets,
- skill reuse,
- stopping policies,
- information-seeking experiments,
- search-geometry selection,
- distillation of expensive reasoning into cheaper policy.

Goal:

`successful_task / reasoning_compute -> up`

## 4. Context and agent efficiency

Question: how often is the model forced to reread, regenerate or transport irrelevant state?

Mechanisms:
- prefix/prompt caching,
- context compaction,
- external memory,
- persistent structured state,
- tool schema retrieval,
- programmatic tool calling,
- deterministic filtering outside the LLM,
- procedural skills.

Goal:

`repeated_context_work -> down`

## 5. Serving efficiency

Question: how much useful inference can the hardware execute per second?

Mechanisms:
- continuous batching,
- paged KV cache,
- speculative decoding,
- multi-token prediction,
- fused kernels,
- FlashAttention/FlashInfer-style kernels,
- CUDA graphs,
- expert parallelism,
- prefill/decode disaggregation,
- communication/compute overlap,
- FP8/FP4/INT quantization.

## 6. Tokens-per-second metrics

Never report `tok/s` without a denominator/context.

Distinguish:

### Single-stream decode
How fast one user's output advances.

### Aggregate per-GPU throughput
Total tokens across all simultaneously served requests.

### Node/cluster throughput
Total output across many accelerators.

### Effective speculative output rate
Visible accepted tokens per target-model verification cycle.

A claim such as `2,000 tok/s` can mean radically different things depending on which metric is used.

## 7. Why batching helps

Single-stream autoregressive decode often underutilizes accelerator arithmetic because each step resembles a large matrix operating on a small activation vector and is frequently memory-bandwidth limited.

Batching multiple sequences increases arithmetic intensity and reuses the same model weights across many concurrent tokens.

Therefore:

`aggregate tok/s` can rise dramatically without each individual stream reaching the same speed.

## 8. Quantization

For `P` parameters:

- BF16 storage ~= `2P` bytes,
- FP8 ~= `P` bytes,
- 4-bit ~= `0.5P` bytes.

During bandwidth-bound decoding, fewer bytes transferred per token can increase throughput significantly.

Leviathan favors quantization-aware training when extreme low precision is a target architecture property rather than an after-the-fact deployment requirement.

## 9. Speculative decoding

Classic decode:

`target pass -> 1 token`

Speculative decode:

`cheap draft -> k candidate tokens -> target verifies candidates in parallel -> accept valid prefix`

Effective speed depends on:

- draft cost,
- acceptance length,
- target verification cost,
- synchronization overhead.

A poorly matched drafter can make inference slower.

## 10. Multi-token prediction

Multi-token prediction trains hidden state to propose several future tokens/positions from one representation.

Leviathan's broader extrapolation is to use this principle for action trajectories:

`state_t -> candidate actions[t:t+k] + predicted world states[t+1:t+k]`

This should be combined with receding-horizon verification rather than blindly executing the entire prediction.

## 11. KV and recurrent state

Attention-based models store K/V state for prior tokens. KV-reduction approaches and recurrent state-space models reduce the memory burden in different ways.

Leviathan does not assume one memory mechanism is globally optimal. A hybrid system can use:

- local attention for immediate detail,
- compressed latent/KV state for medium horizon,
- recurrent state for long-running summaries,
- external memory for episodic history.

## 12. Fused kernels and memory movement

Inference can be dominated by memory movement rather than mathematical FLOPs.

Fusing operations reduces cycles like:

`HBM -> compute -> HBM -> compute -> HBM`

by keeping intermediates in registers/on-chip memory longer.

This is why a model's architecture and its kernel implementation cannot be treated as independent performance questions.

## 13. Programmatic tool execution

Agent token efficiency often improves more from moving deterministic intermediate work outside the model than from reducing model size.

Bad pattern:

`LLM -> tool -> 20k token result -> LLM -> tool -> 20k token result -> LLM`

Better pattern:

`LLM writes controller program -> program calls/filter tools -> compact result -> LLM`

The model sees only information requiring semantic judgment.

## 14. Cognitive amortization

The strongest lifetime efficiency mechanism is learning.

First encounter:
- search,
- tool use,
- large reasoner,
- many tokens.

Repeated encounter:
- compiled procedural skill,
- retrieval,
- smaller model,
- fewer actions.

Eventually:
- parametric intuition/direct policy.

Thus experience should drive:

`compute_required_for_known_skill -> down`

## 15. Multiplicative efficiency

Total task cost can be approximated as a product of several factors:

`C_task ~ representation_units * active_compute * forward_passes * memory_traffic * system_overhead`

Different innovations attack different terms:

- BLT -> representation units,
- sparse/dynamic MoE -> active compute,
- MTP/speculation -> forward passes,
- GQA/MLA/Mamba/KV engineering -> memory traffic,
- vLLM/SGLang/tool orchestration -> system overhead,
- agentic RL/skills -> unnecessary cognitive steps.

Because these factors multiply, several moderate improvements can yield a large end-to-end gain.

## 16. Hardware-aware cognition

A future meta-controller should include device conditions in routing:

`route_score = expected_quality - queue_cost - communication_cost - memory_cost - latency_cost`

The theoretically strongest expert/model is not always the fastest or most efficient executor at that moment.

## 17. Benchmarking requirements

For every Leviathan configuration record:

- time to first token,
- inter-token latency,
- single-stream tok/s,
- aggregate tok/s,
- GPU utilization,
- HBM bandwidth utilization,
- KV-cache footprint,
- active parameter count,
- speculative acceptance length/rate,
- tool calls/task,
- reasoning tokens/task,
- wall-clock task time,
- success rate,
- cost per successful task.

The final metric should always include **task success**, because a fast model that fails is not efficient.

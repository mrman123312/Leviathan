# The 157-Lesson Library

This is the complete numbered lesson inventory distilled from the full research discussion. These are not all direct claims from one paper; some are explicit **SYNTHESIS** lessons created by comparing multiple systems. See `00-source-ledger.md` for source boundaries.

## World models, reasoning and training openness

1. Predict representations, not necessarily pixels.
2. A world model should support counterfactual action prediction.
3. World understanding and world generation can share representations.
4. Physical reasoning benefits from persistent latent state rather than only linguistic descriptions.
5. Reasoning algorithms themselves can emerge from reinforcement learning.
6. Pure outcome-driven RL can create strong reasoning while also producing policy pathologies.
7. Expensive search can eventually become learned intuition.
8. One model can learn multiple compute/reasoning policies.
9. Reasoning does not have to be permanently enabled.
10. Full training histories are themselves research artifacts.
11. The same model family can span edge to frontier scales.
12. Architecture, data and training should be studied together rather than inferring everything from final weights.
13. Small and giant models can share a training methodology while using different compute arrangements.

## Kimi-style sparse multimodal agents and closed-frontier systems lessons

14. Massive total capacity does not require massive active capacity.
15. Latent MoE can reduce expert computation and communication.
16. Full attention does not need to exist at every layer.
17. Depth itself can be selectively accessed.
18. Quantization should sometimes be trained in rather than applied afterward.
19. Agent ability must be trained on trajectories, not just final responses.
20. Preserve reasoning/task state across tool interactions.
21. Vision-in-the-loop creates self-correcting artifact generation.
22. Reasoning effort should adapt to difficulty and uncertainty.
23. Task-wide budgets and per-step reasoning budgets should be separate controls.
24. Prompt caching can make long agent loops economically viable.
25. Stable context should not be recomputed every turn.
26. Tool definitions should be retrieved dynamically rather than dumped into every prompt.
27. Programmatic tool calling is often superior to LLM -> tool -> LLM -> tool loops.
28. Intermediate deterministic data should be processed by code rather than repeatedly consumed by an LLM.
29. Train for work-per-token rather than merely next-token quality.
30. Reward shorter successful trajectories.
31. Use programs to orchestrate several tools without returning through the expensive model after every operation.
32. Separate stable context from new context using cache-aware serving.
33. Preserve reasoning/task state across actions.
34. Long-lived agents need persistent notes plus searchable old context.
35. Increased reasoning compute should sometimes mean more experiments, not a longer monologue.
36. Agents should select some actions for information gain.
37. External environment feedback is often a better verifier than self-judgment.
38. Asynchronous tools allow cognition to become a parallel computational process.
39. Compute allocation can change midway through a task.

## RL infrastructure, memory and self-generated training

40. RL infrastructure should separate rollout generation, reward computation, policy updates, distributed execution and serving.
41. The learner should not be tightly coupled to one reward mechanism.
42. Agent state should live independently from the context window.
43. The agent should be able to modify its own memory.
44. Memory should be an active cognitive operation, not passive vector retrieval.
45. Persistent agents need identity/state continuity across sessions.
46. Separate memory storage from foundation-model weights.
47. Extract useful facts/events instead of replaying whole histories.
48. Retrieval itself needs ranking and relevance control.
49. Memory can reduce context cost without requiring continual pretraining.
50. The model can generate its own training tasks.
51. Self-generated tasks should target learnability, not random difficulty.
52. The same model can alternate between proposer and solver roles.
53. Self-generated experience requires grounded verification.
54. Training distribution can become endogenous rather than fixed by humans.

## Verification and process reward

55. Formal verification can provide nearly ideal reward signals in restricted domains.
56. Large proofs should be broken into verifiable subgoals.
57. Search plus formal verification is stronger than linguistic self-confidence.
58. Reward-model quality may depend more on data quality than raw dataset size.
59. Verifier/reward models can be specialized for reasoning, code, safety, conversation and other domains.
60. Learned verifiers are useful but must not be mistaken for ground truth.
61. Verify intermediate reasoning steps, not only final answers.
62. Multimodal reasoning should check that a claimed premise is actually supported by the image/input.
63. Process reward can attach credit closer to where an error happened.

## Physical action and robotics

64. Robot action should be represented natively.
65. Continuous action generation can use flow matching.
66. Actions can alternatively be tokenized with specialized action tokenizers.
67. Abstract semantic knowledge should guide low-level physical control.
68. Hierarchical control should map goal -> subtask -> motor trajectory.
69. Semantic knowledge and motor control may need architectural insulation to reduce destructive interference.
70. Useful robot foundation policies do not necessarily need enormous parameter counts.
71. Small general robot models can be adapted with relatively modest demonstrations.
72. Action chunking reduces control overhead.
73. Diffusion models can generate entire action trajectories.
74. Cross-robot pretraining can create transferable manipulation representations.
75. Action-sequence prediction can replace single-action autoregression.
76. Heterogeneous robot datasets can produce generalist policies.
77. Modularity helps transfer across cameras, robot morphologies, action spaces and sensors.
78. Language and goal images can share a general action-conditioning architecture.

## Post-training, efficient reasoning and model architecture

79. Reasoning behavior can be added through post-training without inventing a new base architecture.
80. Large reasoning-trace datasets can substantially reshape model policy.
81. Refusal/uncensored behavior is mainly an alignment and post-training choice, not a magical reasoning architecture.
82. Grouped-query attention reduces KV-cache cost.
83. Medium-sized models become much more practical under aggressive quantization.
84. Separate total intelligence capacity from active compute.
85. Multi-token prediction attacks autoregressive serial latency.
86. Predicting farther into the future may improve internal trajectory representations.
87. Sparse MoE and multi-token prediction are complementary: one reduces compute/pass, the other reduces passes/output.
88. Neural computation itself can be dynamically allocated.
89. Easy tokens should be able to select effectively less computation.
90. Compute routing requires dynamic hardware load balancing.
91. Adaptive computation is useful below the level of visible reasoning-token count.
92. Reward the complete agent trajectory.
93. Process rewards improve long-horizon credit assignment.
94. Wall-clock time should be part of the definition of agent quality.
95. Parallel tool use can reduce agent latency.
96. Correct eventually is not enough; useful work per time/compute/action matters.

## Tokenization, recurrence and multimodal output

97. Tokenizers are not sacred.
98. Raw bytes avoid fixed-vocabulary fragmentation problems.
99. Representation granularity should vary with predictability.
100. High-entropy input deserves finer computation.
101. Sequence memory does not necessarily require full attention over the entire past.
102. Recurrent state can enable linear-time sequence processing.
103. Hardware-aware architecture design matters as much as asymptotic theory.
104. Attention and recurrence need not be mutually exclusive.
105. Local attention and recurrent long-term state can coexist.
106. Different timescales of memory may deserve different mechanisms.
107. Thinking and speaking should be decoupled.
108. Semantic computation and acoustic realization have different time/compute requirements.
109. Native multimodal representations preserve information lost through text bottlenecks.
110. Multiple output streams can share a common semantic core.

## GUI and native action interfaces

111. Stop encoding inherently spatial actions as text coordinates.
112. Use dedicated action heads.
113. Generate several candidate targets and verify them.
114. Small specialized heads can add major capabilities without retraining the entire foundation model.
115. Do not spend equal visual compute on every part of a screenshot.
116. Select GUI-relevant visual tokens.
117. Specialized lightweight perception-action models can outperform giant general VLMs for narrow interaction loops.

## Procedural learning and agent interfaces

118. Expensive reasoning can be compiled into executable skills.
119. Procedural memory can grow without updating foundation-model weights.
120. Automatic curricula can make an agent progressively explore harder tasks.
121. Execution errors are valuable learning signals.
122. Skills should be compositional.
123. Procedural memory can reduce the pressure for constant weight updates and catastrophic forgetting.
124. The agent-computer interface is part of intelligence.
125. Better interfaces can produce capability gains without changing model weights.
126. Agent frameworks do not have to be gigantic.
127. Tool selection is itself a learned capability.
128. Large tool libraries require retrieval before execution.
129. Syntactic correctness matters as much as semantic tool choice.
130. Tool-use agents need recovery behavior when calls fail.
131. Complex goals can be represented as dependency graphs rather than serial lists.
132. Parallel cognition can dramatically reduce wall-clock latency.
133. For computer use, prefer structured access over vision when possible: API > accessibility/UI tree > vision > raw coordinates.

## Serving and systems efficiency

134. Continuous batching.
135. Paged KV caches.
136. Prefix caching.
137. Speculative decoding.
138. Disaggregated prefill/decode where useful.
139. Expert parallelism for MoE.
140. Overlap networking with computation.
141. Fused kernels.
142. CUDA graphs.
143. Quantization-aware serving.

## Population search and evolutionary cognition

144. Maintain populations of candidate solutions.
145. Do not commit the entire reasoning budget to one chain.
146. Let cheap models explore broadly.
147. Let expensive models improve selected candidates.
148. Execute and measure candidates externally.
149. Separate creativity from truth: generator != evaluator.
150. Evolution can accumulate improvements that no single model call finds directly.

## Experience generation and generated worlds

151. Agents can generate useful training experience by living in environments.
152. Self-play need not be limited to formal board games.
153. Experience can be recycled into the next generation of agent.
154. General action policies can span different 3D environments.
155. Training environments themselves can be generative models.
156. World models can become experience factories.
157. Agent scaling can shift from more static internet data toward more generated, grounded interaction.

---

## Synthesis beyond the 157

The numbered inventory implies several larger ideas that should be treated as Leviathan synthesis rather than as claims of one source:

- **Persistent belief state:** maintain facts, hypotheses, causal links, uncertainty and provenance rather than only a transcript.
- **Metacognitive routing:** learn which cognitive algorithm to invoke.
- **Multi-timescale continual learning:** working state -> episodic memory -> semantic abstraction -> procedural skill -> plastic parameters -> slow core consolidation.
- **Grounded prediction-error learning:** compare predicted and actual environment outcomes before promoting experience into learning.
- **Cognitive compilation:** repeated expensive reasoning should become reusable procedures and eventually cheaper parametric intuition.
- **Trust-weighted learning:** the strength of a learning update should depend on provenance, verifier quality, novelty, consistency and utility.
- **Hardware-aware cognition:** routing decisions should consider communication, memory bandwidth, queueing and device load as well as model quality.

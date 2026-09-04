# Master Principles

The 157 observations collapse into a smaller set of architecture rules. These are the principles Leviathan treats as reusable across models, environments and hardware.

## 1. Adaptive representation

**Sources:** BLT, ShowUI, multimodal encoders.  
**Rule:** Spend representation resolution where information demands it.

A fixed tokenizer or fixed visual patch allocation assumes every part of the input deserves the same granularity. Leviathan instead treats representation resolution as a controllable resource:

`resolution = f(entropy, novelty, task relevance, risk)`

Predictable text can be compressed aggressively. Rare symbols, code, measurements, uncertain visual regions and safety-critical details should receive finer representation.

## 2. Adaptive neural capacity

**Sources:** LongCat, Kimi K3, Step 3.5 Flash.  
**Rule:** Do not activate the whole brain for every computational unit.

`active_compute_t = f(context, uncertainty, difficulty, load)`

The one model should route easy work through fewer internal parameter bases or forward
passes and difficult work through greater internal capacity. It must not choose among a
portfolio of independently stateful models at runtime. Routing must account for
hardware congestion, not only model-side quality.

## 3. Predict beyond one step

**Sources:** Step MTP, Cosmos, RDT, world-model planning.  
**Rule:** Hidden state should encode likely trajectories, not only the immediate next symbol.

Leviathan should predict short horizons of actions and environment states:

`belief_t -> candidate actions[t:t+k] + predicted states[t+1:t+k]`

Execution should still be receding-horizon: predict several steps, verify, execute only the trustworthy prefix, observe, replan.

## 4. Maintain persistent belief state

**Sources:** Letta, Astra-style persistent state, world models.  
**Rule:** Context is not memory and chat history is not a world model.

Belief state should explicitly track:

- current facts,
- hypotheses,
- uncertainty,
- causal relationships,
- provenance,
- goals,
- constraints,
- unresolved contradictions,
- active plans,
- prior failed attempts.

## 5. Memory is plural

**Sources:** Letta, Mem0, Voyager.  
**Rule:** Use separate memory systems for separate functions.

- **working memory** — immediate task state,
- **episodic memory** — what happened and when,
- **semantic memory** — generalized facts and concepts,
- **procedural memory** — executable/reusable skills,
- **parametric memory** — knowledge consolidated into model parameters.

Learning rates should satisfy roughly:

`external memory >> plastic modules >> core weights`

## 6. Reasoning should be adaptive

**Sources:** Qwen thinking modes, Fable, Sol/Astra, LongCat.  
**Rule:** Spend deliberation according to expected value.

`reasoning_budget = f(difficulty, uncertainty, stakes, verifier availability, expected information gain)`

Harder thinking should sometimes mean more tests or experiments, not merely more hidden reasoning tokens.

## 7. Compile expensive cognition into cheap skills

**Sources:** Voyager, distillation, RL, procedural memory.  
**Rule:** Repeated successful deliberation should become reusable structure.

`novel problem -> expensive search -> verified solution -> skill -> abstraction -> intuition`

This is **cognitive amortization**: experience should increase competence while reducing compute required for familiar tasks.

## 8. Do not force everything through text

**Sources:** GUI-Actor, Qwen3-Omni, pi/openpi, RDT.  
**Rule:** Use native output spaces.

A central semantic state can feed specialized heads for:

- text,
- speech/audio codes,
- GUI regions/actions,
- structured API calls,
- robot trajectories,
- images/video.

Text is a communication interface, not necessarily the universal internal action representation.

## 9. Use hierarchical goals and actions

**Sources:** pi0.5, agent planners.  
**Rule:** Separate intent from execution.

`goal -> subgoals -> plan -> action chunks -> low-level control`

This allows abstract knowledge to guide action without forcing the same network to solve every motor/control detail.

## 10. Use deterministic computation when possible

**Sources:** programmatic tool use, SWE-agent, AlphaEvolve-style evaluators.  
**Rule:** Neural models should not emulate reliable deterministic systems.

Use code, databases, solvers, compilers, tests, parsers and formal systems for tasks they can perform exactly. Reserve expensive neural computation for ambiguity, planning, interpretation, abstraction and hypothesis generation.

## 11. Verification should be external whenever possible

**Sources:** DeepSeek-Prover, AZR, GUI-Actor verifier, EVPV-PRM, AlphaEvolve.  
**Rule:** The model should not be the sole judge of its own success.

Trust order:

`physical reality > formal proof > deterministic execution > trusted measurement > independent verifier > learned reward model > same-model self-critique`

Learning strength should decrease as the verifier moves down this hierarchy.

## 12. Reward trajectories, not only answers

**Sources:** MiniMax agentic RL, process reward.  
**Rule:** A correct result reached through wasteful or dangerous behavior is not equivalent to an efficient, robust result.

Conceptual objective:

`reward = success - time_cost - compute_cost - action_cost - error_cost - risk_cost`

Exact coefficients should be task-dependent and learned/calibrated rather than universal constants.

## 13. Generate experience

**Sources:** AZR, SIMA, Genie, world-model environments.  
**Rule:** Static human datasets should not remain the only growth mechanism.

`agent -> action -> environment -> observation -> verification -> learning`

Generated/simulated experience must retain provenance and must not be treated as equivalent to grounded reality.

## 14. Let the learner choose what to learn next

**Sources:** AZR, Voyager curriculum.  
**Rule:** Curriculum generation is itself an optimization problem.

A useful learning task maximizes something like:

`expected learning progress * expected usefulness * transfer * verifier reliability / (cost + risk)`

Pure novelty is not enough.

## 15. Maintain competing hypotheses

**Sources:** AlphaEvolve, candidate-grounding systems, search.  
**Rule:** Avoid early path lock-in.

When uncertainty is material, preserve multiple hypotheses and let evidence eliminate branches. Do not spend an entire reasoning budget repairing a bad first guess.

## 16. Compute should be heterogeneous

**Sources:** the entire stack.  
**Rule:** One general model should allocate heterogeneous internal compute and typed
external instruments without becoming a portfolio of minds.

Internal resources include parameter bases, attention heads, shared-state passes,
perceptual projections, world-state projections and native action heads. Search trees,
hypotheses and memory entries are data operated on by the same model. Formal solvers,
code execution, sensors, effectors and independent measurements may remain external
because they are instruments with frozen contracts, not separately goaled cognition.

The metacognitive policy that allocates these resources is a learned head of the same
checkpoint and shares the same task state.

## 17. Metacognition is the master control problem

**SYNTHESIS**

The core question becomes:

> **How should I think about this problem?**

The meta-controller chooses among:

`direct | recall | retrieve | reason | calculate | search | simulate | experiment | parallelize | evolve | invoke-skill | ask | act`

A conceptual objective is:

`mode* = argmax(expected_success_gain + information_gain - compute - latency - risk)`

## 18. Learning must be multi-timescale

**SYNTHESIS**

The safe progression is:

`working state -> episodic memory -> semantic abstraction -> procedural skill -> plastic parameters -> slow core consolidation`

Experiences should require progressively stronger evidence to move toward slower, more destructive forms of memory.

## 19. Confidence and provenance are data, not prose

**SYNTHESIS**

Every consequential belief should store not only a value but also:

`(value, confidence, provenance, timestamp, evidence, contradictions)`

This enables active experimentation, calibration, trust-weighted learning and later rollback.

## 20. The learner must not control its own constitution

**SYNTHESIS / governance rule**

A continually learning system should not have unilateral authority over:

- verifier definitions,
- deployment gates,
- rollback infrastructure,
- safety policies,
- model signing,
- root training infrastructure.

The learner and governor must be separable.

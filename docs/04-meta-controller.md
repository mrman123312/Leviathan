# Metacognitive Controller

The metacognitive controller is Leviathan's proposed master subsystem. It does not directly solve every task. It selects **which cognitive procedure should run**, how much resource it should receive, when it should stop, and what evidence would justify escalation.

## 1. Why a controller is necessary

The source systems we studied already expose different forms of adaptive cognition:

- BLT changes representation granularity.
- LongCat changes neural compute per token/context.
- Qwen/Fable/Astra-style systems change reasoning effort.
- Voyager chooses reusable skills instead of rediscovery.
- AlphaEvolve-like systems explore populations instead of one chain.
- MiniMax-style agentic RL values shorter/better trajectories.
- world models make simulation available.
- tools make deterministic computation available.

A general system must decide among all of these options.

## 2. Cognitive mode set

```text
DIRECT        high-confidence immediate response/action
RECALL        retrieve semantic/episodic memory
SKILL         run procedural memory
REASON        serial deliberation
TREE_SEARCH   branch over a modest candidate set
EVOLVE        population-based exploration and selection
SIMULATE      predict counterfactual outcomes
EXPERIMENT    act to reduce epistemic uncertainty
TOOL          delegate deterministic work
PARALLELIZE   build a DAG and run independent branches
ASK           obtain missing information externally
ACT           perform grounded environment action
WAIT/OBSERVE  when information will arrive without costly intervention
```

## 3. Decision state

The controller receives a compact state rather than the full raw history:

```text
MetaState
  task_type
  current_goal
  success_probability
  epistemic_uncertainty
  aleatoric_uncertainty
  stakes
  risk_budget
  compute_budget
  latency_budget
  available_verifiers
  available_tools
  candidate_skills
  world_model_confidence
  branching_factor_estimate
  expected_information_gain
  recent_failures
  hardware_load
```

## 4. Utility model

A conceptual score for cognitive mode `m`:

`U(m) = P_success_gain(m) + beta*information_gain(m) + tau*transfer_value(m) - lambda*compute(m) - mu*latency(m) - rho*risk(m) - kappa*irreversibility(m)`

Important points:

- `P_success_gain` is marginal improvement over the current policy, not raw accuracy.
- information gain matters when uncertainty is epistemic.
- transfer value rewards experiments/solutions likely to create reusable skills or knowledge.
- irreversibility penalizes actions whose consequences are difficult to undo.

## 5. Escalation policy

Leviathan should start with the cheapest plausible cognition mode and escalate when evidence justifies it.

Example ladder:

```text
DIRECT
  -> RECALL
  -> SKILL
  -> REASON(low)
  -> TOOL / VERIFY
  -> REASON(high)
  -> TREE_SEARCH
  -> SIMULATE
  -> EXPERIMENT
  -> EVOLVE / expensive search
```

This is not a rigid order. A formal proof problem may jump immediately to theorem-prover search. A familiar automation task may jump directly to `SKILL`.

## 6. Stop policy

Reasoning should stop when the expected value of another cognitive step becomes negative:

`continue if expected_quality_gain + information_gain > compute_cost + latency_cost + risk_cost`

This prevents both premature stopping and endless overthinking.

## 7. Information-seeking behavior

When multiple hypotheses survive, choose actions that discriminate among them.

For hypotheses `H1..Hn` and candidate action `a`:

`IG(a) = expected reduction in entropy over H after observing outcome(a)`

A high-value experiment should:

- split plausible hypotheses,
- have a trustworthy outcome signal,
- be low cost,
- be reversible when possible,
- transfer knowledge beyond the immediate case.

## 8. Search-geometry routing

The controller should estimate problem branching structure:

- low branching + strong internal model -> serial reasoning,
- moderate branching -> beam/tree search,
- high branching + cheap exact evaluator -> population search/evolution,
- causal uncertainty -> experiment,
- physical uncertainty -> world-model simulation + grounded action,
- known deterministic transformation -> tool/skill.

## 9. Hardware-aware routing

A theoretically best internal parameter route may be a bad wall-clock choice when its
gather, memory or synchronization cost is high.

Routing can include a systems term:

`score(route) = expected_quality - latency_penalty - queue_penalty - communication_penalty - memory_penalty`

This is the system-level extension of LongCat's lesson that adaptive routing and hardware load balancing cannot be designed independently.

## 10. One model, multiple typed operations

The controller chooses modes and resource allocations inside one cognitive model:

```text
fewer parameter bases       cheap direct inference
more parameter bases        difficult high-value inference
shared-state passes         bounded refinement
world-state projection      state/action prediction
perceptual projection       visual/audio grounding
native action head          robot/GUI action encoding
```

Formal solvers, deterministic tools, sensors, executors and independent measurements
remain external boundaries. They can transform or verify a frozen contract; they do not
become separately stateful cognitive models. Runtime cognition always resolves to one
foundation checkpoint and one shared task state.

## 11. Learning the controller

The controller itself can be trained from trajectories.

A trajectory record should include:

```text
initial meta-state
chosen cognitive mode
resource budget
steps taken
verification evidence
success/failure
latency
compute
risk incidents
information gained
whether a reusable skill resulted
```

Possible training objectives:

- imitation from strong orchestration traces,
- offline policy learning from historical trajectories,
- RL against task success + efficiency,
- contextual bandits for low-risk routing decisions,
- constrained RL for safety/risk limits.

## 12. Avoid reward hacking

Do not reward the controller solely for looking efficient. A model can appear efficient by:

- stopping early,
- avoiding difficult tasks,
- choosing easy-to-game verifiers,
- hiding uncertainty,
- preferring tasks that produce simple rewards.

Therefore trajectory reward must include:

- externally measured task success,
- calibration,
- safety/regression scores,
- verifier quality,
- completeness under the original goal.

## 13. Example

Task: "Find the root cause of a new production latency regression."

```text
1. RECALL
   retrieve prior incidents and deployment timeline

2. TOOL
   query metrics and compare release window

3. REASON
   form 4 plausible hypotheses

4. EXPERIMENT
   choose one profile/test that best separates the hypotheses

5. PARALLELIZE
   inspect database and network path concurrently

6. VERIFY
   benchmark candidate explanation against observed latency

7. SKILL-COMPILE
   if the diagnostic procedure is reusable, store it as a procedure
```

The intelligence is not only in each step. It is in selecting the sequence of cognitive modes.

## 14. Open research questions

- How should mode utility be calibrated across domains?
- How can the controller estimate information gain before running an experiment?
- How do we prevent it from systematically preferring verifiable but unimportant work?
- How does it recognize a truly novel problem where its routing policy is out of distribution?
- How should hardware cost and cognitive quality trade off under real-time load?
- How can controller updates be made without destabilizing safety behavior?

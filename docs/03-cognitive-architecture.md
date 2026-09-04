# Leviathan Cognitive Architecture

This document is the main systems design. It combines the source-backed primitives into a single hypothetical architecture while keeping source-derived mechanisms separate from Leviathan synthesis.

## 1. Three nested loops

### Fast loop: perception and action
Timescale: milliseconds to seconds.

`multimodal input -> adaptive representation -> active compute routing -> state update -> short-horizon prediction -> native action`

Responsibilities:
- encode text/vision/audio/sensor input,
- allocate representation granularity,
- allocate active model capacity,
- update the current belief state,
- predict immediate consequences,
- dispatch native actions.

### Deliberative loop: planning and search
Timescale: seconds to hours.

`belief state -> meta-controller -> reasoning/search/simulation -> plan -> action/tool -> verification -> belief revision`

Responsibilities:
- choose cognitive mode,
- maintain multiple hypotheses,
- decompose goals,
- run tools and experiments,
- use world models for counterfactuals,
- verify outcomes,
- replan under new evidence.

### Consolidation loop: learning
Timescale: minutes to weeks.

`trajectory -> event extraction -> verification -> causal credit -> novelty/utility scoring -> memory promotion -> optional parameter update`

Responsibilities:
- turn raw trajectories into structured episodes,
- decide what was learned,
- compile reusable skills,
- update plastic parameters,
- run regression tests before slow core consolidation.

---

## 1.5. One identity, many internal organs

All three loops belong to one persistent `LeviathanAgent`. Models, tools, search
branches, hypotheses and Cognitive Parameter Cells are internal resources, not
separately goaled agents. The unified agent alone owns:

- the immutable original-goal reference,
- serialized task/belief state,
- the append-only trajectory,
- action contracts and the external execution gateway,
- the boundary to independent verification,
- verified promotion into durable memory.

Between the numerical function substrate and the larger transformation primitives,
Leviathan adds an L1.5 **Parameter Ecology**. A sparse set of stateful parameter cells
forms proposals, exchanges bounded aggregate messages, recruits peers when disagreement
is high, and either converges or stops at a hard budget. Cells cannot act, verify, write
durable memory or alter governance. Repeated externally verified coalitions may be
compiled into cheaper procedural routing.

The current Python implementation is a behavioral reference for these contracts. The
neural hypothesis still requires function-preserving basis decomposition, routing,
communication and ablation experiments. See `docs/17-one-agent-recursive-plan.md`.

---

## 2. High-level module graph

```text
                    USER / WORLD
                         |
                         v
             ADAPTIVE MULTIMODAL PERCEPTION
              |        |        |       |
            text     vision    audio   sensors
              \        |        |      /
               \       |        |     /
                v      v        v    v
                 ADAPTIVE REPRESENTATION
                         |
                         v
                  PERSISTENT BELIEF STATE
          facts / hypotheses / causal graph /
            uncertainty / provenance / goals
                         |
                         v
                  METACOGNITIVE CONTROLLER
        +---------+---------+---------+---------+
        |         |         |         |         |
      recall    reason    search   simulate   experiment
        |         |         |         |         |
        +---------+---------+---------+---------+
                         |
                         v
                  HIERARCHICAL PLANNER
                 goal -> subgoal -> act
                         |
          +--------------+---------------+
          |              |               |
        tools         GUI/API          robot/speech
          |              |               |
          +--------------+---------------+
                         |
                         v
                     ENVIRONMENT
                         |
                         v
                    VERIFICATION
        reality / proof / execution / measurement /
              independent critic / learned PRM
                         |
                         v
                 CAUSAL CREDIT ASSIGNMENT
                         |
                         v
                      MEMORY BUS
          working / episodic / semantic / procedural
                         |
                         v
                CONSOLIDATION ENGINE
                 |                  |
              skills            plastic updates
                 |                  |
                 +--------+---------+
                          v
                    improved system
```

---

## 3. Persistent belief state

A transcript is not enough. Leviathan maintains explicit state:

```text
BeliefState
  facts[]
  hypotheses[]
  causal_edges[]
  uncertainties[]
  provenance[]
  goals[]
  constraints[]
  contradictions[]
  active_plan
  failed_attempts[]
  open_questions[]
```

Each consequential belief should include:

`(value, confidence, provenance, timestamp, evidence_refs, contradiction_refs)`

This makes uncertainty and source quality computable rather than rhetorical.

### Belief revision

Given predicted observation `o_hat` and actual observation `o`:

`prediction_error = distance(o_hat, o)`

The update should change:
- confidence in the current hypothesis,
- estimated causal edges,
- uncertainty,
- plan priority,
- whether a new experiment is required.

---

## 4. Metacognitive routing

The controller chooses a cognitive mode before the system commits expensive compute.

Candidate modes:

- `DIRECT` — answer/act from high-confidence current state.
- `RECALL` — retrieve episodic/semantic memory.
- `SKILL` — execute a compiled procedure.
- `REASON` — serial deliberation.
- `SEARCH` — beam/tree search over alternatives.
- `EVOLVE` — population-based candidate generation and selection.
- `SIMULATE` — use a world model/counterfactual predictor.
- `EXPERIMENT` — act to reduce epistemic uncertainty.
- `TOOL` — delegate exact work to deterministic software.
- `PARALLELIZE` — create a dependency DAG and run independent branches concurrently.
- `ASK` — request missing information when external input is the highest-value action.
- `ACT` — perform a grounded action.

Conceptual utility:

`U(mode) = expected_success_gain + beta*information_gain - lambda*compute - mu*latency - rho*risk`

The controller is itself a learning target.

---

## 5. Hierarchical planning

Plans should be represented at multiple abstraction levels:

`goal -> semantic subgoals -> operations -> action chunks -> low-level action`

Example:

```text
Goal: fix a production latency regression
  1. identify regression window
  2. compare releases
  3. profile bottleneck
  4. generate candidate causes
  5. choose discriminating tests
  6. implement candidate fix
  7. benchmark
  8. verify no regressions
```

The planner should attach dependencies so independent branches can run concurrently.

---

## 6. Multiple search geometries

No single reasoning algorithm is optimal for all tasks.

| Problem shape | Preferred mode |
|---|---|
| familiar + deterministic | skill/direct |
| familiar multi-step | hierarchical procedure |
| low branching | serial reasoning |
| moderate branching | beam/tree search |
| high branching + cheap verifier | population/evolution |
| unknown causal mechanism | experiment |
| physical uncertainty | simulation + action |
| formal domain | search + formal verifier |

The controller should learn this mapping from experience.

---

## 7. Native output heads

A common semantic state should fan out to specialized decoders:

`semantic_state -> {text, speech, GUI, API, robot, image/video}`

Examples:
- GUI action -> region/action distribution, not stringified `(x,y)` whenever possible.
- robot action -> continuous action chunk or specialized action tokens.
- API action -> schema-valid structured call.
- speech -> acoustic/codec stream via a specialized decoder.

This removes unnecessary translation through natural language.

---

## 8. Tool hierarchy

Prefer the highest-structure interface available:

`formal solver / API > database/query > accessibility/UI tree > vision grounding > raw pixel coordinate`

The model should not emulate deterministic computation if a trusted tool exists.

Examples:
- arithmetic -> calculator,
- code correctness -> compiler/tests,
- database aggregation -> query engine,
- proof validity -> theorem prover,
- exact file transformation -> code,
- GUI automation -> structured accessibility/API when available.

---

## 9. World modeling

Leviathan treats the world model as more than a video generator.

Required functions:

1. **state estimation** — infer latent current state.
2. **forward dynamics** — predict `state_{t+1}` under candidate action.
3. **counterfactual comparison** — compare alternative actions.
4. **uncertainty estimation** — know where predictions are weak.
5. **causal diagnosis** — estimate which prior decisions contributed to an outcome.
6. **simulation provenance** — mark simulated experience so it is not silently treated as real.

The preferred representation may combine semantic state with latent dynamics rather than requiring photorealistic generation for every prediction.

---

## 10. Verification hierarchy

Trust should be ordered approximately:

1. grounded physical reality,
2. formal proof,
3. deterministic execution/test,
4. calibrated trusted measurement,
5. independent model/verifier,
6. learned reward/process model,
7. same-model self-critique.

The lower the verifier, the smaller the permitted learning update.

---

## 11. Memory hierarchy

Leviathan uses five memory classes:

### Working
Immediate task state. Small, fast, discardable.

### Episodic
Structured experience:

`prediction -> action -> observation -> verification -> outcome`

### Semantic
Generalized facts/relationships distilled from multiple episodes.

### Procedural
Executable/reusable skills compiled from successful trajectories.

### Parametric
Knowledge internalized into plastic or core parameters.

Not every memory should be promoted through all five stages.

---

## 12. Cognitive compilation

Repeated successful reasoning should become progressively cheaper:

`novel task -> expensive search -> verified trajectory -> reusable skill -> semantic abstraction -> parametric intuition`

This yields **cognitive amortization**:

- first success: high cost,
- later successes: lower cost,
- mature skill: near-direct execution.

The desired lifetime trend is:

`capability / compute -> upward`

for skills repeatedly encountered and verified.

---

## 13. Continual learning boundary

Core weights must not update directly from raw experience.

Promotion path:

```text
raw trajectory
  -> episodic store
  -> verifier gate
  -> novelty/utility gate
  -> semantic or procedural memory
  -> repeated evidence
  -> plastic parameter candidate
  -> regression/safety evaluation
  -> shadow deployment
  -> optional slow core consolidation
```

Core-model learning should be the slowest and most conservative memory mechanism.

---

## 14. Governance boundary

The learner must not have unilateral control over:

- evaluator definitions,
- safety/regression suites,
- model signing,
- deployment promotion,
- rollback systems,
- root training infrastructure.

`learner != governor`

This separation is required because continual learning otherwise creates a circular authority problem: the system could learn to redefine the criteria that judge its learning.

---

## 15. Research blockers

Even with every component above implemented, the following remain unsolved:

- universal persistent belief representation,
- reliable open-world reward/verification,
- causal credit across thousands of actions,
- continual learning without destructive forgetting or goal drift,
- calibration after repeated self-modification,
- deciding when simulator experience is trustworthy,
- learning a cross-domain metacognitive routing policy,
- maintaining verifier independence,
- safely controlling self-generated curriculum.

These are first-class research problems, not glue-code tasks.

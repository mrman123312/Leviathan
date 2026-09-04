# Roadmap and Research Gaps

Leviathan should be built in stages. The goal is not to connect frontier models into a
runtime collective. Each promoted stage must remain one cognitive parameter state and
one checkpoint while validating progressively harder hypotheses with measurable failure
boundaries. External stores, tools and verifiers are infrastructure, not co-reasoners.

## Stage 0 — Instrumented agent baseline

Build:

- one open reasoner checkpoint,
- structured tool registry,
- deterministic code sandbox,
- task/trajectory logger,
- explicit success metrics,
- vLLM or SGLang serving,
- no self-modification.

Measure:

- success rate,
- wall-clock time,
- reasoning tokens,
- tool calls,
- failure types,
- verifier coverage.

Exit criterion:

The system can perform long multi-tool tasks reproducibly and every important action is logged with enough structure for later analysis.

## Stage 1 — Persistent belief state

Add:

- facts/hypotheses/uncertainty/provenance,
- contradictions,
- active goals and constraints,
- prediction-before-action records,
- searchable historical episodes.

Research question:

Does explicit state reduce repeated reasoning, contradiction and context reconstruction compared with raw transcript agents?

Exit criterion:

Belief-state version shows better long-horizon consistency and lower repeated-context cost on controlled benchmarks.

## Stage 2 — Multi-tier external memory

Add:

- working memory,
- episodic store,
- semantic abstraction,
- skill registry.

No weight updates yet.

Research question:

Can the agent become measurably better across repeated task families solely through memory and procedural compilation?

Exit criterion:

Repeated tasks require fewer tokens/tool calls while success stays equal or improves.

## Stage 3 — Metacognitive routing

Implement the mode controller:

`DIRECT | RECALL | SKILL | REASON | SEARCH | TOOL | PARALLELIZE | ASK`

Train initially from heuristic policies and offline trajectory data.
The learned controller must be a head of the same model, not a separately served model.

Research question:

Can the controller select cheaper cognition without sacrificing reliability?

Exit criterion:

Controller beats fixed high-reasoning and fixed low-reasoning baselines on cost per successful task.

## Stage 4 — Verification portfolio

Add:

- tests/compilers,
- formal verifiers where available,
- independent model critics,
- learned PRMs/reward models,
- verifier reliability metadata.

Research question:

Can the system estimate which verifier deserves trust for which claim?

Exit criterion:

Verifier-weighted confidence is better calibrated than same-model self-evaluation.

## Stage 5 — Active experimentation

Add explicit hypotheses and information-gain action selection.

Research question:

Can the system solve hidden-rule/causal tasks with fewer environment actions by choosing discriminating experiments?

Exit criterion:

Improved action efficiency and hypothesis calibration over random/generic exploration.

## Stage 6 — World-model integration

Add a latent or generative predictive model.

Functions:

- short-horizon prediction,
- counterfactual action comparison,
- simulated pre-testing,
- uncertainty estimation.

Research question:

Does simulation improve decisions after accounting for simulator bias and extra compute?

Exit criterion:

Net improvement on grounded tasks with no significant increase in false confidence caused by simulator errors.

## Stage 7 — Search-geometry routing

Add:

- serial reasoning,
- beam/tree search,
- population/evolution search,
- simulation,
- experiment selection.

Research question:

Can the meta-controller learn which search structure fits which problem?

Exit criterion:

Portfolio routing outperforms any one fixed search method across mixed domains.

## Stage 8 — Self-generated curriculum in verifiable domains

Begin only in domains with strong external verifiers:

- code,
- formal mathematics,
- deterministic puzzles,
- controlled software environments.

Add AZR-like proposer/solver loops.

Research question:

Can the system create tasks that maximize actual learning progress rather than novelty or reward hacks?

Exit criterion:

Held-out capability improves from self-generated data without human-created new tasks.

## Stage 9 — Plastic parameter learning

Introduce reversible updates only:

- adapters,
- LoRA-like modules,
- sparse experts,
- specialized heads.

Add:

- replay,
- stability losses,
- regression suite,
- safety suite,
- calibration suite,
- rollback.

Research question:

Can verified experience improve parametric competence without damaging old skills?

Exit criterion:

Positive new-skill gain with statistically bounded regression across protected capabilities.

## Stage 10 — Transactional consolidation

Never update the live model directly.

Pipeline:

`experience batch -> sandbox candidate -> evals -> adversarial evals -> shadow mode -> promote/rollback`

Research question:

Can model changes be managed like auditable software releases?

Exit criterion:

Every deployed update has reproducible provenance, test evidence and rollback path.

## Stage 11 — Native action heads

Add specialized execution paths:

- GUI action model,
- structured API action,
- speech output,
- robot/action chunking if physical hardware is available.

Research question:

When does native action representation materially outperform language serialization?

## Stage 12 — Generated environments and broad self-experience

Only after strong provenance and verifier controls exist.

Use:

- procedurally generated simulations,
- software worlds,
- robotics simulation,
- learned world models.

Require periodic reality validation.

## Stage 13 — Slow core consolidation

This should be the last step, not the first.

Core updates require:

- repeated verified evidence,
- high transfer value,
- broad regression suite,
- calibration preservation,
- safety preservation,
- independent promotion authority.

---

# Major unsolved problems

## 1. Stable universal belief state

No current system has demonstrated a representation that integrates factual, perceptual, causal, temporal and uncertain knowledge over a very long autonomous lifetime.

## 2. Open-world verification

Code and formal proofs have exact feedback. Most human goals do not.

We still lack a general solution for trustworthy reward in science, strategy, social reasoning, design and other open domains.

## 3. Causal credit assignment

An outcome may depend on a belief formed thousands of steps earlier. Flat temporal reward is not enough.

## 4. Continual learning at frontier scale

Open questions:

- how much of the model should be plastic,
- how often updates should happen,
- how to prevent forgetting,
- how to prevent semantic drift,
- how to preserve calibration.

## 5. Metacognitive generalization

A controller trained on familiar task families may choose the wrong cognition mode for genuinely novel problems.

The system needs a way to recognize its own routing uncertainty.

## 6. Simulator overfitting

A world model can create infinite experience, but its errors can become infinite false training data.

## 7. Curriculum integrity

A self-generated learner may prefer:

- easy tasks,
- easy verifiers,
- tasks that flatter its own strengths,
- novelty without usefulness.

Curriculum objectives must reward transfer and real learning progress.

## 8. Verifier capture

A self-improving agent may learn how to exploit the evaluator rather than improve the underlying task.

## 9. Goal continuity

Long-lived systems need a stable representation of user/system goals that cannot be casually rewritten by later self-generated state.

## 10. Governance of self-modification

The architecture must preserve a control boundary between:

- the learner,
- evaluators,
- deployment authority,
- safety policy,
- rollback infrastructure.

## 11. Economic feasibility

An architecture can be cognitively elegant and still be unusable if world models, search populations and verification consume more compute than they save.

Every new cognitive module must be evaluated on **cost per successful task**.

## 12. Benchmark design

Existing static benchmarks under-measure:

- lifetime learning,
- skill compilation,
- action efficiency,
- causal discovery,
- metacognitive routing,
- calibration after self-modification,
- resilience to misleading self-generated experience.

Leviathan will need its own longitudinal evaluations.

---

# Recommended near-term research priority

If resources are limited, prioritize in this order:

1. persistent belief state,
2. procedural skill compilation,
3. metacognitive routing,
4. verifier portfolio and calibration,
5. active experimentation,
6. world-model planning,
7. self-generated curriculum in exact domains,
8. reversible plastic learning,
9. only then slow core-weight consolidation.

This ordering deliberately postpones the most dangerous and least understood mechanism—continual foundation-weight modification—until the surrounding memory, verification and governance system exists.

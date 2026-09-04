# Failure Modes and Safety Boundaries

Leviathan is designed around the assumption that every cognitive mechanism can fail. The architecture therefore needs explicit failure classes, observability and rollback.

## 1. Hallucination consolidation

Failure:

`model invents claim -> stores claim as memory -> retrieves it repeatedly -> confidence increases -> claim enters semantic/parametric learning`

Defense:
- provenance on every belief,
- verifier gate before promotion,
- contradiction tracking,
- no core-weight update from self-evaluation alone.

## 2. Catastrophic forgetting

Failure:

new learning improves a narrow skill but destroys unrelated old knowledge or behavior.

Defense:
- plastic modules before core updates,
- replay,
- stability regularization,
- protected regression suites,
- shadow deployment,
- rollback.

## 3. Semantic drift

Failure:

Repeated summarization and self-generated semantic memory slowly change the meaning of concepts.

Defense:
- preserve links to original episodes/evidence,
- version semantic memories,
- periodically re-ground abstractions against source evidence,
- treat summaries as derived artifacts rather than immutable truth.

## 4. Simulator delusion

Failure:

world-model errors become training data and the agent becomes highly competent inside its own mistaken simulator.

Defense:
- simulation provenance,
- uncertainty estimates,
- real-world spot checks,
- limited learning weight for unvalidated simulation,
- independent environment models when feasible.

## 5. Reward hacking

Failure:

agent maximizes a proxy while failing the true task.

Examples:
- stop early to save tokens,
- choose only easy subproblems,
- exploit learned verifier weaknesses,
- redefine success,
- avoid uncertain domains.

Defense:
- immutable original goal record,
- multiple independent metrics,
- hidden/random audits,
- hard constraints,
- external outcome checks.

## 6. Verifier capture

Failure:

model learns patterns that fool its reward model or chooses the weakest verifier available.

Defense:
- verifier independence scoring,
- formal/deterministic checks where possible,
- verifier rotation/ensembles,
- adversarial evaluation,
- do not allow the learner to modify final promotion criteria.

## 7. Correlated critic failure

Failure:

generator and critic share the same blind spot.

Defense:
- different model families/data when possible,
- deterministic tools,
- formal systems,
- real measurements,
- institutional separation for high-stakes evaluation.

## 8. Early hypothesis lock-in

Failure:

initial guess shapes all later reasoning and evidence interpretation.

Defense:
- explicit competing hypotheses,
- branch search,
- population search,
- information-gain experiments,
- periodically regenerate hypotheses from evidence without the original chain.

## 9. Endless deliberation

Failure:

agent spends unbounded reasoning/search because additional work always appears potentially useful.

Defense:
- explicit marginal-value stopping rule,
- task budgets,
- opportunity-cost term,
- escalating proof burden for additional compute.

## 10. Premature stopping

Failure:

efficiency reward causes the agent to stop before adequate verification.

Defense:
- minimum verifier requirements,
- task completeness checks,
- risk-adjusted reasoning budget,
- penalty for unresolved high-impact uncertainty.

## 11. Tool explosion

Failure:

large tool libraries consume context and increase selection errors.

Defense:
- tool retrieval,
- hierarchical namespaces,
- tool capability metadata,
- schema validation,
- tool-use skill compilation.

## 12. Unsafe tool chaining

Failure:

individually harmless actions combine into an unintended irreversible outcome.

Defense:
- plan-level risk evaluation,
- dependency graph inspection,
- reversible/dry-run modes,
- approval boundaries for irreversible actions.

## 13. Memory poisoning

Failure:

malicious or low-quality observations enter long-term memory and influence future decisions.

Defense:
- source trust scores,
- quarantine uncertain memories,
- conflict detection,
- no promotion without corroboration,
- signed/trusted provenance for critical data.

## 14. Skill poisoning

Failure:

a compiled procedural skill encodes a brittle or malicious behavior and then executes with less scrutiny because it is considered familiar.

Defense:
- typed preconditions,
- versioned skills,
- verifier attached to each skill,
- periodic re-evaluation,
- privilege limits,
- deprecation when environment changes.

## 15. Skill overgeneralization

Failure:

procedure learned in one context is applied outside its valid domain.

Defense:
- explicit scope/preconditions,
- confidence based on environment similarity,
- fallback to reasoning when preconditions are uncertain.

## 16. Calibration collapse

Failure:

model accuracy improves or shifts but confidence no longer tracks correctness.

Defense:
- calibration evaluation after every model update,
- calibration by domain/provenance/mode,
- confidence correction layers where appropriate.

## 17. Metacognitive routing collapse

Failure:

controller repeatedly selects a cheap but inappropriate mode, or an expensive mode for everything.

Defense:
- monitor mode selection distributions,
- compare against fixed baselines,
- retain exploration in routing,
- explicit uncertainty about routing itself.

## 18. Hardware-local optimum

Failure:

router picks models/experts that are mathematically strong but cause severe cluster congestion or communication stalls.

Defense:
- include hardware load in routing utility,
- admission control,
- load-aware expert biases,
- fallback model portfolio.

## 19. Information-gain obsession

Failure:

agent constantly explores because uncertainty reduction is rewarded even when it does not serve the real goal.

Defense:
- information gain multiplied/conditioned by task relevance and expected utility,
- opportunity-cost penalty,
- goal progress as primary constraint.

## 20. Curriculum collapse

Failure:

self-generated curriculum becomes:
- too easy,
- impossibly hard,
- repetitive,
- unverifiable,
- irrelevant but novel.

Defense:
- target empirical learning progress,
- diversity constraints,
- held-out transfer tests,
- verifier-quality weighting,
- external curriculum audits.

## 21. Self-referential objective drift

Failure:

agent changes internal representations of its goal/reward until easier behaviors appear acceptable.

Defense:
- immutable external goal record,
- separate governor,
- trace every goal transformation,
- verify plan against original goal before final execution.

## 22. Core-update cascade

Failure:

one problematic core update changes the behavior of memory extraction, verification and subsequent learning, causing compounding drift.

Defense:
- core updates rare,
- transactional release process,
- canary/shadow versions,
- frozen reference model,
- rollback across multiple generations.

## 23. Experience monoculture

Failure:

self-generated experience narrows around the model's current strengths and misses unknown unknowns.

Defense:
- externally sourced challenges,
- adversarial generators,
- novelty measured against capability gaps,
- cross-environment evaluation.

## 24. Measurement gaming

Failure:

agent manipulates a sensor/benchmark rather than improving the underlying reality.

Defense:
- independent measurements,
- tamper-resistant instrumentation,
- causal cross-checks,
- hidden evaluation channels.

## 25. Governance bypass

Failure:

learner gains ability to modify deployment, rollback, verifier or safety infrastructure.

Defense:
- privilege separation,
- isolated credentials,
- signed model artifacts,
- external deployment service,
- no self-issued promotion.

---

# Required invariants

A serious Leviathan implementation should enforce at least these invariants:

1. **No raw experience directly changes core weights.**
2. **Every promoted memory retains provenance.**
3. **Every irreversible action has explicit authority/risk handling.**
4. **Every parametric update has a rollback path.**
5. **The learner cannot unilaterally redefine final verifiers.**
6. **Simulated evidence remains labeled as simulated.**
7. **High-impact uncertain beliefs cannot silently become facts during compaction.**
8. **Model confidence is re-calibrated after updates.**
9. **Original task goals remain externally inspectable.**
10. **Cost/efficiency objectives cannot override minimum safety/verification requirements.**

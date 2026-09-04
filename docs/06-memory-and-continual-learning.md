# Memory and Continual Learning

Leviathan treats memory and learning as a promotion pipeline rather than a single database or immediate weight update.

## 1. Five memory classes

### Working memory
Fast, small, task-local state.

Contains:
- current goal,
- active subgoals,
- immediate evidence,
- current hypotheses,
- tool results still needed,
- temporary variables.

Working memory is disposable and should be aggressively compacted.

### Episodic memory
Structured records of experience:

`prediction -> action -> observation -> verification -> consequence`

Episode fields should include:

```text
Episode
  id
  timestamp
  context summary
  beliefs_before[]
  prediction
  action
  observation
  verifier evidence
  outcome
  confidence_before
  confidence_after
  provenance
  causal links
  reusable?
```

### Semantic memory
Generalized knowledge extracted across episodes.

A semantic claim should retain links back to supporting episodes and contradictory evidence.

### Procedural memory
Reusable executable skills or policies.

A skill should contain:

```text
Skill
  name
  purpose
  preconditions
  inputs
  steps/tool calls
  expected outputs
  verifier
  failure recovery
  provenance
  success statistics
  version
```

### Parametric memory
Knowledge encoded in model parameters.

Leviathan separates:

- **plastic parameters** — adapters/sparse modules/skill heads that can update relatively often,
- **core parameters** — slow, high-risk consolidation target.

## 2. Promotion pipeline

Raw experience must never write directly to core weights.

```text
raw event
  -> episodic record
  -> verifier gate
  -> novelty gate
  -> consistency gate
  -> utility/transfer gate
  -> semantic or procedural memory
  -> repeated evidence
  -> plastic update candidate
  -> regression/safety evaluation
  -> shadow deployment
  -> optional core consolidation
```

Each stage requires stronger evidence.

## 3. Trust-weighted learning

For experience `e`, define a learning trust score:

`w(e) = truth_quality * provenance_quality * novelty * utility * consistency * verifier_reliability`

The exact function should be calibrated, not treated as a universal formula.

Possible policy:

- formal proof / deterministic test -> high trust,
- trusted physical measurement -> high trust,
- independent corroboration -> medium-high trust,
- learned verifier -> medium trust,
- simulator-only result -> conditional trust,
- same-model self-evaluation -> low trust.

## 4. Plastic versus core learning

Model state:

`theta_system = theta_core + phi_plastic + external_memory`

Learning rates should satisfy roughly:

`eta_memory >> eta_plastic >> eta_core`

Interpretation:

- memory updates constantly,
- plastic modules update under verified evidence,
- core model changes rarely after extensive evaluation.

## 5. Candidate update loss

A safe continual update should optimize new learning while preserving old capability:

`L = L_new + lambda_replay*L_replay + lambda_stability*L_stability + lambda_safety*L_safety`

Where:

- `L_new` learns verified new experiences,
- `L_replay` replays old capabilities,
- `L_stability` penalizes destructive parameter drift,
- `L_safety` protects alignment/safety behaviors.

## 6. Replay

A continual learner must not train exclusively on today's experiences.

Replay should sample:

- high-value old knowledge,
- rare skills,
- safety invariants,
- calibration cases,
- adversarial regression tests,
- known catastrophic-forgetting probes.

Replay sampling itself should be adaptive: capabilities showing early regression deserve more replay weight.

## 7. Cognitive compilation

A key lifetime-learning mechanism is converting expensive successful trajectories into cheaper procedures.

```text
first encounter:
  more active parameters/passes + search state + tools

repeated verified encounters:
  abstract common structure

stable pattern:
  executable procedural skill

broad transferable pattern:
  semantic abstraction

very stable/general pattern:
  plastic or core parameter consolidation
```

This is **cognitive amortization**.

Desired trend for repeated task family `T`:

`compute_required(T, experience_count) -> down`

while:

`success_rate(T, experience_count) -> up`

## 8. Skill compilation criteria

Compile a trajectory into a skill when:

- the task pattern is recurring,
- the successful procedure is stable,
- preconditions are identifiable,
- outputs can be verified,
- failures have known recovery paths,
- the procedure saves meaningful compute/time,
- it does not encode brittle context-specific assumptions as universal rules.

## 9. Skill invalidation and versioning

Procedural memory can become stale.

Each skill should track:

- success rate,
- last verification date,
- environment/version assumptions,
- known failure cases,
- deprecation status.

A skill that begins failing should be demoted or retrained rather than silently trusted.

## 10. Semantic consolidation

Semantic memory should be built from multiple episodes, not a single confident anecdote.

A semantic claim needs:

- supporting episodes,
- contradictory episodes,
- domain/scope,
- confidence,
- source quality,
- known exceptions.

This prevents accidental overgeneralization.

## 11. Plastic module strategies

Candidate mechanisms include:

- LoRA/adapters,
- sparse experts,
- expandable memory modules,
- specialized heads,
- task/skill modules,
- retrieval-conditioned adapters.

The architecture does not assume one technique is best. The important principle is reversibility: plastic updates should be easy to isolate, test and roll back.

## 12. Transactional learning

Never patch the live production model in-place.

```text
live model theta_t
  -> verified experience batch
  -> sandbox candidate theta'
  -> capability evals
  -> old-skill regression
  -> safety evals
  -> calibration evals
  -> adversarial evals
  -> shadow deployment
  -> promote or destroy candidate
```

Learning should behave like a transaction: commit only if all required invariants hold.

## 13. Shadow deployment

A candidate model/version can receive real tasks without controlling real outcomes.

Compare candidate versus current production on:

- task quality,
- latency,
- token/compute cost,
- hallucination/error rate,
- calibration,
- safety,
- goal adherence,
- old-skill regressions.

Only then promote.

## 14. Continual calibration

After every learning cycle, recompute calibration by domain and verifier class.

A model can improve average accuracy while becoming dangerously overconfident. Calibration must therefore be an independent promotion gate.

## 15. Self-generated experience

Experience generated by AZR-like curricula, simulated environments or self-play should be tagged and weighted by provenance.

A simulated discovery should not become a real-world semantic fact without grounded confirmation.

## 16. Curriculum selection

A useful curriculum target maximizes approximately:

`learning_progress * expected_usefulness * transfer * verifier_reliability / (cost + risk)`

Avoid objectives that reward novelty alone; otherwise the system may spend its lifetime exploring unpredictable but useless states.

## 17. Governance

The learning system must not control:

- the final verifier suite,
- deployment promotion,
- rollback authority,
- signing keys,
- safety invariants,
- privileged training infrastructure.

A learner that can rewrite both itself and the criteria that approve its rewrites has no meaningful external control boundary.

## 18. Main unsolved research issues

- catastrophic forgetting at foundation-model scale,
- false-memory consolidation,
- correlated verifier errors,
- poisoning by malicious experience,
- distinguishing novelty from noise,
- preserving calibration after many updates,
- semantic drift across years,
- deciding which experiences deserve parametric learning,
- proving that repeated self-training does not gradually narrow or distort the system's world model.

# World and Belief Model

Leviathan distinguishes a **world model** from a **belief state**.

A world model predicts how states change. A belief state records what the agent currently thinks is true, how sure it is, and why. General intelligence needs both.

## 1. State representation

A minimal belief object:

```text
Belief
  id
  proposition / latent key
  value
  confidence
  uncertainty_type
  provenance
  timestamp
  evidence_refs[]
  contradiction_refs[]
  causal_parents[]
  causal_children[]
  scope
  status
```

`status` can include:

- observed,
- inferred,
- simulated,
- hypothetical,
- verified,
- contradicted,
- deprecated.

## 2. Provenance classes

Every belief/experience should retain where it came from:

```text
REAL_OBSERVATION
FORMAL_RESULT
DETERMINISTIC_EXECUTION
TRUSTED_MEASUREMENT
EXTERNAL_SOURCE
INDEPENDENT_MODEL
LEARNED_VERIFIER
SIMULATION
SELF_INFERENCE
SELF_EVALUATION
```

This provenance later controls learning strength.

## 3. Uncertainty

Leviathan separates:

### Aleatoric uncertainty
The environment itself is stochastic or noisy.

### Epistemic uncertainty
The agent lacks information or has competing explanations.

Only epistemic uncertainty is generally reducible through additional evidence.

The controller therefore needs to distinguish:

`uncertainty -> can_action_reduce_it?`

rather than simply reacting to a generic confidence score.

## 4. Hypothesis set

Do not collapse uncertainty prematurely into one narrative.

```text
HypothesisSet
  H1: probability/confidence + evidence + predictions
  H2: probability/confidence + evidence + predictions
  H3: probability/confidence + evidence + predictions
```

Each hypothesis should make explicit predictions. An explanation that cannot generate differentiating predictions is difficult to verify.

## 5. Forward dynamics

World-model interface:

`predict(state, action, horizon) -> distribution(next_states, observations, rewards, confidence)`

The system should be able to ask:

- What happens if I do A?
- What happens if I do B instead?
- Which result would distinguish H1 and H2?
- How uncertain is the prediction?

Latent predictive models are preferred when photorealistic generation is unnecessary.

## 6. Counterfactual planning

For candidate action `a`:

`Q(a) = expected_task_value(a) + beta*information_gain(a) - cost(a) - risk(a)`

The world model is not only a planner. It is also a way to estimate the information value and reversibility of actions.

## 7. Active experimentation

Suppose:

- H1: latency is database-bound.
- H2: latency is network-bound.
- H3: latency is CPU contention.

A weak agent collects more generic evidence.

A stronger agent searches for:

`a* = argmax expected reduction in uncertainty(H1,H2,H3)`

Examples:
- disable/query-cache test,
- isolated network benchmark,
- CPU flamegraph.

The preferred experiment has high discriminative power, low cost and a trustworthy outcome.

## 8. Belief revision

Before action:

`predicted_outcome = world_model(state, action)`

After action:

`prediction_error = compare(predicted_outcome, observed_outcome)`

Use prediction error to update:

- hypothesis confidence,
- causal links,
- model calibration,
- future action policy,
- whether the episode qualifies for learning.

## 9. Causal graph

Leviathan should distinguish correlation from proposed cause.

```text
Node: deployment X
  -> changed dependency Y
  -> increased lock contention
  -> increased tail latency
```

Each edge should carry:

- evidence strength,
- intervention evidence if available,
- source/provenance,
- known confounders,
- confidence.

A mature causal graph helps both forward planning and backward credit assignment.

## 10. Backward diagnosis

World models should support retrospective questions:

> Which earlier belief/action most likely caused the final failure?

A conceptual counterfactual contribution for node `i`:

`credit_i ~= outcome(actual trajectory) - expected outcome(trajectory with node i changed)`

Exact counterfactuals are usually impossible. A learned causal/world model can estimate them, but these estimates must be treated as uncertain rather than exact reward.

## 11. Simulation is not reality

Simulated trajectories must carry provenance.

A model-generated environment can be useful for:

- cheap planning,
- curriculum generation,
- exploration,
- safety pre-testing,
- hypothesis filtering.

But an agent can overfit to simulator errors.

Therefore learning weight should generally obey:

`grounded real/formal evidence > validated simulation > weak simulation > self-evaluation`

Simulated discoveries should be promoted only after reality checks when the claim concerns the real world.

## 12. State compaction

A persistent belief state cannot grow without bound.

Compaction should:

- preserve high-confidence causal facts,
- preserve unresolved uncertainty,
- retain provenance links,
- summarize redundant episodes,
- archive low-relevance raw data,
- never silently convert uncertain claims into facts.

The key invariant:

`compression must not erase epistemic status`.

## 13. Calibration

The system should periodically test whether confidence values predict actual correctness.

For beliefs with confidence near 0.8, roughly 80% should survive trustworthy verification over time in the relevant domain.

Calibration should be tracked by:

- domain,
- provenance class,
- cognitive mode,
- verifier type,
- model/version.

This matters after continual updates because self-modification can silently destroy prior calibration.

## 14. Interface boundaries

The belief model does not decide actions on its own.

It exposes:

- current state,
- hypotheses,
- uncertainty,
- world-model predictions,
- causal dependencies,
- provenance.

The meta-controller and planner decide how to use that information.

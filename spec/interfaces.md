# Module Interfaces

This file defines conceptual contracts between Leviathan modules. The goal is to avoid hiding system state inside opaque prompts.

## Belief

```python
Belief(
    id: str,
    proposition: str | None,
    latent_key: str | None,
    value: object,
    confidence: float,
    uncertainty_type: Literal['aleatoric', 'epistemic', 'mixed', 'unknown'],
    provenance: Provenance,
    timestamp: datetime,
    evidence_refs: list[str],
    contradiction_refs: list[str],
    causal_parents: list[str],
    causal_children: list[str],
    status: Literal['observed','inferred','simulated','hypothetical','verified','contradicted','deprecated'],
)
```

## Provenance

```python
Provenance(
    kind: Literal[
        'real_observation',
        'formal_result',
        'deterministic_execution',
        'trusted_measurement',
        'external_source',
        'independent_model',
        'learned_verifier',
        'simulation',
        'self_inference',
        'self_evaluation',
    ],
    source_id: str,
    source_version: str | None,
    trust_prior: float,
)
```

## Hypothesis

```python
Hypothesis(
    id: str,
    statement: str,
    confidence: float,
    supporting_evidence: list[str],
    contradicting_evidence: list[str],
    predictions: list[Prediction],
)
```

A hypothesis should produce testable predictions whenever possible.

## Prediction

```python
Prediction(
    id: str,
    hypothesis_id: str | None,
    action_id: str | None,
    expected_observation: object,
    confidence: float,
    horizon: int,
    timestamp: datetime,
)
```

Predictions are stored **before** outcomes to prevent hindsight rationalization.

## Observation

```python
Observation(
    id: str,
    modality: str,
    payload_ref: str,
    provenance: Provenance,
    timestamp: datetime,
    measurement_uncertainty: float | None,
)
```

## Cognitive mode decision

```python
CognitiveDecision(
    mode: CognitiveMode,
    expected_success_gain: float,
    expected_information_gain: float,
    expected_transfer_value: float,
    compute_cost: float,
    latency_cost: float,
    risk_cost: float,
    irreversibility_cost: float,
    hardware_cost: float,
    confidence: float,
    rationale_features: dict[str, float],
)
```

The system should log features/utility terms even if the internal policy is neural.

## Plan node

```python
PlanNode(
    id: str,
    parent_goal: str,
    description: str,
    dependencies: list[str],
    action_type: str,
    executor: str,
    verifier: str | None,
    expected_cost: float,
    expected_information_gain: float,
    risk: float,
    reversible: bool,
    status: str,
)
```

Plans form DAGs, not only lists.

## Action

```python
Action(
    id: str,
    kind: Literal['tool','api','gui','robot','speech','text','wait','ask'],
    executor: str,
    payload: object,
    preconditions: list[str],
    expected_outcome: Prediction | None,
    reversible: bool,
    authorization_class: str,
)
```

## Verification

```python
Verification(
    id: str,
    target_id: str,
    verifier_type: str,
    verifier_id: str,
    verifier_version: str | None,
    result: object,
    passed: bool | None,
    confidence: float,
    independence_score: float,
    calibration_bucket: str | None,
    provenance: Provenance,
    timestamp: datetime,
)
```

## Episode

```python
Episode(
    id: str,
    context_summary: str,
    beliefs_before: list[str],
    predictions: list[str],
    actions: list[str],
    observations: list[str],
    verifications: list[str],
    outcome: object,
    task_success: float,
    compute_cost: float,
    wall_clock_seconds: float,
    risk_events: list[str],
    causal_graph_ref: str,
    reusable_candidate: bool,
)
```

## Skill

```python
Skill(
    id: str,
    name: str,
    purpose: str,
    preconditions: list[str],
    inputs_schema: dict,
    outputs_schema: dict,
    procedure_ref: str,
    verifier_ref: str | None,
    recovery_ref: str | None,
    provenance_episode_ids: list[str],
    successes: int,
    failures: int,
    environment_assumptions: list[str],
    version: str,
    status: Literal['experimental','active','degraded','deprecated'],
)
```

## Learning candidate

```python
LearningCandidate(
    id: str,
    source_episode_ids: list[str],
    target: Literal['semantic_memory','procedural_memory','plastic_parameters','core_parameters'],
    truth_quality: float,
    provenance_quality: float,
    novelty: float,
    utility: float,
    consistency: float,
    transfer_value: float,
    verifier_reliability: float,
    aggregate_trust: float,
)
```

## Parameter update candidate

```python
ParameterUpdateCandidate(
    id: str,
    base_model_version: str,
    target_module: str,
    training_batch_ref: str,
    replay_batch_ref: str,
    safety_batch_ref: str,
    status: Literal['prepared','training','evaluating','shadow','approved','rejected','rolled_back'],
    eval_results: dict,
)
```

## Required event log

Every meaningful cognitive operation should emit an append-only event:

```python
Event(
    id: str,
    event_type: str,
    timestamp: datetime,
    module: str,
    input_refs: list[str],
    output_refs: list[str],
    model_version: str | None,
    cost: dict,
    metadata: dict,
)
```

This log is essential for later causal credit assignment and auditing.

## Trust contract

No module may convert a low-trust result into a high-trust belief simply by rewriting or summarizing it.

Formally:

`derived_trust <= f(source_trusts, independent_new_evidence)`

Compression alone cannot increase epistemic status.

## Learning contract

No `LearningCandidate(target='core_parameters')` may be promoted without:

- independent verification,
- protected replay suite,
- regression suite,
- safety suite,
- calibration suite,
- rollback artifact,
- shadow evaluation,
- external promotion authority.

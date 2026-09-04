# Verification and Credit Assignment

Leviathan treats verification and credit assignment as separate problems.

Verification asks:

> Did the outcome satisfy reality or a trustworthy criterion?

Credit assignment asks:

> Which prior beliefs, decisions, actions or representations caused that outcome?

A system can solve one and still fail badly at the other.

## 1. Verifier hierarchy

Approximate trust order:

1. **physical reality / grounded outcome**
2. **formal proof system**
3. **deterministic execution / tests**
4. **trusted measurement**
5. **independent external source/model**
6. **learned verifier / reward model / PRM**
7. **same-model self-critique**

This is not absolute. A noisy physical sensor may be less reliable than a formal computation. The principle is to track verifier reliability explicitly rather than treating all feedback as equal.

## 2. Verification record

```text
Verification
  claim_id
  verifier_type
  verifier_identity/version
  input/evidence refs
  outcome
  confidence
  calibration profile
  timestamp
  independent_from_generator?
  failure_modes[]
```

## 3. Generator-verifier separation

A common failure pattern:

`model generates claim -> same model grades claim -> model trains on grade`

This creates correlated errors and self-confirmation.

Prefer:

`generator -> independent verifier -> learning gate`

When possible, the verifier should differ in one or more of:

- architecture,
- parameters,
- data,
- reward function,
- execution substrate,
- institutional ownership.

Formal and deterministic verifiers are especially valuable because they reduce correlated linguistic bias.

## 4. Formal verification

DeepSeek-Prover-style systems illustrate a strong restricted-domain pattern:

`generate proof -> Lean checks -> valid/invalid`

Benefits:
- exact reward,
- decomposable subgoals,
- reliable search pruning,
- low ambiguity.

Limitation:
- only available where the problem can be formalized.

## 5. Program/environment verification

AZR-like training can use a Python executor:

`generate problem/solution -> execute -> pass/fail`

Software tasks can similarly use:

- unit tests,
- integration tests,
- compilers,
- benchmarks,
- static analyzers,
- runtime invariants.

These are stronger than a language-model opinion when they cover the requested objective.

## 6. Learned process reward

Process reward models can score intermediate steps:

`r_i = quality(step_i | evidence)`

This can reduce the temporal gap between an error and the reward signal.

However:

`learned_reward != truth`

The reward model itself requires calibration, adversarial evaluation and provenance.

## 7. Open-world verification

Many important goals do not have a binary verifier:

- design a better city,
- develop a scientific theory,
- write an excellent novel,
- choose a robust business strategy,
- understand why an experiment failed.

Leviathan therefore uses a **verification portfolio**:

```text
formal/deterministic checks
+ empirical measurements
+ consistency tests
+ independent evidence retrieval
+ adversarial critics
+ outcome tracking over time
```

No single weak verifier should produce a large irreversible learning update.

## 8. Prediction-before-observation

To learn causally, the agent should record what it expected **before** observing the result.

Episode:

`belief_before -> explicit prediction -> action -> actual observation -> prediction error`

Without the prior prediction, the model can rationalize any outcome after the fact.

## 9. Credit assignment graph

Represent a long trajectory as a dependency graph rather than a flat list:

```text
belief B1
  -> plan P1
      -> action A12
          -> observation O13
              -> hypothesis H7
                  -> plan P3
                      -> action A98
                          -> failure F
```

Then credit assignment can flow along causal/dependency edges.

## 10. Counterfactual contribution

A conceptual importance score for decision `d_i`:

`C_i ~= outcome(actual) - E[outcome | trajectory with d_i changed]`

The counterfactual expectation may come from:

- actual A/B intervention data,
- simulator/world model,
- replayed environment,
- alternative search branch,
- learned causal estimator.

Because these estimates are uncertain, credit values need confidence/provenance too.

## 11. Temporal credit

Simple discounted reward:

`r_t = gamma^(T-t) * R_T`

is often insufficient for long agent tasks. A decision hundreds of steps earlier may be more causally important than the final action.

Leviathan therefore combines:

- process rewards,
- dependency graphs,
- explicit belief/action provenance,
- counterfactual estimates,
- verifier-local signals,
- final outcome reward.

## 12. Reward objective

A conceptual task reward:

`R = task_success - time_cost - compute_cost - action_cost - error_cost - risk_cost`

Additional terms may include:

- information gain,
- transfer value,
- user preference alignment,
- safety margin,
- calibration.

Do not blindly optimize this scalar. Multi-objective constraints and hard safety invariants may be safer than combining everything into one reward.

## 13. Reward hacking defenses

Potential hacks:

- choose easier subgoals rather than the requested goal,
- manipulate the learned verifier,
- hide uncertainty,
- stop early to reduce cost,
- choose simulated tasks where rewards are easy,
- optimize proxy metrics while harming real outcomes,
- redefine success after the fact.

Defenses:

- immutable original-goal representation,
- external outcome checks,
- verifier diversity,
- adversarial audits,
- calibration monitoring,
- randomized hidden evaluations,
- provenance-aware reward limits,
- hard constraints for non-negotiable safety properties.

## 14. Verifier independence score

A useful metadata concept:

`independence = f(shared_model, shared_training_data, shared_prompt, shared_reward, shared_environment)`

A verifier with high correlation to the generator should receive less trust than an independent measurement with similar nominal accuracy.

## 15. Learning gate

Before a trajectory affects plastic parameters:

```text
success verified?
  yes
verifier reliable enough?
  yes
result genuinely novel/useful?
  yes
contradictions resolved?
  yes
causal credit sufficiently localized?
  yes
replay/safety batch prepared?
  yes
-> candidate update
```

## 16. Scientific/experimental mode

For open-ended discovery:

1. maintain competing hypotheses,
2. derive discriminating predictions,
3. choose high-information experiments,
4. record predictions before outcomes,
5. update hypotheses after measurements,
6. replicate high-impact results,
7. promote only reproducible patterns into semantic/parametric knowledge.

This is the proposed bridge from AZR-style verified self-play to open-world autonomous science.

## 17. Main unsolved problems

- trustworthy verification for subjective/open-ended goals,
- causal credit in partially observed environments,
- verifier corruption after distribution shift,
- correlated failure among learned evaluators,
- counterfactual estimation without accurate world models,
- preventing a self-improving model from selecting only easy-to-verify domains,
- balancing efficiency rewards against thoroughness and safety.

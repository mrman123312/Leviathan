# Fundamentals Ultra — Evidence-Gated Research Audit

Date: 2026-08-16 America/Toronto / 2026-08-17 UTC

## Purpose

Fundamentals Ultra is the competitive Leviathan line. It keeps the mature Stockfish substrate and accepts research only after causal tests. Greenfield and historical branches remain research donors; branch chronology is **not** a promotion rule.

Promotion ladder:

`correctness -> A/A calibration -> equal-node -> fixed-time -> deep-oracle regret -> paired games -> SPRT when justified`

A mechanism that improves one stage but loses the practical fixed-time/search-depth trade is rejected or returned to the lab.

## Frozen comparison controls

- Stockfish: `5062aee519a1ba262d472d8ab139851ced56573e`
- Stockfish NNUE: `nn-ab28990d4ea3.nnue`
- Fundamentals v2.1: `4e2c76f898bcbb22a8f79a6c8dd1286c2d80e513`
- Fundamentals phase-fix: `bd9c97c1433546b37327b8e0a0aef7c2810df7bc`
- Fresh paired corpus: 50 deterministic 8-ply openings, both colors, 100 games
- Corpus SHA-256: `444b13d062e5aa38e72f6ffc5a2f3aa8daf92d4b6712a2d40646c857b51795da`
- Corpus Stockfish depth-10 range: -62..+78 cp; mean absolute imbalance approximately 25 cp

The 100-game identical-binary A/A control scored 48.5%, demonstrating that differences of roughly one percentage point in a 100-game sample are still noise-scale rather than reliable Elo claims.

## Production baseline — PROMOTED

### Fundamentals v2.1.1 phase-fix

Status: **PROMOTED — correctness/quality improvement, no demonstrated strength tax**

Fresh 100-game results:

- v2.1 vs Stockfish @100 ms: `4W / 90D / 6L = 49.0%`
- phase-fix vs Stockfish @100 ms: `5W / 89D / 6L = 49.5%`
- phase-fix vs v2.1 @100 ms: `5W / 89D / 6L = 49.5%`
- phase-fix vs v2.1 @80k nodes/move: `3W / 94D / 3L = 50.0%`

Deep-oracle panel, 50 fresh corpus positions, 80k-node move choice / 500k-node Stockfish grade:

- phase-fix mean regret: **0.78 cp**
- v2.1 mean regret: **1.02 cp**
- phase-fix oracle agreement: **70%**
- v2.1 oracle agreement: **66%**
- zero measured regret: **45/50 vs 43/50**

Interpretation: game results are statistically neutral, while the oracle panel points mildly in the favorable direction. Because phase-fix repairs semantics and does not show a measurable strength loss, it is the last-known-good production spine.

## Architecture-independent research — PROMOTED

### Lc0 + Stockfish dual-view training/data infrastructure

Status: **PROMOTED AS RESEARCH INFRASTRUCTURE — ZERO PLAYING-HOT-PATH CHANGE**

Integrated under `research/`:

- exact dataset/toolchain/source locks;
- public Lc0 archive catalog + bounded materializer;
- native Lc0 decode path;
- official Lc0 -> Stockfish `.plain` -> validated `.binpack` bridge;
- lineage-safe dual-view schema;
- fusion tests preventing one Lc0-derived position from becoming multiple fake independent samples;
- Lc0-only / Stockfish-only / A+B / uncertainty / conditional-policy model ablation plan.

Production engine/evaluator is not replaced by an unproven trained model. A learned evaluator must independently earn promotion.

## Lossless speed research

### P0 + P1 dead-organ fast paths

Status: **RETEST_REQUIRED — strong signal, promotion blocked by A/A calibration**

Mechanism:

- skip DSL feature-vector construction when DSL is not ready;
- snapshot dead Risk/DSL/Trace/Policy/Atlas/rule-50 organ readiness;
- avoid trace-key computation when tracing is disabled;
- avoid hot-loop calls into disabled optional systems.

Historical strict replication of P0+P1+P3: about **+3.906% median**, 31/31 faster, exact behavior signatures; direct P3 ablation only about **+0.20%**, so P3 is not worth its complexity rent.

Fresh v2.1 P0+P1: exact nodes/transcripts on three workloads and roughly +5% apparent median speed, but A/A calibration was biased by about +0.66%, so result was invalidated.

Fresh phase-fix P0+P1: exact nodes/transcripts on three workloads and roughly +3% apparent median speed, but the first calibration geometry again failed. A stricter symmetric A-X-A calibration is in progress. No promotion until calibration is clean or fixed-time games independently justify it.

## Configuration research

### Fundamentals Authority 2

Status: **IN PROGRESS**

Authority 1 is rescue-only. Authority 2 additionally funds protected search by reducing genuinely late/stable quiet branches. The configuration is being tested on phase-fix with 100-game equal-time, equal-node, and Stockfish gates. It will not become the production default because the theory is attractive; it must earn the resource trade empirically.

## Narrow search research

### Repeated-history SplitKey

Status: **IN PROGRESS / COMPLEXITY-RENT WATCH**

Mechanism: preserve ordinary board-key TT reuse, but XOR a history namespace only when `pos.has_repeated()` is already true.

Historical panel: 51.25% equal-node / 55% fixed-time in a noisy 40-game screen.

Fresh deep-oracle panel on phase-fix: **exactly neutral** — 0.78 cp mean regret, 70% agreement, 45/50 zero-regret for both candidate and baseline. Fresh 100-game direct/Stockfish gates are in progress. If they are also flat, reject on complexity rent.

### Q-frontier

Status: **RETEST_REQUIRED**

Actual historical patch, not branch label: qsearch normally keeps two captures; Q-frontier allows at most two additional captures only when their crude material ceiling remains close enough to alpha.

Historical result: equal-node identical to control at 46.25%; fixed-time 50% vs control 46.25%, but active/off bench ratio worsened. Materialized isolated and phase-fix+A+B candidates are under fresh 100-game/equal-node/oracle confirmation. Not presumed promoted.

### Margin Skeptic

Status: **RETEST_REQUIRED**

Idea: re-open only a bounded reduced quiet branch that narrowly failed below alpha, using already-paid-for LMR evidence rather than a global reduction change. Historical cumulative implementation depended on helper logic from a larger v5 branch; a minimal isolated reconstruction is being tested. No promotion until it builds and clears all four gates.

## Research rejected from production

### Late proof / Evidence-Lattice cumulative stack

Status: **REJECTED_STRENGTH_AS_IMPLEMENTED**

The broad 19-descendant screen showed severe completed-depth loss despite equal or greater nodes:

- v7 persistent-proof: ~40.6% screen, mean depth ~14.5
- v8 Evidence-Lattice: ~34.4%, mean depth ~12.9
- v8.5 LMR receipt: ~25%
- v8.5a witness hygiene: ~21.9%

These branches accumulated roughly 600-870 source additions across search/movepick/evidence/trace. Do not inherit the cumulative architecture. Individual low-cost signals may be re-derived independently.

### Boundary / rival uncertainty (v2.5)

Status: **REJECTED_RESOURCE_EFFICIENCY**

Historical 40-game panel:

- boundary: 50% equal-node but 46.25% fixed-time and ~23% active-node expansion;
- rival: 41.25% equal-node / 47.5% fixed-time, ~31% active-node expansion.

The implementation bought extra certainty too expensively.

### Threat-pressure family (v2.6)

Status: **REJECTED_RESOURCE_EFFICIENCY**

- Threat-LMR: 47.5% equal-node / 48.75% fixed-time with ~39% active-node expansion;
- Sentinel: 43.75% equal-node / 52.5% fixed-time — favorable one-regime sample without equal-node support;
- combined: 45% / 47.5%.

No production promotion.

### Pressure-intersection / rival widening (v2.8)

Status: **REJECTED_FIXED_TIME**

- rival64: 52.5% equal-node / 46.25% fixed-time, ~3.8% active-node expansion;
- rival128: 55% equal-node / **41.25% fixed-time**, ~11.3% active-node expansion.

Clear example of local search-quality gains that do not repay their wall-clock cost.

### Heavy history cutoff guard (v2.9)

Status: **REJECTED_FIXED_TIME / RETAIN ONLY NARROW SPLITKEY HYPOTHESIS**

Historical cutoff-guard: 56.25% equal-node but only 46.25% fixed-time. The cheap SplitKey fragment is being tested separately; the guard itself is out.

### Plyless / decision-depth line

Status: **REJECTED_STRENGTH_AS_IMPLEMENTED**

Historical equal-node tests failed. Do not transplant the cumulative line into Fundamentals Ultra.

### P3 NNUE 2-remove/1-add specialization

Status: **REJECTED_COMPLEXITY_RENT**

Direct replication added only about +0.20% speed and was inconclusive. P0/P1 contain the stronger generator; P3 stays out.

## Current interpretation

The surviving meta-pattern is stronger than any individual patch:

1. Keep Stockfish's mature substrate and NNUE.
2. Prefer signals already computed by the search over new bookkeeping systems.
3. Prefer deleting dead work over adding speculative intelligence.
4. Require equal-node quality **and** fixed-time efficiency.
5. Keep new training/data capability off the hot path until it proves downstream strength.
6. Never treat branch age/name as evidence that the underlying mechanism was present or strong.

This document will be updated when the remaining P0/P1, Authority 2, SplitKey, Q-frontier, and Margin Skeptic gates resolve.

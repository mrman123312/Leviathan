# Fundamentals Ultra — Evidence-Gated Research Ledger

Authority date: 2026-08-17

## Protected baseline

`leviathan/fundamentals-ultra-v0@0625df7581109e4db15a936b4b0237d9476e2074`

Mechanism: Fundamentals v2.1 + phase-safe pawn semantics + explicitly materialized qsearch frontier (cap4 / 48cp late-ceiling gate).

Evidence at freeze:
- 100-game fixed-time vs current official Stockfish `5062aee519a1ba262d472d8ab139851ced56573e`: 50.0% (7/86/7).
- 100-game equal-node vs Fundamentals v2.1: 50.0% (4/92/4).
- 100-game fixed-time vs Fundamentals v2.1: 50.5% (7/87/6).
- 50-position old deep-oracle panel: 0.54cp mean regret / 78% oracle move agreement vs 1.02cp / 66% for v2.1.

This is a parity launchpad, not proof of superiority.

## Promotion contract

A mechanism does not enter a successor Ultra baseline because of branch age, novelty, one good Elo screen, or one oracle outlier.

Promotion normally requires:
1. source provenance and exact mechanism materialization;
2. correctness / compile / UCI smoke;
3. A/A calibration when timing matters;
4. equal-node or exact-transcript test to separate chess quality from throughput;
5. fixed-time paired games;
6. isolated root-restricted deep-oracle regret on a fresh holdout;
7. direct current-Stockfish confirmation for production strength claims;
8. 1/2/4-thread check for a promoted playing stack;
9. rollback to the protected baseline on regression.

## Strong negatives — do not silently resurrect

### Late proof / Evidence Lattice stack
Status: REJECTED AS COMPETITIVE BASE.
Reason: large source/metadata growth caused severe completed-depth collapse despite similar or greater nodes.
Harvest: provenance-valid evidence and failure-corpus ideas only.

### Plyless / decision-depth line
Status: REJECTED.
Reason: failed equal-node strength / regret tests.

### Boundary uncertainty
Status: REJECTED AS IMPLEMENTED.
Historical screen: ~50% equal-node but 46.25% fixed-time with ~23% active-node expansion.
Lesson: careful re-search can buy information too expensively.

### Rival preservation
Status: REJECTED AS IMPLEMENTED.
Historical screen: 41.25% equal-node / 47.5% fixed-time.

### Threat-pressure / Threat Sentinel
Status: REJECTED AS IMPLEMENTED.
Threat Sentinel's 52.5% fixed-time screen was contradicted by 43.75% equal-node. Threat-LMR remained below parity and expanded nodes heavily.
Lesson: a flashy fixed-time result without pure-search support is not promotion evidence.

### Alpha-beta root committees / portfolios
Status: REJECTED.
Corrected same-corpus, isolated-state 80k-node regret panel:
- monolithic Ultra: 3.30cp / 87.5% oracle move agreement
- portfolio2: 5.60cp / 72.5%
- portfolio3: 3.725cp / 75%
- duel-v2.1: 4.325cp / 70%
- duel-current-Stockfish: 3.50cp / 75%
- committee3: 3.775cp / 77.5%
Lesson: dividing a small alpha-beta budget among similar views loses to mature concentrated search. A useful second view must be orthogonal and cheap.

### Crude Fundamentals buyback retuning
Status: STOPPED — LOW INFORMATION GAIN.
Shared 40-position panel baseline: 1.40cp mean regret.
Examples:
- Recapture 256 -> 128: 2.25cp.
- Passer 320 -> 160: 1.575cp.
- Passer 320 -> 640: 1.675cp.
- All buybacks x1.5: 2.55cp, 0 better / 8 worse.
- Endgame 128 -> 384: no change on panel.
Lesson: current bottleneck is not obvious rescue magnitude tuning.

## Surviving strength signal — qsearch frontier

### Cap5 tail miner
Status: PROVISIONAL STRUCTURAL SIGNAL, NOT YET PROMOTED.

150 fresh positions (50 each after 8/14/22 plies), cap4 and cap5 each 60k nodes. Deep current Stockfish graded only disagreements with independent root-restricted processes.

- disagreements: 33 / 150 (22%)
- cap5 better: 21
- cap4 better: 11
- tie: 1
- mean delta on disagreements: +5.18cp for cap5
- median delta: +5cp
- cap5 rescues >=20cp: 6
- cap5 harms >=20cp: 3
- cap5 rescues >=50cp: 1
- cap5 harms >=50cp: 0

By position horizon:
- 8-ply: 4-3, +1.86cp mean, no >=20cp events.
- 14-ply: 9-6, -0.44cp mean, 1 rescue >=20 / 3 harms >=20.
- 22-ply: 8-2, +16.5cp mean, 5 rescues >=20 / 0 harms >=20.

Interpretation: universal cap5 is not proven. Frontier width appears to interact with position maturity / irreversible tactical state. Active experiments test phase-conditioned and q-depth-conditioned frontiers.

### Preserved adversarial witness
`experiments/regressions/qsearch_frontier_tail.json`

QF-T01 remains a provisional witness until regraded by the hardened isolated oracle. It is failure memory even if cap5 itself is rejected.

## Surviving lossless-speed lane

### P0+P1 dead-organ fast paths
Status: ACTIVE CONFIRMATION.
Historical strict replication of parent composition showed ~+3.9% median speed with exact behavior transcripts and calibrated A/A. P3's incremental effect was only ~+0.2% and did not earn complexity.

Current Ultra-kernel run has already passed:
- compile;
- three normalized exact-behavior gates;
- 100-game equal-node identity stage;
- 100-game fixed-time stage.
Direct current-Stockfish and regret confirmation remain gating evidence.

### Active Fundamentals classification fast path
Status: TESTING.
Caches decision-invariant move classifications/config reads inside the active Fundamentals hot path. Must match exact normalized transcripts before any speed result counts.

## Scaling evidence

Ultra-v0 vs same-thread current Stockfish, 60 games / 150ms per move:
- T1: 49.17% (1/57/2)
- T2: 47.5% (1/55/4)
- T4: 50.0% (3/54/3)

No convincing strength scaling regression from this small sample, but Ultra searched consistently fewer nodes than Stockfish at the same wall time (~7-8% deficit across T1/T2/T4). This is an active throughput problem and motivates lossless-speed work.

## Representation frontier

Current official Stockfish NNUE already includes full-threat features and pair features, but its learned network still terminates in a scalar value output. Therefore "add threat inputs" is not an orthogonal moonshot in 2026.

High-value missing learned objects remain:
- policy / candidate probability;
- search-error / uncertainty prediction;
- selective frontier-admission probability.

### Lc0 policy-head complementarity probe
Status: ACTIVE.
Question: can a one-node independent policy proposal identify deep-oracle moves missed by an 80k Ultra search often enough, and cheaply enough, to justify distilling policy into a small auxiliary head?

If no, kill the route before expensive training.
If yes, next prototype is not full Lc0-in-search; it is a cheap auxiliary policy/search-error head that changes node allocation while preserving Stockfish's scalar NNUE as the value backbone.

## Active structural experiments

- phase-conditioned qsearch frontier (cap4 early, cap5 only in mature states)
- restored qsearch local depth (`qPly`) as a lost selectivity variable
- uncapped evidence-only frontier
- cheap skipped-frontier bound carry
- LMP protected-scope visibility / history-conditioned LMP
- Q-frontier cap5 fresh finalists

Do not hybridize these merely because they are live. Only survivors may compose.

## Research invariant

The current target is not "more Leviathan code." It is more useful information per unit of search cost.

When Stockfish makes an assumption, ask:
1. What information was unavailable or too expensive when the assumption was introduced?
2. Is that information already available now?
3. Can we expose it before the heuristic throws the candidate away?
4. Can we remove the need for the proxy instead of tuning the proxy forever?

No merge without explicit user authorization.

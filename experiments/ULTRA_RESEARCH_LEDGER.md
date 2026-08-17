# Fundamentals Ultra — Evidence-Gated Research Ledger

Authority date: 2026-08-17

## Protected baseline

`leviathan/fundamentals-ultra-v0@0625df7581109e4db15a936b4b0237d9476e2074`

Mechanism: Fundamentals v2.1 + phase-safe pawn semantics + explicitly materialized qsearch frontier (cap4 / 48cp late-ceiling gate).

Evidence at freeze:
- 100-game fixed-time vs current official Stockfish `5062aee519a1ba262d472d8ab139851ced56573e`: 50.0% (7/86/7).
- 100-game equal-node vs Fundamentals v2.1: 50.0% (4/92/4).
- 100-game fixed-time vs Fundamentals v2.1: 50.5% (7/87/6).
- 50-position old deep-oracle panel: 0.54cp mean regret / 78% oracle move agreement vs 1.02cp / 66% for v2.1. This historical oracle result is now provisional because the older grader reused oracle state.

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

## Measurement integrity

`tools/oracle_regret.py` now emits `LV_ISOLATED_ROOT_REGRET_V2` and uses separate oracle processes for root selection, best-move grading, candidate grading, and baseline grading with root-move restriction.

Reason: the earlier grader reused one Stockfish oracle across selection and grading, allowing TT/history state and search order to contaminate nominally independent grades.

Policy:
- game results remain valid;
- old regret numbers are historical/provisional unless reproduced under V2;
- new promotion claims must use the isolated grader or an equivalently strict replacement.

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

### Broad LMP visibility / history exemptions
Status: REJECTED AS A PROMOTION FAMILY.
Fresh 60-game screens vs Ultra-v0:
- scope-visible: 45.83% equal-node, 52.5% fixed-time;
- history4k: 51.67% equal-node, 49.17% fixed-time;
- history16k: 47.5% equal-node, 47.5% fixed-time; isolated-oracle mean improvement only +0.07cp with mixed tails.
Lesson: merely keeping more quiets visible because they are in a hand-written scope or have high history is not reliable evidence. The scope-visible fixed-time flash reproduces the same compute-path/noise failure signature seen in Threat Sentinel.
Harvest: protected quiets need calibrated error evidence, not generic exemptions.

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

Interpretation: universal cap5 is not proven. Frontier width appears to interact with position maturity / irreversible tactical state. Active experiments test phase-conditioned, evidence-conditioned, frontier-debt, and q-depth-conditioned frontiers.

### Evidence96 uncapped frontier
Status: PROVISIONAL; FRESH HARDENED CONFIRMATION ACTIVE.
Original 60-game screen vs Ultra-v0:
- 51.67% equal-node (3/56/1)
- 51.67% fixed-time (7/48/5)
- old-oracle delta was only +0.07cp and is not promotion evidence.
Mechanism: after the first four qsearch captures, additional captures are admitted only when their crude capture ceiling remains at least `alpha + 96`, replacing the hard ordinal frontier with an evidence-conditioned frontier.
Lesson so far: unlike several rejected families, the signal was symmetric across equal-node and fixed-time; magnitude is too small to promote without fresh isolated-oracle replication.

### Preserved adversarial witness
`experiments/regressions/qsearch_frontier_tail.json`

QF-T01 remains a provisional witness until regraded by the hardened isolated oracle. It is failure memory even if cap5 itself is rejected.

## Promoted lossless-speed lane

### P0+P1 dead-organ fast paths
Status: PROMOTED AS LOSSLESS INFRASTRUCTURE, NOT AS A PER-NODE CHESS CLAIM.

Materialized successor branch: `leviathan/fundamentals-ultra-p01-qfrontier@3404fa2ea9b4fe75236e2584be9da3416e646ec2`.
Source delta: `ae6f326512bd418c30c7bf9919f9546866619a57`.

Exact-behavior gates against Ultra-v0 all match:
- default: 3,130,023 nodes and identical normalized transcript hash;
- depth11: 790,195 nodes and identical normalized transcript hash;
- nodes50k: 2,451,456 nodes and identical normalized transcript hash.

30-block ABBA speed replication on Ultra-v0:
- identical 1,767,725-node search behavior and transcript hash in every block;
- candidate faster in 30/30 blocks;
- geometric mean speedup +3.9308%;
- approximate 95% interval +3.4142% to +4.4499%;
- worst block still +1.26%.

Fresh 100-game confirmation:
- equal-node vs Ultra-v0: 50.0% (7/86/7); every opening pair netted exactly 0.5 because search behavior is identical;
- fixed-time vs Ultra-v0: 53.0% (12/82/6), naive +20.9 Elo; pair-bootstrap score interval approximately 49.5%-56.5%, so the Elo magnitude is not treated as proven;
- fixed-time vs current Stockfish: 49.5% (4/91/5).

Causal conclusion: P0+P1 buys more of the same chess per second. The speed claim is strong; the 53% game sample is supporting transfer evidence, not a standalone Elo proof.

### Active Fundamentals classification fast path
Status: TESTING.
Caches decision-invariant move classifications/config reads inside the active Fundamentals hot path. Must match exact normalized transcripts before any speed result counts.

## Scaling evidence

Ultra-v0 vs same-thread current Stockfish, 60 games / 150ms per move:
- T1: 49.17% (1/57/2)
- T2: 47.5% (1/55/4)
- T4: 50.0% (3/54/3)

No convincing strength scaling regression from this small sample, but Ultra-v0 searched consistently fewer nodes than Stockfish at the same wall time (~7-8% deficit across T1/T2/T4).

The promoted P0+P1 lane recovers about half of that measured throughput deficit without changing search decisions. A direct 100k-node Ultra-P01 vs current Stockfish duel is active to separate remaining per-node search quality from throughput.

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

### Search Error Atlas v0
Status: PROVISIONAL MOONSHOT / DATA EXPERIMENT; NO PLAYING-STRENGTH CLAIM.

Hypothesis: Stockfish-style search contains observable pre-error telemetry. If shallow search trajectories can prospectively predict where a 100k-node move has >=15cp or >=30cp deep-oracle regret, a future controller could move verification budget toward likely prediction errors instead of globally widening search.

Screen features intentionally use information obtainable from search trajectory / board state: move stability across 12k/40k/100k nodes, score drift, early top-two gap, depth/seldepth, branching, game ply, material, check state, and position complexity proxies.

The screen uses out-of-fold logistic predictions and compares them against simple one-feature baselines. A pass requires AUC >=0.70, >=0.05 AUC lift over the best simple baseline, >=60% of positive errors concentrated in the top-risk quartile, and at least 2x mean regret in that quartile. A pass earns only a fully fresh prospective holdout; it does not earn integration.

Long-range generator if validated: each expensive heuristic shortcut should either return a proof or leave behind calibrated evidence about its own prediction error. This could unify qsearch frontier, LMP/LMR, null move, and ProbCut under a learned verification-budget controller rather than independent hand-tuned patches.

## Active structural experiments

- phase-conditioned qsearch frontier (clean predeclared holdout plus separate exploratory grid; the grid currently has materializer failures and receives no evidence credit)
- frontier-debt-conditioned fifth capture
- restored qsearch local depth (`qPly`) as a lost selectivity variable
- uncapped evidence-only frontier / Evidence96 hardened confirmation
- cheap skipped-frontier bound carry
- ProbCut paid-evidence near-miss memory
- failed-null tempo-fragility memory
- Q-frontier cap5 fresh finalists
- Lc0 policy complementarity
- Search Error Atlas v0

Do not hybridize these merely because they are live. Only survivors may compose.

## Research invariant

The current target is not "more Leviathan code." It is more useful information per unit of search cost.

When Stockfish makes an assumption, ask:
1. What information was unavailable or too expensive when the assumption was introduced?
2. Is that information already available now?
3. Can we expose it before the heuristic throws the candidate away?
4. Can we remove the need for the proxy instead of tuning the proxy forever?

No merge without explicit user authorization.

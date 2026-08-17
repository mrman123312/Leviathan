# Fundamentals Ultra — Phase 3 Checkpoint

Date: 2026-08-17

This is an additive checkpoint. It preserves Phase 1/2 records rather than rewriting history after later evidence.

## 1. Ultra-P01 shows a positive equal-node signal against current Stockfish

Fresh discovery duel:
- Leviathan: `leviathan/fundamentals-ultra-p01-qfrontier@3404fa2ea9b4fe75236e2584be9da3416e646ec2`
- Stockfish: `5062aee519a1ba262d472d8ab139851ced56573e`
- 1 thread, 64 MB hash, exactly 100,000 nodes per move
- 50 fresh balanced opening pairs / 100 games
- result: **14 wins / 76 draws / 10 losses = 52.0%**
- naive logistic Elo: +13.9
- candidate/opponent mean nodes were effectively identical (~99.4k because some games terminate before the nominal limit on a few moves)

Pair-cluster uncertainty:
- 50 pair scores: five 0.25, thirty-seven 0.50, seven 0.75, one 1.00
- bootstrap 95% score interval approximately **48.5% to 56.0%**

Interpretation:
- this is the first positive direct same-node signal against current Stockfish in the current Ultra line;
- it is qualitatively stronger evidence than a fixed-time-only flash because throughput is removed as an explanation;
- 100 games / 50 pairs are insufficient to claim a proven positive Elo margin.

Action:
- launched a **fresh 400-game / 200-pair equal-node replication** with four independent opening shards, unchanged P01 and Stockfish revisions, same 100k-node budget.
- workflow: `fundamentals-ultra-p01-equal-node-replication.yml`
- promotion/strength claim waits for this replication.

## 2. Evidence96 is rejected under the hardened oracle

Fresh isolated root-restricted 50-position oracle confirmation:
- baseline Ultra-v0 mean regret: 0.72 cp
- Evidence96 mean regret: 0.74 cp
- baseline oracle-move agreement: 72%
- Evidence96 agreement: 66%
- 13/50 root choices changed
- among changed roots: 5 improved, 4 worsened, 4 oracle-equivalent
- total rescue-minus-harm across all 50 positions: -1 cp
- largest rescue +10 cp; largest harm -13 cp

Decision:
- reject Evidence96 as a promotion route;
- do not allow a later lucky game score to overrule the clean per-position oracle result;
- replacing a hard capture-count frontier with one static capture-ceiling threshold remains another scalar proxy, not calibrated error evidence.

## 3. Search Error Atlas v0 learned model failed; raw trajectory topology remains the only surviving fragment

Full 90-position discovery corpus contained only two >=15 cp errors and one >=30 cp error.

Out-of-fold logistic screen at 15 cp:
- AUC: 0.3523
- top-risk quartile captured 0/2 positive errors
- risk concentration failed
- model screen failed

Simple one-feature baselines:
- move-instability AUC: 0.875
- recent score-drift AUC: 0.3068
- inverse top-two-gap AUC: 0.1136

Because only two positives exist, the 0.875 instability AUC is **not** treated as a generalizable model result.

Decision:
- reject Error Atlas v0 as a learned classifier;
- do not lower the regret threshold after observing sparse labels;
- preserve only the discovered search-state topology hypothesis: stable early -> late fracture.

## 4. Late-fracture prospective replication is producing early counterevidence

Predeclared prospective test:
- 150 new positions, split across 18/26/34-ply horizons;
- same 12k -> 40k -> 100k Ultra-P01 trajectory;
- 700k independent Stockfish root grading;
- no threshold changes after corpus generation.

At the first 60/150 completed positions (two of five shards):
- stable: 38 positions, mean regret 0.50 cp, max 18 cp, one >=15 cp error;
- early-flip-then-stable: 12 positions, mean 0.0 cp;
- late-fracture: 6 positions, mean 0.17 cp, max 1 cp, zero >=15 cp errors;
- churn: 4 positions, mean 2.25 cp, max 8 cp.

This is **unfavorable** to the discovery-set late-fracture story so far. It is not a final decision because the predeclared aggregate requires the full 150 positions and validity counts, but no post-hoc rescue or threshold change is permitted.

If the final result fails, archive late fracture as another discovery-set pattern that did not transfer.

## 5. Lc0 policy probe failure was harness-only and has been repaired

The first run completed engine/Lc0/network/corpus setup but the policy scan exited before position 1 because the wrapper passed `--threads=1`; the `policyhead` search mode does not accept that CLI flag.

Classification: experiment execution failure, no chess evidence.

Repair:
- removed unsupported CLI flag;
- added an explicit `uciok` smoke gate;
- pinned the comparison kernel to the published P01 branch;
- reran the exact same predeclared 40-position policy-complementarity experiment.

## 6. Current search-frontier conclusion

The following simple frontier triggers have now failed to become robust general mechanisms:
- fixed extra capture count (cap5);
- game-phase gating;
- scalar alpha/futility frontier debt;
- crude skipped-bound carry;
- static evidence-ceiling uncapping (Evidence96).

This is enough evidence to stop searching this scalar-threshold family.

The unresolved question is no longer how to tune the qsearch cap. It is whether search can identify **prediction-error risk** or **orthogonal rival evidence** strongly enough to redirect a fixed verification budget.

## 7. Active high-information runs

- 400-game fresh equal-node P01 vs current Stockfish replication.
- 150-position late-fracture prospective replication.
- qDepth restored-local-quiescence-depth A/B variants: real builds and games now running.
- ProbCut paid-near-proof memory: repaired candidates now in equal-node games.
- repaired Lc0 policy complementarity probe.

No new scalar qsearch frontier experiments should be launched unless a genuinely new decision variable is identified.

No merge without explicit user authorization.

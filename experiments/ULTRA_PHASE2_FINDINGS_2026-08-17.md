# Fundamentals Ultra — Phase 2 Findings

Date: 2026-08-17

This file is an additive evidence checkpoint. It does not rewrite earlier beliefs retroactively. The older research ledger remains historical provenance; this record captures what later prospective tests changed.

## 1. P0+P1 is the current lossless speed successor

Protected search-quality baseline:
`leviathan/fundamentals-ultra-v0@0625df7581109e4db15a936b4b0237d9476e2074`

Published lossless successor:
`leviathan/fundamentals-ultra-v1-speed@e0d9a11482c9349d0de73081490baf2f7c15101e`

Source mechanism:
`ae6f326512bd418c30c7bf9919f9546866619a57` — dead optional-organ fast paths.

Strong causal evidence:
- normalized search transcript and node count identical across default/depth11/nodes50k gates;
- independent 30-block ABBA replication: candidate faster 30/30 blocks;
- geometric mean speedup +3.9308%; approximate 95% interval +3.4142% to +4.4499%;
- 100-game equal-node A/B vs Ultra-v0 = 50.0%, with identical decisions;
- 100-game fixed-time A/B = 53.0%, treated as supporting transfer evidence rather than a proven Elo magnitude;
- fixed-time vs current Stockfish = 49.5% on that 100-game sample.

## 2. Active Fundamentals cache does not stack with P0+P1

Standalone exact-behavior speed screen initially suggested about +1.54% geometric/mean-scale throughput with calibrated A/A control.

Required A/B/A+B interaction panel then tested:
- P01 alone: +3.0191%, 95% approx +2.8239% to +3.2147%, faster 20/20 blocks;
- active-cache alone: +0.5004%, 95% approx +0.3237% to +0.6775%, faster 19/20;
- P01 + active-cache: +2.7926%, 95% approx +2.5739% to +3.0118%, faster 20/20;
- combo was ~0.2199% slower than P01 alone;
- observed interaction was sub-multiplicative.

Decision: do not integrate the active-cache patch into P01. The standalone signal overlaps P01's saved hot-path work.

Surviving speed idea: a future P2 should share move-classification facts across pruning/SEE/LMR consumers rather than micro-cache within only one helper. Do not launch until higher-value structural runs resolve.

## 3. Universal cap5 and simple game-phase gating are prospectively rejected

Earlier 150-position tail miner showed a suggestive cap5 disagreement signal, especially in a 22-ply bucket. This was never promoted because the sign test/bootstrap were not decisive.

Fresh prospective mixed-horizon holdout with hardened isolated oracle falsified the broad story.

Global cap5:
- baseline mean regret 1.2833cp, agreement 80%;
- candidate mean regret 2.0833cp, agreement 75%;
- 10-ply bucket +1.70cp worse;
- 18-ply bucket +0.75cp worse;
- 26-ply bucket effectively flat (-0.05cp candidate-minus-baseline);
- contained real 26/15/30cp harms alongside rescues.

Predeclared `game_ply >= 16` cap5 gate:
- baseline 1.2833cp;
- candidate 1.5667cp;
- agreement 80% -> 78.33%;
- 10-ply +0.15cp worse;
- 18-ply +0.75cp worse;
- 26-ply essentially flat.

Decision: reject universal cap5 and simple phase gating. Preserve only the fact that rare fifth-capture rescues exist.

## 4. Scalar frontier debt is rejected as calibrated uncertainty

Tested fifth-capture admission from `frontierDebt = max(0, alpha - futilityBase)`.

Debt192:
- 100-game equal-node score 49.0%;
- isolated oracle baseline 1.92cp / 82% agreement / max 63cp;
- candidate 0.42cp / 88% / max 9cp;
- only 5/50 root choices changed;
- one +63cp rescue dominated the result; removing it reduced mean advantage to ~+0.24cp.

Debt128:
- 100-game equal-node exactly 50.0%;
- oracle 1.92cp -> 1.76cp;
- agreement 82% -> 80%;
- rescues included +63/+14cp but harms included -45/-20/-14cp.

Debt64:
- 100-game equal-node 50.5%;
- oracle 1.92cp -> 1.06cp;
- agreement 82% -> 86%;
- max error 63cp -> 24cp;
- changed 10/50 roots;
- largest rescue +63cp, harms -24/-20cp;
- removing the single +63cp rescue flips the mean effect negative (~-0.41cp).

Decision: `alpha - futilityBase` can identify some disaster states but cannot reliably distinguish rescue from harm. Reject it as a general uncertainty signal.

Surviving fragment: rare catastrophic pruning misses exist and are worth predicting with multiple features.

## 5. Simple skipped-bound carry is not robust

Cap4 bound-carry vs clean cap4 control:
- 80-game equal-node 52.5%;
- 80-game fixed-time 50.625%;
- isolated-oracle mean benefit only +0.20cp;
- 2 positions improved, 3 worsened;
- removing the single +19cp rescue makes the mean negative.

Cap3+carry looked slightly better on aggregate but changes both cap and evidence propagation, so causal credit is unresolved; its oracle mean also flips negative after removing its largest rescue.

Decision: do not promote crude skipped-futility bound propagation. Preserving a crude bound is not equivalent to knowing that the skipped move deserves verification.

## 6. Failed-null tempo fragility is currently evidence-insufficient

Thresholds 160 and 224 both made 0/50 root choices differ from control on the fresh hardened-oracle set. Regret/agreement/max-error were exactly identical.

Decision: no chess-quality conclusion. Before any further game testing, measure whether the internal flag activates. If it never activates, the threshold/feature is dead on the sampled distribution; if it activates but has no root effect, suppressing only quiet-skip is too weak.

## 7. Broad LMP visibility remains rejected

Fresh results already recorded in the main ledger are reinforced by Phase-2 interpretation:
- scope-visible: fixed-time flash but poor equal-node;
- history thresholds inconsistent;
- no meaningful hardened-oracle gain.

Decision: protected quiets require calibrated error evidence, not generic scope/history exemptions.

## 8. Current hidden-generator hypothesis

The qsearch/pruning experiments now share a pattern:
- move count can find rescues but also harms;
- game phase does not transfer;
- scalar frontier debt finds catastrophic tails but is poorly discriminative;
- crude bound carry does not solve the discrimination problem;
- broad visibility exemptions alter compute paths without reliable per-node gain.

Therefore the next architectural question is no longer "what threshold should the pruning proxy use?"

It is:

> Can the engine estimate the probability that a pruning/search shortcut is wrong, using multiple pieces of evidence already produced by search, and allocate verification budget to that risk rather than globally widening search?

This is the motivation for Search Error Atlas and, if independently complementary, a cheap policy/search-error auxiliary head.

## 9. Search Error Atlas v0 — early dataset diagnosis

The first 60/90 ordinary balanced positions contain zero >=15cp errors under 100k Ultra-P01 choice vs 700k independent Stockfish grading; worst observed regret so far is 12cp.

Do not lower the predeclared error threshold after seeing this data.

If the full 90-position corpus remains this clean, classify v0 as insufficient error density rather than model failure. The next corpus should mine difficult states using shallow/deep move instability, tight candidate gaps, score drift and/or independent policy disagreement, then evaluate a learned risk model prospectively on a fresh unselected set.

## 10. Dual-NNUE disagreement audit

Current official Stockfish at `5062aee519a1ba262d472d8ab139851ced56573e` exposes one active `EvalFile`; `Search::Worker::evaluate()` evaluates the one active replicated network. There is no already-computed big/small NNUE pair in the hot path.

Decision: do not claim a second-network disagreement sensor is nearly free. It remains a possible selective experiment later, but its runtime/complexity rent currently exceeds trajectory-based risk signals.

## 11. Active high-value tests

- 100-game equal-node Ultra-P01 vs current Stockfish at 100k nodes/move on a fresh 50-opening corpus. This is the pivotal per-node-quality diagnostic.
- Evidence96 fresh confirmation with hardened oracle + equal-node/fixed-time/current-Stockfish gates.
- qDepth local quiescence-depth variants after full signature-cone repair.
- ProbCut paid-near-proof memory after structural materializer repair.
- Lc0 policy-head complementarity under deep oracle.
- Search Error Atlas completion / error-density gate.

## Research rule after Phase 2

Do not return to scalar qsearch-cap tuning unless new evidence identifies a genuinely new decision variable.

The working target is **more useful information per unit of search cost**, not more search everywhere.

No merge without explicit user authorization.

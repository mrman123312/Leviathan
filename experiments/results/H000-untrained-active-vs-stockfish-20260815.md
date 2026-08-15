# H000 — Untrained active Immortal v1 vs Stockfish

Date: 2026-08-15

## Purpose

Immediate smoke-strength comparison requested before training. This is a composite configuration test, not a causal attribution test for any single Leviathan organ.

## Engines

- Leviathan implementation: `c7c50c0e42c484d6b67d81c91c5401bfae8d4008`
- Official Stockfish master: `5062aee519a1ba262d472d8ab139851ced56573e`
- Both built separately as AVX2 binaries on the same GitHub Actions runner.

At match time, official Stockfish master was the same Stockfish parent commit from which Leviathan was forked.

## Match protocol

- 100 games
- 50 opening positions
- every opening played twice with engine colors reversed
- 50 ms per move
- 1 thread each
- 64 MB hash each
- maximum 180 plies
- same GitHub Actions runner

## Leviathan configuration

All implemented live organs were switched on where possible.

Learned organs had no trained artifacts, therefore they were enabled with deliberately neutral models rather than fabricated weights:

- Policy: enabled, neutral LVTP1
- MetaSearch: enabled, neutral LVTM1, authority 1
- Risk: enabled, neutral LVTR1, authority 1
- Atlas: enabled, empty LVTA1

The only non-neutral search changes were conservative non-trained controls:

- specialist tactical/PV LMR buyback, veto 256 reduction units
- authority-1 Search DSL that could only make reductions less aggressive:
  - checks: -192
  - captures: -96
  - PV nodes: -64
  - clamp: [-256, 0]

Authority 1 structurally prevented the DSL from making LMR more aggressive.

## Result

- Leviathan wins: **24**
- Draws: **44**
- Leviathan losses: **32**
- Leviathan score: **46.0%**
- Naive Elo estimate: **-27.85 Elo**
- Pair-clustered approximate 95% Elo interval: **[-64.07, +7.76]**

Pair totals across the 50 reversed-color opening pairs (0..2 points for Leviathan):

- 0.0: 2 pairs
- 0.5: 12 pairs
- 1.0: 28 pairs
- 1.5: 8 pairs
- 2.0: 0 pairs

Pair sign comparison excluding tied pairs: 8 favorable vs 14 unfavorable; two-sided exact sign p ~= 0.286. This match therefore does not establish a statistically significant difference, but the point estimate is negative and supplies no evidence for promotion.

Terminations:

- checkmate: 56
- threefold repetition: 28
- max plies: 11
- insufficient material: 4
- fifty-move rule: 1

## Decision

**REJECT / DO NOT PROMOTE this composite untrained active configuration.**

The current hand-authored LMR buyback + specialist settings did not demonstrate an advantage and had a negative point estimate. The result does not condemn the MetaSearch/Policy/Risk/Atlas hypotheses because those organs were neutral in this match; they had no trained information to contribute.

Next causal tests should isolate the non-trained LMR controls individually and then train the learned organs from search-regret/teacher data before another full-stack comparison.

## Evidence

GitHub Actions run: `31915298364`
Artifact: `leviathan-vs-stockfish-20260815`
Artifact SHA-256: `e7dbc2b058cceac3fb28acf06085af1e94fdcce928e5286f1a7a351073a6d718`

# PROJECT LEVIATHAN — Work Lab Ledger

Authority date: 2026-08-16

Frozen Stockfish parent: `5062aee519a1ba262d472d8ab139851ced56573e`

Frozen active Leviathan reference for late speed/strength studies: `fbccfb6eb5cd335b1ce8fc5c5efad9e36be4e19d`

This ledger separates observed results, causal interpretations, and promotion decisions. A green workflow is not evidence unless its scientific assertions were exit-code enforced.

| ID | Family | Candidate/reference | Evidence | Status | Causal interpretation / surviving fragment |
|---|---|---|---|---|---|
| W000 | Meta-lab | Existing Actions harnesses | `speed-v3-tt-before-eval` printed `FUNCTIONAL DIVERGENCE` while the job concluded success because a failing producer was piped to `tee` without `pipefail`. | HARNESS FAILURE | All scientific pipelines must use `set -euo pipefail`; prior green status alone is insufficient. |
| W001 | Lossless speed | Phase-B optional-hook elision vs `fbccfb6e` | 15 alternating pairs; 3,210,480 identical nodes; median ratio 1.03270. | RETEST | Strong signal, but inconsistent with later same-parent cumulative ratios and hosted-runner variance. |
| W002 | Lossless speed | NNUE 1→1 no-threat path vs `fbccfb6e` | 21 alternating pairs; identical nodes; median 1.00120. | REJECT | Too small for complexity and runner noise. |
| W003 | Lossless speed | MovePicker pawn-history hoist vs `fbccfb6e` | 21 alternating pairs; identical nodes; median 1.00790. | RETEST | Small, plausible fragment; needs same-run factorial confirmation. |
| W004 | Strength | Sparse Uncertainty v3 `sparse-both` | 40 equal-node games at 51.25%; 40 fixed-time games at 52.5%. | RETEST | Interesting direction; sample far too small for promotion. |
| W005 | Lossless speed | Cumulative-1 vs `fbccfb6e` | 21 alternating pairs; identical nodes; median 1.01842. | RETEST | Phase B + pawn hoist underperformed prior Phase-B-only run; noise or negative interaction unresolved. |
| W006 | Lossless speed | NNUE one-add/two-remove capture lane vs `fbccfb6e` | 21 alternating pairs; identical nodes; median 1.00693. | RETEST | Plausible isolated fragment; requires interaction test. |
| W007 | Strength | PV-only Rival Ambiguity Reuse | Fresh 100-game equal-node 22/53/25 (48.5%); fixed-time 20/59/21 (49.5%); 30 disagreement positions gave mean regret advantage -7.3 cp. | REJECT / FRAGMENT | Frozen candidate is not stronger. Retain only the idea that near-singular rivals expose an uncertainty signal; do not retain its current depth-buyback policy. |
| W008 | Lossless speed | Search-level pawn-history hoist | 21 alternating pairs; identical nodes; median 0.99972. | REJECT | Wrong placement or no reusable benefit. |
| W009 | Lossless speed | Cumulative-2 including capture lane vs `fbccfb6e` | 31 alternating pairs; identical nodes; median 1.01025. | RETEST | Lower than cumulative-1; likely negative interaction or noise. |
| W010 | Lossless speed | Pre-expanded NNUE threat hot tiers | 2,048/4,096/8,192 tiers covered 55.03%/71.28%/85.56% but produced ratios 0.90598/0.90355/0.89229. The reported footprint was understated by 2x; actual extra arrays were 4/8/16 MiB plus the slot map. | REJECT | Large cache-footprint regression. The profile and “coverage is not locality” lesson survive; pre-expansion does not. |
| W011 | Diagnostics | Deep-oracle Regret Atlas | Dev mean regret 2.42 cp, agreement 87.5%; holdout mean regret 9.67 cp, agreement 66.7%, but the workflow did not enable Leviathan Fundamentals. | INVALID COMPARISON / KEEP DATASET | The run compared effectively dormant/identical engines and cannot diagnose a Leviathan effect. The position corpus and oracle protocol can be repaired and reused. |
| W012 | Strength | ProbCut near-miss reuse | Four 40+40-game screens; no consistent improvement over control. | REJECT / FRAGMENT | Preserve near-miss events as diagnostic evidence only. |
| W013 | Lossless speed | TT cutoff before missing NNUE eval | Node signature changed 3,210,480 → 2,737,845; hidden by pipeline. | REJECT AS LOSSLESS | May be reconsidered only as a search-changing strength candidate with equal-node tests. |
| W014 | Diagnostics | Exact-position NNUE evaluation reuse | 1,127,801 calls, 1,127,672 unique keys, only 129 repeats. | REJECT CACHE | Exact evaluation memoization has essentially no headroom in this trace. |
| W015 | Decision-depth | qforced fresh holdout | 100 equal-node 22/51/27 (47.5%); fixed-time 16/59/25 (45.5%). | REJECT | Node reduction and 7.8% faster wall time did not preserve strength. Never call this speed improvement. |
| W016 | Meta-lab | Strict same-run speed factorial | Local smoke: exact normalized transcripts at default 3,210,480 nodes, depth-11 1,286,415 nodes, and fixed-node 2,451,202 nodes. One-round A/A failed calibration as expected; 15-round hosted panel pending. | RUNNING | Determines whether the surviving lossless fragments replicate and compose. No speed conclusion is allowed unless A/A passes. |
| W017 | Meta-lab | Historical Actions audit | 90 workflows; most late experiments patch source at runtime; only aggregate nodes were checked in the key speed workflows; experiment registry stops at P012. | REPAIRING | New experiments require immutable refs, normalized per-position behavior transcripts, hard exit propagation, timeouts, pinned actions, and ledger registration. |

## Promotion rule

No candidate becomes a new baseline from one hosted-runner benchmark. Lossless speed promotion requires exact signatures, a valid A/A control, positive paired evidence in two independent runs, and no regression on a game-position timing panel. Search-changing promotion requires equal-node evidence before fixed-time evidence and a sufficiently powered match or justified SPRT.

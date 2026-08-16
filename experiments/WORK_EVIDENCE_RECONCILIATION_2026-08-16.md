# PROJECT LEVIATHAN — Evidence Reconciliation

Authority date: 2026-08-16

This document reconstructs what was actually implemented and measured. It distinguishes source commits, workflow-generated candidates, controls, observations, and causal conclusions. Cross-run hosted-runner medians are never compounded into a promotion claim.

## Baselines and source identity

| Role | Commit | Meaning |
|---|---|---|
| Frozen Stockfish parent / `master` | `5062aee519a1ba262d472d8ab139851ced56573e` | Tree `3b51a6c6d0e5d0fc44a4fde457d270340cb35280`, identical to official Stockfish; frozen bench 2,884,956 nodes. |
| Late active Leviathan reference | `fbccfb6eb5cd335b1ce8fc5c5efad9e36be4e19d` | v2.1.2, 70 commits after the frozen parent. Fundamentals defaults off and authority defaults zero. This is not Stockfish. |
| Late speed research head | `b6160d547886df2e030752de0dddcb80006aceb5` | Chronological workflow/profile history. It is not the cumulative candidate; most later candidates were created by CI rewrites. |
| Frozen cumulative lossless candidate | `9fe94b6c81ccf3b1defe3200d6c08098a8dba1e1` | First source-addressable composition of P0 optional-organ fast paths, P1 cached readiness, P2 MovePicker pawn-entry hoist, and P3 NNUE capture lane. |
| Strict reconciliation lab | `leviathan/lab-v4-reconciliation` | Source-addressable candidates, event-SHA checkouts, normalized search transcripts, A/A calibration, provenance, hard exit propagation, and a permanent ledger. |

The old `speed-v3-profile` commit order must not be read as candidate composition. `speed-v3-lossless-fastpaths` and `speed-v3-nnue-shapes` also contain workflow files rather than their narrated source candidates.

## The missing parent-relative speed chain

The v2.1.2 validation used the same `fbccfb6e` binary with Fundamentals off and on:

| Mode | Median time | Nodes |
|---|---:|---:|
| Fundamentals off | 2,349 ms | 2,884,956 |
| Fundamentals on | 2,719 ms | 3,210,480 |

Workflow/run/job: `.github/workflows/v212-critical-pawn-validation.yml`, `31922568003`, `95104736051`.

The on/off node ratio was 1.112835 and the median off/on wall ratio was 0.866127. This is not a Stockfish-parent comparison: both rows are the integrated fbcc binary, searches differ, and later speed runs used different hosted runners. Multiplying that median by a later cumulative median would yield a synthetic number, not evidence. A same-run frozen-parent / fbcc-off / fbcc-on / candidate ladder is required.

## Reconciled lossless-speed observations

Every historical candidate below matched only the final 3,210,480 aggregate bench nodes. None archived a normalized per-position score/PV/bestmove transcript, so every apparent win remains a retest candidate.

| Candidate | Workflow commit | Run / job / artifact | Historical median | Decision |
|---|---|---|---:|---|
| Phase-B optional-hook elision | `e1ceba365adc8f1aa8942bb5f4cfbe35455e68cd` | `31925806818` / `95113068630` / `9257812041` | +3.2703%, 15/15 faster | RETEST; conflicts with later compositions. |
| NNUE 1→1 no-threat lane | `03dd4dcf49164d2b7b7a3adc9ccfcce247e3dbd2` | `31925947113` / `95113413541` / `9257873064` | +0.1201%, 12/21 faster | REJECT complexity/noise. |
| MovePicker pawn-entry hoist | `f4855ad7dad50dac8b19d8b52a2900ebb2678a0a` | `31925987087` / `95113509068` / `9257879022` | +0.7901%, 18/21 faster | RETEST. |
| Cumulative-1 | `55ad1c9a8961a6e08b24ec38693a940fba7194ad` | `31926264505` / `95114239404` / `9257966900` | +1.8424%, 21/21 faster | RETEST; apparent negative interaction/noise. |
| NNUE one-add/two-remove capture lane | `0c5e442ec404da0c019357409b4d8733a212a0f9` | `31926278393` / `95114274202` / `9257972588` | +0.6925% | RETEST. |
| Search.cpp pawn-history hoist | `56d68e3ad4716a0e143fc09121297e65b71e768f` | `31926414910` / `95114605524` / `9258010007` | −0.0284% median | REJECT tested placement. |
| Cumulative-2 | `c9ac49914fee18aa87c85b236c86768ea121e4a1` | `31926685653` / `95115262145` / `9258103561` | +1.0251%, 29/31 faster | RETEST; direct reference was fbcc, not an immediate stacked reference. |

Strict local smoke testing of frozen commit `9fe94b6c…` matched the active reference at all timing-independent transcript fields:

| Workload | Nodes | Behavior digest |
|---|---:|---|
| Default active bench | 3,210,480 | `cc3b283d6a9d56560b4cee98f6e8037aefbf082f20f9e39f52aede435cb6487e` |
| Depth 11 | 1,286,415 | `abf4e077c6e2aedfb51d14ddb126f2a1bc7e10f4516967f54e3556073ee4defe` |
| Fixed 50k/position bench | 2,451,202 | `d028f6e0f942ec455a3b7c186f3e08cc5b9eb2a5741a8373ddc1e364fc234d5a` |

These are exact-behavior results, not speed results. The first hosted strict run (`31930767544`) stopped at a build-layout harness failure before timing. The repaired run is `31930898695`.

## Strength observations

| Candidate | Evidence | Reconciled interpretation |
|---|---|---|
| ATL-246 PV stitching | 40-game/equal-budget screen deteriorated from 4.7 cp oracle regret and 91.7% agreement for ordinary Stockfish to 58.9 cp and 25%. | DECISIVE REJECT of PV stitching. Does not reject long-horizon compression. |
| Decision Depth / qforced | Fresh 100 equal-node: 22/51/27 (47.5%); 100 fixed-time: 16/59/25 (45.5%). | REJECT. Fewer nodes and faster wall time did not preserve strength. |
| Sparse Uncertainty v3 `sparse-both` | Run `31926006580`, candidate job `95113556775`, control `95113556801`. Candidate: 51.25% equal-node and 52.5% fixed-time; control: 51.25% and 45%, 40 games/mode. | RETEST only. Separate candidate-vs-Stockfish and control-vs-Stockfish jobs do not isolate the mechanism. Direct retest branch: `leviathan/lab-v4-sparse-retest`. |
| PV-only Rival Ambiguity Reuse | Frozen source `1b8ce4ea697fcb222e04904934039b0d83ffba59`; fresh 100 equal-node 22/53/25 and fixed-time 20/59/21; 30 disagreements gave −7.3 cp mean regret advantage. | REJECT implementation. Preserve only the near-singular-rival uncertainty signal. |
| ProbCut near-miss profiles | Four 40+40 screens; no profile improved consistently; all were candidate-vs-Stockfish rather than direct candidate-vs-fbcc. | REJECT mechanisms; retain event stream as diagnostic data. |

RAR screen and holdout configurations differed, and its disagreement oracle ran with Fundamentals disabled. It is useful causal evidence about the RAR patch under that base, but it cannot validate the earlier active-stack narrative.

## Diagnostic and negative results

- Regret Atlas run `31926884151` is not a Leviathan comparison. Its tool configured only Threads/Hash; both engines therefore selected exactly the same move in all 48 positions. The corpus and oracle protocol survive after repair.
- TT-before-NNUE run `31927381123` changed nodes from 3,210,480 to 2,737,845. Python raised `FUNCTIONAL DIVERGENCE`, but the producer was piped to `tee` without `pipefail`, so CI was green. Reject as lossless; reconsider only as a search-changing candidate.
- Exact NNUE evaluation reuse has negligible intra-search headroom: 1,127,801 calls, 1,127,672 unique keys, 129 repeats, maximum count three. Persistent reuse across actual game moves remains unmeasured.
- NNUE threat hot tiers covered 55.03% / 71.28% / 85.56% of profiled accesses but regressed exact-node speed by 9.40% / 9.64% / 10.77%. The profiler understated extra arrays by 2x; actual additions were 4/8/16 MiB plus the slot map. Reject pre-expansion; preserve the cache-footprint lesson.

## Laboratory audit

At reconciliation time the repository had 90 active workflows and 142 recorded runs: 103 success, 27 failure, 11 cancelled, and one stale in-progress run. Only two workflows specified timeouts. Approximately 59 workflows mutated source inline, 14 explicitly checked out mutable branches, 41 installed unpinned Python dependency ranges, and only about eight pinned every external action to a full SHA.

The 266 artifacts total only about 6.51 MiB but expire in November 2026. `experiments/registry.csv` stops at P012 and contains none of the later studies.

Mandatory rules for new evidence:

1. Candidate and reference are immutable commits or content-addressed source trees.
2. Lossless claims require aggregate nodes plus normalized per-position scores, nodes, PVs, and best moves.
3. Every piped scientific assertion runs under `set -euo pipefail`.
4. Environment, compiler, CPU, NNUE, source, binary, opening, option, dependency, and artifact hashes are retained.
5. Equal-node strength comes before fixed-time strength.
6. Hosted-runner timing is a screen; promotion requires replication and a stable-hardware/game-position panel.
7. A green workflow is not evidence unless all scientific assertions were exit-code enforced.
8. Failures are classified as harness, implementation, or scientific failures and written into `WORK_LAB_LEDGER.md`.

## Active evidence gates

- `W016`: strict P0/P1/P2/P3 speed factorial with two controls, component interactions, three exact transcripts, 15 drift-sandwiched rounds, bootstrap intervals, and A/A calibration.
- `W019`: direct Sparse Uncertainty retest: 100 paired equal-node games, 100 paired fixed-time games, and 40 deep-oracle disagreements against the exact same active reference.
- If Sparse survives, the next test is verification-only versus rival-preservation-only versus both.
- If any lossless speed composition survives twice, the next test is a same-run frozen-parent / fbcc-off / fbcc-on / candidate ladder followed by a game-position timing panel.

## Work laboratory results after reconciliation

The strict factorial and replication work froze the actual candidates instead of reconstructing them inside CI:

| Candidate | Frozen commit | Composition | Decision |
|---|---|---|---|
| P01 | `a38620d49cd28d0f1bddd776658dda4b47dd4c96` | Optional-organ fast paths plus cached readiness | PROMOTED lossless speed baseline relative to active `fbccfb6e`. |
| P013 | `e157b9332a0c71263b57b39757b22431e3f7f31c` | P01 plus NNUE capture lane | DROP P3; the extra lane repeatedly reduced total gain. |

On run `31932620602`, P013 replicated at +3.906% median versus active reference with exact transcripts, while its direct P3 increment over P01 was only +0.202% and missed the predeclared magnitude/consistency gate. The subsequent 50-position promotion panel (`31933515456`, job `95131830611`) was stronger evidence: all variants searched exactly 3,001,928 nodes and produced the same hash over every position's nodes, score, bound, depth, seldepth, and full PV. P01 delivered +4.706% median, bootstrap 95% CI [+3.855%, +6.695%], faster in 9/9 rounds. P013 delivered +3.867%, CI [+3.268%, +5.769%]. P01 is therefore the retained stack.

This promotion is explicitly relative to active Leviathan, not the frozen Stockfish parent. W029 supplied two same-run parent-relative fixed-node throughput experiments; behavior-changing parent comparisons are not described as lossless or as strength evidence. Both exact gates passed on a fresh 50-position corpus. Active-reference throughput was 0.94719 and 0.95344 of parent, a reproducible 4.66–5.28% deficit. P01 measured 0.96729 and 0.99776 of parent. Because identical source, binaries, hardware model, runner image, corpus, and options produced parent-relative P01 estimates ranging from 0.22% to 3.27% lower throughput, no single point estimate is scientifically publishable. A longer direct or stable-hardware panel is required. P01 nevertheless beat active reference in every round-matched comparison, with run medians +2.21% and +4.73%.

The direct Sparse Uncertainty retest failed: 47.5% over 100 equal-node games and 48.5% over 100 fixed-time games. A common 240-position oracle ablation also failed to rescue verification-only, softening-only, or their combination. The earlier apparent positive depended on a candidate-specific disagreement stopping condition and does not survive causal control.

Persistence diagnostics replicated substantial next-root computation reuse, but not better decisions. A four-arm 50-position local ablation attributed most warm-start speed to the existing TT, and a capacity census found 512 MiB was 5.63% slower than 64 MiB. A protected persistent forest is therefore rejected unless a future mechanism demonstrates decision-quality gain rather than mere retained work.

Two structural-compression moonshots were forced through small falsifiable tests. Local geometric failure certificates had only 21.05% exact-best agreement even among preserving neighbors; witness-4 refinement increased footprint, reduced coverage, and did not improve precision. Reject that representation. Partial-order quiet-move diamonds were initially more interesting: 46/352 (13.07%) engine-top-8 pairs fully commuted across every legal opponent reply at bounded board-state level. Follow-through reduced this to 7/352 pairs that also retained both deferred moves in the top four behind the same best opponent reply. A charged equal-total-node preload then scored 1/11/2 against a 300k-node forced-move oracle with -1.71 cp mean regret advantage. Reject the explicit hint. Different paths retain different repetition histories, neither move is forced, and Stockfish's TT likely already captures most exact convergence.

The next lossless loop starts from promoted P01 and attacks three independent cost generators. P5 classifies Fundamentals move facts once and reuses them; P6 replaces a function-local non-trivial static with process-wide inline state; P7 outlines dormant optional-organ bodies. Frozen commits are `8af1e6f5`, `8f28d374`, and `e373f88f`. All three match P01's complete normalized transcript on the 3,210,480-node default, 1,286,415-node depth-11, and 2,451,202-node fixed-node workloads. Local wall, pinned triple-corpus, and RUSAGE CPU timing were all invalidated by failed A/A controls. P7 reduced total `.text` by 1,008 bytes but left the three templated search symbols unchanged, so code size is mechanistic context rather than performance evidence. W031 moves all three candidates to separate calibrated 31-round hosted screens; no component can be promoted from that first screen alone.

W031 completed as run `31936437025`. All three candidates again passed every exact transcript gate. P5 was noise: 0.99926 median, CI [0.99597, 1.00379], 14/31 with valid calibration, and its hosted `.text` grew by 1,280 bytes. Reject the fact object; retain only the smaller hypothesis that `ready()` can be hoisted without materializing every move fact. P6 was the sole provisional win at 1.00854, CI [1.00496, 1.01530], 24/31 with valid calibration. P7's nominal 1.00446 result is invalid because its same-binary A/A control was already 1.00332 and the calibration interval excluded 1.0. P6 now requires independent replication and a fresh game-position panel; P7 gets at most one calibrated retry.

Repairing the dormant Regret Atlas comparison exposed a different strength generator. Every one of its 11 shallow-oracle misses already contained the eventual oracle move in the top eight after only 1,000 nodes; the problem was rank oscillation, not candidate admission. Directly voting for an older/stabler PV and a 12k+3k triggered MultiPV verification both worsened regret. The surviving fragment is epistemic rather than prescriptive: recent root-PV churn marked positions with higher shallow regret and greater marginal benefit from substantially more search. On the exploratory 48 positions, an online completed-iteration policy stopped stable roots and continued volatile roots to 100k, using 2,643,909 total nodes. A uniform control used 2,645,337 nodes. Mean deep-oracle regret was 3.77 versus 6.58 cp, but only 4/42/2 decisions differed and the corpus created the hypothesis. W032 therefore predeclares the policy on a new 100-position holdout and adds a shuffled-budget arm to distinguish useful allocation from merely variable allocation.

W032 survived its first fresh holdout. Run `31937191372` matched aggregate compute within 0.061%: adaptive 6,088,678 nodes, uniform 6,092,208, shuffled 6,092,345. Mean deep-oracle regret was 2.89 / 3.57 / 4.64 cp. Adaptive beat uniform by 0.68 cp with bootstrap CI [0.22, 1.23] and 10/88/2 outcomes; it beat the shuffled-budget ablation by 1.75 cp, CI [0.57, 3.21], 12/87/1. This validates the allocation signal, not yet a stronger engine. Stockfish already incorporates best-move instability in clock time management, so the next loop must replicate the oracle result and then beat native time management rather than merely uniform node spending.

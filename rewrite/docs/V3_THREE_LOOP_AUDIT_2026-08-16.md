# Leviathan Rewrite v3 — Three-Loop Optimization & Deep Audit

Date: 2026-08-16
Base control: `489e154b231b0922702892c76ab44efddf26bef5` (`rewrite-v2-bootstrap-strength`)
Branch: `leviathan/rewrite-v3-three-loop-optimization`

## Standard

This pass uses a stricter definition of **better**:

- cleaner/modular code is not a strength result;
- correctness must survive independent differential testing;
- a hot-path rewrite must demonstrate fixed-depth/fixed-time benefit;
- inherited mechanisms remain controls until replacements earn parity or superiority;
- regressions are preserved as evidence instead of rationalized away;
- an attempted fix that materially harms the relevant benchmark is automatically rolled back unless correctness requires it immediately.

## Loop 1 — repair the search-budget contract

### Failure found
`go movetime` silently inherited a hard `max_depth=5`. The previous 150 ms Stockfish games therefore did **not** compare equal search budgets: Leviathan stopped at depth 5 while Stockfish continued searching.

### Rework
- `SearchLimits::max_depth == 0` now means no artificial depth ceiling.
- `go movetime N` searches iteratively until the deadline or hard safety ply.
- explicit `go depth N` remains deterministic.
- deadline polling tightened.
- regression test requires a sparse timed search to exceed depth 5.

### Gate
Normal and Fathom-linked CI passed.

## Loop 2 — TT locality and truthful search identity

### Failures found
- `std::unordered_map` was used as the TT: heap-heavy and cache-unfriendly.
- broad path-sensitive keys destroyed normal transposition reuse.
- the first direct-map replacement policy could overwrite deeper same-key information too aggressively.

### Rework
- fixed-capacity cache-local `TranspositionTable`.
- normal positions use board identity directly.
- history shadow is introduced only when repetition context exists or the current search horizon can cross rule 50.
- TT hits/stores exposed as instrumentation.
- replacement now protects deeper same-key/collision information unless the newcomer earns replacement.
- repetition counts/path-repeat state are tracked per ply instead of rescanning the whole reversible path at every node.

### Gate
Normal and Fathom-linked CI passed.

## Loop 3 — eliminate copy-dominated hot state

### Failures found
- every search child copied a full 64-square `Position`.
- legal-move validation copied the position again per candidate.
- `in_check` repeatedly scanned the board to find kings.
- `key()` repeatedly reconstructed board identity.
- move lists allocated dynamically in hot search.

### Rework
- reversible `UndoState` make/unmake.
- one mutable search position per path.
- search consumes pseudo-legal actions, applies once, validates own-king safety, then undoes.
- cached king squares.
- incremental deterministic position key.
- fixed-capacity `MoveList` with fail-loud capacity exhaustion.
- search ordering no longer heap-allocates move vectors.
- perft diagnostic moved to the same make/unmake substrate.

### Gate
Perft, special-move undo, FEN/key roundtrips, normal CI and Fathom-linked CI passed.

## Deeper audit — defects found and fixed

### En-passant repetition identity
Raw FEN EP metadata incorrectly distinguished positions even when EP was impossible. Position identity now includes EP only when an EP capture is genuinely legal, including king-safety legality.

Regression cases:
- phantom EP == no EP;
- legal EP != no EP;
- pinned/illegal EP == no EP.

### Quiescence promotions
Tactical generation previously omitted quiet promotions. Qsearch tactical actions now include all promotions, including underpromotions.

### Rule-50 terminal precedence
A node at halfmove 100 was returned as a draw before mate could be recognized. Search now preserves checkmate precedence over the rule-50 draw. Dedicated mate-at-100 and non-terminal-draw-at-100 tests were added.

### Repetition hot path
History was rescanned repeatedly. Search now tracks per-ply repetition counts/path-repeat state incrementally, while keeping history-shadow TT identity only where search truth can differ.

### Evaluator hot path
The distilled evaluator repeatedly rescanned all 64 squares for individual features. It now builds one `FeatureSummary` and derives material, PSQT, pawn and king features from that pass.

### Move-list allocation
Move generation in the hot search path used dynamic vectors. Current search uses fixed-capacity inline move storage and aborts on impossible capacity exhaustion rather than silently truncating the tree.

### Perft control path
The diagnostic perft helper continued to use the copy-heavy public legal-move path after search had moved to make/unmake. It now uses the same in-place pseudo-legal → apply → own-check reject → undo sequence, restoring most of the rules-engine performance deficit while retaining exact counts.

## Deep-audit experiments rejected / rolled back

### Precomputed move-score sorting
We tested computing each move-ordering score once into a scored buffer before sorting. In interaction with the concurrent qsearch experiment, fixed-depth throughput regressed. The optimization was not retained; it can be retested independently later with profiler evidence.

### Exact qsearch stalemate stand-pat detection
A semantic edge was identified: at a horizon stalemate, ordinary stand-pat can return a non-draw value because qsearch does not normally enumerate all quiet legal moves.

A brute-force exact repair was implemented by probing full legal-move existence before relevant stand-pat exits. Same-runner ablation showed a major hot-path penalty:
- strong pre-experiment v3 fixed-depth speedup over v2: about **2.58×**;
- combined scored-order + strict-stalemate experiment: about **1.99×**;
- scored-order rolled back while strict-stalemate remained: about **1.74×**.

Per the automatic-rollback rule, the brute-force qsearch stalemate repair was rejected. The edge case remains explicitly open and should be solved with a staged/early-exit legal-existence mechanism rather than taxing every relevant qnode.

## Independent correctness audit

`rewrite/tools/differential_audit.py` uses `python-chess` as an independent rules oracle.

Current panel:
- targeted castling;
- legal and pinned en-passant;
- promotions/underpromotions;
- checkmate/stalemate;
- additional rule-heavy positions;
- 160 deterministic random legal positions, seed 8910;
- exact legal-move-set comparison on the full panel;
- depth-2 perft comparison on targeted positions plus 80 generated positions.

The same CI job also builds with AddressSanitizer + UndefinedBehaviorSanitizer and runs the engine self-test.

The final rollback-surviving search head passed:
- normal/core CI;
- independent differential correctness audit;
- ASan/UBSan self-test;
- Fathom-linked donor CI.

## Final same-runner v2 → v3 evidence

The benchmark compiles frozen v2 and current rollback-surviving v3 on the same GitHub runner and repeats each measurement three times.

Final benchmark-of-record run: GitHub Actions `31987272772`, current ref `198fe433bc0e5b50d803c12eb43e66e6cd748bc5` (benchmark marker commit; search code contains the rollback-surviving architecture).

| Test | Frozen v2 | final v3 | Result |
|---|---:|---:|---:|
| startpos depth 5 wall time | 40.936 ms | 15.698 ms | **2.608× faster** |
| startpos depth 5 nodes | 43,359 | 32,704 | fewer nodes at the same completed depth/result |
| startpos depth 5 best move | Nf3 | Nf3 | same move in all 3 runs |
| `go movetime 150` completed depth | 5 (artificial ceiling) | 7 | **+2 completed plies** |
| `go movetime 150` nodes | 43,359 | 365,312 | **8.425× more searched nodes** |
| startpos perft 5 | 4,865,609 | 4,865,609 | exact correctness |
| perft 5 wall time | 332.049 ms | 360.525 ms | v3 ~8.6% slower on this rules-only diagnostic |

The fixed-depth and timed-search gains are genuine architectural improvements over v2. The perft result is intentionally retained as a non-win: the mailbox/ray-scan rule representation is still not universally superior, even after removing the previous copy-heavy diagnostic path.

## Donor/evaluator deep search

### Berserk
Berserk's NNUE implementation is attractive as an architectural reference: compact king-bucket sparse inputs and a small dense tail. However, the released network repository exposes weights without clear model-license metadata. **Weights were not imported.**

### Viridithas
The separate Viridithas network repository explicitly releases its networks under CC0. Modern Viridithas code is AGPL and therefore remains reference-only under the current Leviathan policy; its modern network also uses a substantially more complex feature system (piece-square, pawn tuples, threat inputs, king buckets).

### Stockfish
The already-registered Stockfish CC0 networks remain the strongest immediate candidate for a frozen evaluator backend without retraining from zero. Integrating a compatible runtime behind `Evaluator` remains a P0 strength task.

### Fathom
The pinned Fathom contract confirms `tb_probe_wdl` intentionally requires rule50=0, so the adapter's nonzero-halfmove rejection is faithful. Fathom's root/DTZ API accepts rule50. The remaining gap is that tablebase knowledge is not yet consumed by the search itself; current integration is diagnostic/adapter-level.

## Remaining architecture gaps discovered by the deep audit

These are **not silently declared fixed**.

### P0 — evaluator capacity
`leviathan-distilled-v1` is still a tiny linear residual teacher model, not a competitive NNUE. The next evaluator milestone is a frozen, license-clean strong network backend followed by controlled replacement experiments.

### P0 — revolutionary architecture is still mostly latent
The current engine is still fundamentally iterative deepening + alpha-beta/PVS/simple LMR. `Evaluation.uncertainty`, `volatility`, TT `evidence/debt`, candidate-set search, proof-budget allocation, and the Evidence Lattice do not yet control enough computation to constitute the intended Leviathan search ontology.

### P1 — mature selectivity gap
Not yet independently rebuilt/validated: null-move pruning, SEE pruning, futility/LMP, ProbCut, singular/check extensions, mature continuation/capture/correction histories, sophisticated LMR and interactions.

### P1 — representation gap
The rules engine is still a 64-square mailbox with ray scans. Bitboards/attack tables or a genuinely better replacement remain a major throughput opportunity. The remaining perft deficit is evidence for this.

### P1 — TT design
The direct-mapped fixed table is much cleaner/faster than `unordered_map`, but a clustered/associative replacement policy still needs A/B testing for collision utility and deep-oracle regret.

### P1 — tablebase search use
Fathom is linked behind a clean interface, but search does not yet exploit WDL/DTZ claims.

### P1 — time/SMP
Only depth/movetime are supported. Real clock management, increments, move-to-go, pondering, hash sizing, threads/SMP/NUMA remain absent.

### P2 — qsearch stalemate horizon semantics
The exact brute-force repair was rejected because it materially harmed search throughput. This remains a known edge requiring a cheaper staged legal-existence mechanism.

### P2 — chess semantics/features
- obvious insufficient-material/dead-position shortcuts are not yet a first-class search terminal;
- FEN strictness remains a contract decision;
- Chess960 is absent;
- PV context reconstruction is display-oriented rather than a first-class search-state object.

## Current conclusion

v3 has earned the claim **“materially better rewrite substrate than v2”** on measured dimensions: honest time budgeting, fixed-depth wall time, search depth under time, transposition reuse, state mutation cost, correctness coverage and auditability.

It has **not** earned the claim “stronger than Stockfish” or “every component is already superior.” The strongest remaining explanation for the Stockfish gap is no longer accidental infrastructure such as the depth-5 ceiling. It is evaluator capacity, missing mature selectivity, the still-mailbox representation, absent SMP/time-control machinery, unused tablebase search knowledge, and the fact that the genuinely novel Evidence-Lattice/candidate/proof-budget architecture is still not the dominant decision process.

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
- regressions are preserved as evidence instead of rationalized away.

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
- perft diagnostic also moved to make/unmake.

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

### Move ordering
The comparator repeatedly recomputed move scores during `std::sort`. Current v3 computes each score once into a fixed scored buffer before sorting.

### Qsearch stalemate semantics
Stand-pat could turn stalemate into a fictitious fail-high. Qsearch now checks legal-move existence before stand-pat fail-high and when no tactical legal action exists.

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

All of these gates passed on the post-loop branch before this report was frozen.

## Same-runner v2 → v3 evidence

The benchmark compiles frozen v2 and current v3 on the same GitHub runner and repeats each measurement three times.

Post-audit measurement before the final scored-order/qsearch micro-fix:

| Test | Frozen v2 | v3 | Result |
|---|---:|---:|---:|
| startpos depth 5 wall time | 40.664 ms | 15.761 ms | **2.58× faster** |
| startpos depth 5 nodes | 43,359 | 32,704 | fewer nodes for same reported result |
| `go movetime 150` completed depth | 5 (artificial ceiling) | 7 | **+2 completed plies** |
| `go movetime 150` nodes | 43,359 | 366,080 | **8.44× more searched nodes** |
| startpos perft 5 | 4,865,609 | 4,865,609 | exact correctness |
| perft 5 wall time | 327.209 ms | 357.416 ms | v3 still ~8.5% slower on this diagnostic path |

The perft result is intentionally retained as a non-win: search is materially faster, but the mailbox rules engine is not yet universally faster than the v2 control.

A final rerun after the scored-order/qsearch micro-fix should be treated as the current benchmark of record.

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
The current engine is still fundamentally iterative deepening + alpha-beta/PVS/LMR. `Evaluation.uncertainty`, `volatility`, TT `evidence/debt`, candidate-set search, proof-budget allocation, and the Evidence Lattice do not yet control enough computation to constitute the intended Leviathan search ontology.

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

### P2 — chess semantics/features
- obvious insufficient-material/dead-position shortcuts are not yet a first-class search terminal;
- FEN strictness remains a contract decision;
- Chess960 is absent;
- PV context reconstruction is display-oriented rather than a first-class search-state object.

## Current conclusion

v3 has now earned the claim **“materially better rewrite substrate than v2”** on several measured dimensions: honest time budgeting, fixed-depth wall time, search depth reached under time, transposition reuse, state mutation cost, correctness coverage and auditability.

It has **not** earned the claim “stronger than Stockfish” or “every component is already superior.” The strongest remaining explanation for the Stockfish gap is no longer the accidental depth-5 cap; it is evaluator capacity plus missing search selectivity/representation and the fact that the genuinely novel proof/candidate architecture is still not the engine's dominant decision process.

# Leviathan Rewrite v0 — Greenfield Architecture

This tree is intentionally independent of Stockfish implementation structure. Stockfish remains an oracle and benchmark opponent, not a source architecture.

## Rewrite invariants

1. Chess rules are isolated from search policy.
2. Search state is explicit; no hidden global heuristic state is required for correctness.
3. Every heuristic must have an interface, an observable effect, and an ablation path.
4. The transposition table stores search claims, not unquestioned truth. `evidence` and `debt` fields are reserved from v0 so uncertainty/provenance can become first-class instead of being bolted onto an old TT contract.
5. Position identity is behind `Position::key()`, allowing future history-sensitive or canonicalized identities without contaminating the rest of the engine.
6. Move generation produces legal actions independently of ordering. Ordering is a policy over actions, not a fused state machine.
7. Evaluation is a structured object rather than forcing every subsystem to consume one scalar.
8. Instrumentation and differential testing are architectural requirements, not afterthoughts.
9. No heuristic exists merely because Stockfish has it. Every imported idea must re-earn itself against null, baseline, and interaction tests.
10. The rewrite must stay runnable at each stage.

## v0 modules

- `chess.*`: rules, FEN, position identity, legal move generation, structured static evaluation.
- `search.*`: iterative deepening, alpha-beta control baseline, quiescence, path-sensitive TT claim identity, repetition-aware search history.
- `main.cpp`: minimal UCI shell plus `perft`, self-tests, and FEN inspection.

The alpha-beta baseline is deliberately simple. It is a control condition, not the final Leviathan ontology.

## Independence gate

A subsystem is independent only if it can compile, test, and produce its own behavior without including or linking Stockfish source. Rewrite v0 meets that rule inside this directory.

## Next architecture steps

1. Differential rule harness: compare legal moves/perft against the frozen Stockfish oracle across generated positions.
2. Replace map-based TT with a fixed-capacity cache and explicit provenance/evidence semantics.
3. Extend `Evaluation`: mean, uncertainty, tactical volatility, provenance.
4. Replace monolithic depth with a search-budget object that allocates proof effort by uncertainty/information value.
5. Implement candidate-set search: moves compete for proof budget rather than inheriting a fixed serial move-ordering ontology.
6. Add Leviathan Evidence Lattice natively at the node/claim level.
7. Add deterministic transcript fingerprints and A/A calibration to every rewrite milestone.
8. Only after the clean engine has independent correctness should Stockfish-derived mechanisms be reintroduced one at a time as controlled experimental imports.

## Definition of success

The rewrite is not complete when the files look different. It is complete when removing the entire Stockfish source tree leaves a functioning, tested Leviathan engine, and any surviving Stockfish idea exists only because independent experiments showed that idea still wins.

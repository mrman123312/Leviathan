# Leviathan Fundamentals v2 — Pre-Training Offensive

## Mission

Improve Leviathan before learned Policy/Meta/Risk models contain useful weights. Attack the Stockfish substrate itself: node cost, search allocation, transposition effectiveness, deterministic tactical coverage, endgame safety, root verification, and build/runtime specialization.

No feature earns a strength claim from architecture alone. Every live mechanism is independently switchable and must survive parent-relative NPS plus paired-game evidence.

## Two-axis design

### Velocity
1. Native ISA + PGO + LTO release profile and richer PGO corpus.
2. Hot-path gating so disabled research organs are nearly free.
3. Quiet-regime overdrive: spend fewer nodes on objectively low-volatility late quiet branches.
4. TT geometry/replacement experiments under identical hash budgets.
5. Benchmark harness records nodes, elapsed time, NPS and binary identity.

### Scope
1. Deterministic regime classifier: forcing / recapture / promotion-race / low-material-zugzwang / rule50-pressure / quiet-stable.
2. Sacrifice/check rescue guard against over-aggressive shallow pruning.
3. Recapture and advanced-pawn LMR protection.
4. Zugzwang-aware null-move protection.
5. Rule50 zeroing-move ordering pressure.
6. Bounded forcing mate prover (proof-sound, incomplete) for checking mating nets.
7. Tactical verification reserve at root (optional, evidence-gated).
8. Existing Syzygy, singular search, ProbCut and alpha-beta remain the trusted verifier.

## Fundamental law

Scope additions must be funded. A deterministic signal may buy depth only when another deterministic signal safely saves depth elsewhere, or when a dedicated proof specialist terminates the search early.

## Authority ladder

- 0: disabled / exact Stockfish behavior.
- 1: rescue-only; can reduce pruning/aggression, never increase it.
- 2: balanced regime allocator; can also increase reductions in quiet-stable branches.
- 3: experimental proof/root tools; separately ablatable.

## Promotion gates

1. compile + UCI;
2. frozen-parent bench check with all fundamentals off;
3. neutral/authority-0 exact node equivalence;
4. NPS panel vs parent build;
5. fixed-node search-efficiency panel;
6. paired STC;
7. SPRT screening;
8. LTC;
9. multi-hardware;
10. only then default-on.

## Pre-training tools

Fundamentals v2 deliberately expands deterministic tooling rather than pretending untrained networks are useful:

- regime signal tracer;
- forcing mate proof search;
- TT geometry laboratory;
- rich PGO corpus builder;
- NPS/perf panel;
- tactical/endgame regression bank;
- paired same-binary ablation harness;
- candidate tournament for bounded deterministic search formulas.

The learned organs remain available, but this generation must stand on its own with them neutral or disabled.

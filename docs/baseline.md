# Project Leviathan — Phase 0 Baseline

## Purpose

Freeze and document the unmodified Stockfish baseline before any playing-strength experiment.

## Baseline repository

- Repository: `mrman123312/Leviathan`
- Branch: `leviathan/phase0-baseline`
- Upstream-equivalent baseline commit: `5062aee519a1ba262d472d8ab139851ced56573e`
- Default branch at fork time: `master`
- Baseline state: no Leviathan search modifications

## Current integration points

- Move ordering: `src/movepick.cpp`
  - `MovePicker::score()` assigns capture, quiet, and evasion ordering scores.
  - Quiet-move scoring currently combines main history, pawn/shared history, continuation histories, checking bonuses, threat escape/entry terms, and low-ply history.
- Main alpha-beta search: `src/search.cpp`
  - The main `MovePicker` is instantiated before the move loop.
  - LMR and pruning logic are downstream of move ordering and will remain untouched in P001.
- NNUE value evaluation: `src/nnue/network.h`, `src/nnue/nnue_architecture.h`
  - Existing NNUE remains the value evaluator for initial Leviathan experiments.
- UCI options: `src/engine.cpp` and `src/ucioption.cpp`
  - Future experimental toggles should be added through the existing option mechanism.

## Build system

Stockfish builds from `src/Makefile`.

Reference commands:

```bash
cd src
make net
make -j2 build ARCH=x86-64-avx2
./stockfish bench
```

The Phase 0 GitHub Actions workflow records the exact runner, compiler, CPU information, commit, network, and bench output.

## Baseline validation requirements

Before any playing-strength patch is evaluated:

1. Baseline workflow must compile successfully.
2. `bench` must complete successfully.
3. UCI startup must complete successfully.
4. The baseline commit, compiler, CPU, network identity, build command, and bench output must be archived as a workflow artifact.
5. Any later functional patch must run Stockfish signature/bench checks before match testing.

## P001 research boundary

The first strength experiment is **policy-assisted move ordering only**.

P001 must not alter:

- late-move reductions,
- pruning thresholds,
- extensions,
- value NNUE,
- time management.

The intended first insertion point is the scoring path in `MovePicker::score()` for quiet moves, with the policy contribution bounded and independently switchable. Only after a positive move-ordering result may policy information be tested downstream in LMR.

## Success criterion

Phase 0 is complete when the baseline workflow produces a reproducible successful build and bench record for this frozen baseline.
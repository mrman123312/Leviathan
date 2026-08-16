# Fundamentals v2.1 — allocator tournament and holdout

Date: 2026-08-15

## Code refinement

v2.1 narrows deterministic search spending before any training:

- recaptures no longer bypass SEE pruning unconditionally; only near-balanced recaptures are rescued;
- sparse-endgame buyback is concentrated on early candidates and forcing moves instead of every move;
- authority-2 quiet overdrive no longer spends generic extra reduction on pawn moves;
- quiet overdrive starts later/deeper and is capped more conservatively.

The frozen Stockfish-parent signature remains 2,884,956 nodes with Fundamentals disabled.

## Development screen

Seven profiles, 10 games each, 25 ms/move, 1 thread, 64 MB hash, paired/reversed colors against Stockfish parent `5062aee519a1ba262d472d8ab139851ced56573e`.

Ranking:

1. rescue — 55% (4W 3D 3L)
2. default — 50% (4W 2D 4L)
3. balanced — 50% (3W 4D 3L)
4. speed — 45% (3W 3D 4L)
5. lean — 45% (3W 3D 4L)
6. scope — 40% (3W 2D 5L)
7. tactical — 35% (2W 3D 5L)

Selected profile: **rescue** (authority 1; no quiet overdrive).

## Fresh holdout test

Five unseen openings, each played twice with colors reversed: 10 games total at 100 ms/move, 1 thread, 64 MB hash.

- Leviathan wins: **3**
- Draws: **5**
- Leviathan losses: **2**
- Score: **55.0%**
- Naive score-equivalent Elo: **+34.9 Elo**

Pair scores:

1. 1.0 / 2.0
2. 1.0 / 2.0
3. 1.0 / 2.0
4. 1.0 / 2.0
5. 1.5 / 2.0

## Interpretation

This is a second positive smoke result on a fresh holdout set. It is still far too small to establish a real Elo gain over Stockfish, and it does not establish that v2.1 is stronger than v2. The development tournament does provide useful causal direction: increasingly aggressive quiet overdrive did not help in this sample, while conservative rescue-only search allocation survived both selection and holdout.

Decision: **KEEP / RETEST. Do not promote as proven stronger yet.**

GitHub Actions run: `31917713762`
Artifact: `fundamentals-v21-tournament-c0d3e652574103902c45dc516787eef8de1017f5`
Artifact SHA-256: `c4caca36757b1eb4852c7a7e3326387bfbf63f9b360e8eab6d3c6f782315284a`

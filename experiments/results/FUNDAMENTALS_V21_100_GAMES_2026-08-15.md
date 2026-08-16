# Fundamentals v2.1 — 100 games vs Stockfish

Date: 2026-08-15

## Engines

- Leviathan source commit: `4e2c76f898bcbb22a8f79a6c8dd1286c2d80e513`
- Stockfish source commit: `5062aee519a1ba262d472d8ab139851ced56573e`

## Configuration

Leviathan Fundamentals v2.1 rescue-only profile:

- Fundamentals: enabled
- Authority: 1
- Forcing Buyback: 384
- Recapture Buyback: 256
- Passer Buyback: 320
- Endgame Buyback: 128
- Quiet Overdrive: 0
- Rule50 Pawn Bonus: 3072
- Zugzwang Guard: enabled
- Sacrifice Rescue: enabled
- Rule50 Pressure: enabled
- Policy / MetaSearch / Risk / Specialist / Atlas / Search DSL: disabled

## Protocol

- 100 games total
- 50 paired openings
- every opening played with reversed colors
- 100 ms per move
- 1 thread per engine
- 64 MB hash per engine
- maximum 220 plies
- 5 parallel GitHub Actions shards, 20 games each
- each shard ran both engines on the same runner
- deterministic balanced opening generation with separate fixed shard seeds
- frozen exact source commits for both engines

## Result

- Leviathan wins: **13**
- Draws: **76**
- Leviathan losses: **11**
- Leviathan score: **51.0%**
- Naive score-equivalent Elo: **+6.95 Elo**
- Approximate pair-clustered 95% Elo interval: **[-22.20, +36.20]**
- Leviathan score as White: **55.0%**
- Leviathan score as Black: **47.0%**

Terminations:

- Checkmate: 24
- Threefold repetition: 66
- Insufficient material: 7
- Fifty-move rule: 1
- Stalemate: 1
- Max plies: 1

Shard scores:

- Shard 0: 55.0% (3W 16D 1L)
- Shard 1: 42.5% (2W 13D 5L)
- Shard 2: 47.5% (2W 15D 3L)
- Shard 3: 57.5% (3W 17D 0L)
- Shard 4: 52.5% (3W 15D 2L)

## Interpretation

This 100-game result is modestly positive but statistically inconclusive. It does not establish that Fundamentals v2.1 is stronger than Stockfish; the approximate confidence interval includes zero by a wide margin. It does reject the idea that the current rescue-only configuration is obviously much weaker under this protocol. The correct decision is **KEEP / RETEST**, not promote as proven stronger.

GitHub Actions run: `31918850106`
Final artifact: `leviathan-v21-vs-stockfish-100games-20260815`
Artifact ID: `9255787015`
Artifact SHA-256: `4e7ba9a8de1b21dae64bde1dbe01584ea3ff71dbecd4ba2b3407e8ad3d4611ef`

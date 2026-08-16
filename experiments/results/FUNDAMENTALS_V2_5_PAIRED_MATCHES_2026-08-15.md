# Fundamentals v2 — five paired matches vs Stockfish

Date: 2026-08-15

## Protocol

- 5 paired matches / 10 games total
- Each of 5 openings played twice with colors reversed
- 100 ms per move
- 1 thread per engine
- 64 MB hash per engine
- Maximum 220 plies
- Same GitHub Actions runner
- Leviathan engine commit tested: `a08b6de44fd6b295c13c775866377b292818ddd4`
- Frozen Stockfish parent: `5062aee519a1ba262d472d8ab139851ced56573e`
- Fundamentals v2 authority 2 defaults enabled
- Policy, MetaSearch, Risk, Specialist, Atlas, and Search DSL disabled

## Result

- Leviathan wins: **2**
- Draws: **7**
- Leviathan losses: **1**
- Leviathan score: **55.0%**
- Naive score-equivalent Elo: approximately **+34.9 Elo**

## Paired matches

1. 1.0 / 2.0 — draw, draw
2. 1.5 / 2.0 — draw, Leviathan win
3. 1.0 / 2.0 — draw, draw
4. 1.0 / 2.0 — Leviathan win, Leviathan loss
5. 1.0 / 2.0 — draw, draw

Game terminations:
- Checkmate: 3
- Threefold repetition: 6
- Insufficient material: 1

## Interpretation

This is a positive smoke result, not evidence of a real Elo gain. Ten games are far too few to distinguish a modest strength change from variance. The configuration therefore **survives the smoke gate** but is **not promoted as stronger than Stockfish** yet.

Next test should use a substantially larger paired sample with the same isolated Fundamentals configuration.

GitHub Actions run: `31917368072`
Artifact: `fundamentals-v2-five-paired-matches`
Artifact SHA-256: `4b7a2f802a64198f9cd29bef7de0aa72873881f1abc2cd56a92ad8c35adfe6bc`

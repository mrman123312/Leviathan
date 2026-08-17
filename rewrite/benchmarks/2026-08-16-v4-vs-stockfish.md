# Leviathan Rewrite v4 vs Stockfish — Diagnostic Match

Date: 2026-08-16 America/Toronto / 2026-08-17 UTC

## Exact revisions

- Leviathan: `11dbc03aa3f4991517542b042861e54250f5acd9`
- Stockfish: `5062aee519a1ba262d472d8ab139851ced56573e`
- Stockfish default NNUE downloaded and validated by its build: `nn-ab28990d4ea3.nnue`
- Runner CPU: AMD EPYC 9V74 80-Core Processor
- Both engines ran on the same GitHub Actions runner.

The Leviathan revision differs from the immediately preceding v4 engine head only by benchmark/test tooling; no playing-engine source change was introduced for this match.

## Match protocol

- 16 games
- 8 deterministic openings, each played twice with colors reversed
- openings: Open Game, Queen's Gambit, English, Reti, Sicilian, King's Indian, French, Caro-Kann
- 50 ms `movetime` per move for each engine
- Stockfish configured to 1 thread and 64 MB hash
- maximum 180 plies after the opening seed
- natural chess outcomes with claimable repetition/50-move draws enabled by the controller
- no evaluation adjudication
- no opening-book continuation beyond each four-ply seed
- PGN, per-game JSONL, match log, summary and exact revisions saved as the workflow artifact

Observed mean wall time per move (mean of game means):

- Leviathan: 51.177 ms
- Stockfish: 49.194 ms

The small timing difference does not explain the match result.

## Result

| Engine | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|
| Leviathan | 0 | 1 | 15 | 0.5 / 16 = 3.125% |
| Stockfish | 15 | 1 | 0 | 15.5 / 16 = 96.875% |

- Every Leviathan loss ended by **checkmate**.
- The lone draw was a **threefold repetition** in the Caro-Kann pair with Leviathan as White.
- No engine errors, UCI failures, illegal moves or crashes occurred.
- A naive logistic conversion of 3.125% score gives approximately **-596.5 Elo** for Leviathan relative to Stockfish, but 16 games at this saturated score are far too few for a reliable Elo estimate. Treat it only as a rough scale indicator.

## Per-game result

| # | Opening | Leviathan color | Result for Leviathan | Termination | Plies after seed |
|---:|---|---|---|---|---:|
| 1 | Open Game | White | Loss | Checkmate | 64 |
| 2 | Open Game | Black | Loss | Checkmate | 71 |
| 3 | Queen's Gambit | White | Loss | Checkmate | 96 |
| 4 | Queen's Gambit | Black | Loss | Checkmate | 47 |
| 5 | English | White | Loss | Checkmate | 96 |
| 6 | English | Black | Loss | Checkmate | 65 |
| 7 | Reti | White | Loss | Checkmate | 90 |
| 8 | Reti | Black | Loss | Checkmate | 53 |
| 9 | Sicilian | White | Loss | Checkmate | 62 |
| 10 | Sicilian | Black | Loss | Checkmate | 81 |
| 11 | King's Indian | White | Loss | Checkmate | 104 |
| 12 | King's Indian | Black | Loss | Checkmate | 55 |
| 13 | French | White | Loss | Checkmate | 76 |
| 14 | French | Black | Loss | Checkmate | 59 |
| 15 | Caro-Kann | White | Draw | Threefold repetition | 45 |
| 16 | Caro-Kann | Black | Loss | Checkmate | 59 |

## Interpretation

This is a useful negative result, not a project failure.

The current greenfield rewrite has proven legal move generation, search stability, timing, draw handling and end-to-end UCI operation, but it is **not remotely Stockfish-strength yet**. The match isolates the magnitude of the remaining strength gap under equal short time per move.

The result supports the existing roadmap priority:

1. integrate a frozen strong Stockfish NNUE evaluator behind Leviathan's evaluator boundary;
2. keep the current rewrite as the search/control shell;
3. rerun exactly this paired match as an A/B control;
4. then add the Lc0/Stockfish dual-view learner only after static and deep-oracle gates pass;
5. use losses from this match as failure/frontier training and search-regret positions rather than tuning blindly.

The next comparison should therefore be:

`current Leviathan` vs `Leviathan + frozen Stockfish NNUE` vs `Stockfish`

under the same openings, time budget and runner. That directly measures how much of this 15-loss gap is evaluator knowledge versus search architecture.

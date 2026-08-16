# Leviathan Rewrite v2 — Stockfish 18 Strength Probe

Date: 2026-08-16

## Configuration

- Leviathan: `leviathan/rewrite-v2-bootstrap-strength` at `316ff5a3f64026db4f9604350b15bd48921ab0d1`
- Opponent: official Stockfish 18 tag, resolved by CI to `cb3d4ee9b47d0c5aae855b12379378ea1439675c`
- Color: Leviathan White
- Limit: 150 ms per move for both engines
- Opening: normal start position, no opening book
- Safety ceiling: 200 plies
- Harness: `rewrite/tools/uci_match.py`

## Result

**Stockfish 18 won 0-1 by checkmate after 72 plies (36 moves).**

PGN:

```pgn
1. Nc3 d5 2. Nh3 Nc6 3. e3 e5 4. Bd3 a6 5. Qh5 Nf6 6. Qg5 Be7
7. Qxg7 Rg8 8. Qh6 Bf8 9. Qh4 Rg4 10. Qxg4 Bxg4 11. f3 Bxh3
12. gxh3 d4 13. exd4 Qxd4 14. Ne4 Nxe4 15. Bxe4 f5 16. Bxf5 Qh4+
17. Kd1 Nd4 18. Be4 Qf2 19. Re1 O-O-O 20. c3 Nxf3 21. Bxf3 Qxf3+
22. Re2 Be7 23. a3 Rg8 24. Rb1 Rg2 25. Kc2 Qxe2 26. Kb3 Qd3
27. Ra1 Rxh2 28. Ka2 Rf2 29. Rb1 b5 30. Ra1 e4 31. Rb1 e3
32. dxe3 Bxa3 33. h4 Qc2 34. Bd2 Rxd2 35. c4 Be7 36. cxb5 Qa4# 0-1
```

## Multi-teacher bootstrap signal

Immediately before the game, the CI pipeline labeled 8 seed positions with Leviathan and Stockfish 18 at 75 ms each.

- positions: 8
- best-move disagreements: 5/8
- mean centipawn disagreement where comparable: 72.88 cp

The tiny seed is a pipeline smoke test, not a statistically representative chess corpus.

## Failure analysis

The game falsified the hypothesis that search scaffolding alone would make the current greenfield engine competitive.

The dominant failure is evaluation quality. Leviathan's current native evaluator is intentionally small and largely material/centralization based. It approved an opening trajectory `Nc3, Nh3, e3, Bd3, Qh5, Qg5, Qxg7` and did not recognize the strategic/tactical danger early enough. At ply 13 Leviathan still evaluated `Qxg7` as +92 cp from its own side, while Stockfish's following move was already associated with a very large advantage for Black. By ply 17 Leviathan finally evaluated itself around -288 cp.

The second failure is effective search throughput/depth. During the game Leviathan generally completed depth 5 at 150 ms. Stockfish commonly reached depth 13-18 under the same wall-clock move limit, with substantially more mature pruning, move ordering, incremental NNUE evaluation and low-level optimization.

## Surviving fragments

- independent legal move generation/UCI remained correct for the complete game
- PVS/aspiration/history/killer/LMR search completed without legality or protocol failure
- mate-distance TT normalization behaved coherently into the mating sequence
- the multi-teacher disagreement pipeline produced useful measurable divergence before the match
- the match harness produced a complete legal PGN and per-ply telemetry

## Next causal priority

Do **not** respond to this loss by stacking more arbitrary pruning constants.

1. Give Leviathan a strong pretrained evaluator behind the existing `Evaluator` interface, preferably by integrating a verified NNUE runtime while keeping the model asset separate from search architecture.
2. Profile node-generation/search overhead and replace the current map-based TT with a fixed-capacity cache.
3. Run old-vs-new equal-time/equal-node A/B before adding further pruning.
4. Turn this game's early divergence positions into failure/frontier training examples and label them with multiple teachers.
5. Only after evaluator/search throughput are competitive should we interpret game outcomes as evidence about Leviathan-native proof/evidence mechanisms.

One game is evidence of concrete failure modes, not an Elo estimate.

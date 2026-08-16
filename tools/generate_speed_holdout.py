#!/usr/bin/env python3
"""Generate a deterministic, parent-selected speed-only position holdout."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import chess.engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--seed", type=int, default=2026081613)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--nodes", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    positions: list[str] = []
    seen: set[str] = set()
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 64})
    try:
        game_index = 0
        while len(positions) < args.count:
            if game_index >= args.count * 4:
                raise SystemExit("could not generate enough unique positions")
            engine.configure({"Clear Hash": None})
            board = chess.Board()
            game = object()
            capture_plies = set(rng.sample(range(10, 31), 2))
            for ply in range(31):
                if board.is_game_over(claim_draw=True):
                    break
                legal_count = board.legal_moves.count()
                infos = engine.analyse(
                    board,
                    chess.engine.Limit(nodes=args.nodes),
                    multipv=min(3, legal_count),
                    game=game,
                )
                rows = infos if isinstance(infos, list) else [infos]
                moves = [row["pv"][0] for row in rows if row.get("pv")]
                if not moves:
                    break
                # Seeded top-three diversification. The engine chooses the candidate
                # set; the RNG prevents every game collapsing into one principal line.
                choice = rng.choices(range(len(moves)), weights=(6, 3, 1)[: len(moves)])[0]
                board.push(moves[choice])
                chronological_ply = ply + 1
                if chronological_ply in capture_plies and not board.is_game_over(claim_draw=True):
                    fen = board.fen(en_passant="fen")
                    if fen not in seen:
                        seen.add(fen)
                        positions.append(fen)
                        if len(positions) == args.count:
                            break
            game_index += 1
    finally:
        engine.quit()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(positions) + "\n", encoding="utf-8")
    print(
        f"generated={len(positions)} games_attempted={game_index} "
        f"seed={args.seed} nodes={args.nodes}"
    )


if __name__ == "__main__":
    main()

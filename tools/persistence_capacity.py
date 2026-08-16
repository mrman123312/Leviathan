#!/usr/bin/env python3
"""Measure warm-TT capacity headroom before inventing a protected frontier."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine


MATE_SCORE = 100000


def load_options(path: Path) -> dict[str, Any]:
    options = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(options, dict):
        raise SystemExit("options JSON must be an object")
    return options


def configure(engine: chess.engine.SimpleEngine, options: dict[str, Any]) -> None:
    unknown = sorted(set(options) - set(engine.options))
    if unknown:
        raise SystemExit(f"engine does not expose options: {unknown}")
    engine.configure(options)


def clear(engine: chess.engine.SimpleEngine) -> None:
    engine.configure({"Clear Hash": None})


def score_cp(info: dict[str, Any], pov: chess.Color) -> int:
    value = info["score"].pov(pov).score(mate_score=MATE_SCORE)
    return int(value if value is not None else 0)


def search(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    game: object,
    root_moves: list[chess.Move] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    info = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        game=game,
        root_moves=root_moves,
    )
    return {
        "move": info["pv"][0].uci(),
        "score_cp": score_cp(info, board.turn),
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "nodes": int(info.get("nodes", 0)),
        "wall_ms": (time.perf_counter_ns() - started) / 1_000_000,
    }


def forced_score(
    engine: chess.engine.SimpleEngine, board: chess.Board, move_uci: str, nodes: int
) -> int:
    clear(engine)
    move = chess.Move.from_uci(move_uci)
    return search(engine, board, nodes, object(), [move])["score_cp"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--path-engine", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--hash-sizes", default="4,16,64,512")
    parser.add_argument("--reference-hash", type=int, default=64)
    parser.add_argument("--path-plies", type=int, default=2)
    parser.add_argument("--path-choice-nodes", type=int, default=12000)
    parser.add_argument("--prefix-nodes", type=int, default=250000)
    parser.add_argument("--target-nodes", type=int, default=50000)
    parser.add_argument("--deep-nodes", type=int, default=200000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sizes = [int(value) for value in args.hash_sizes.split(",")]
    if len(set(sizes)) != len(sizes) or args.reference_hash not in sizes:
        raise SystemExit("hash sizes must be unique and include the reference hash")
    fens = [
        line.strip()
        for line in args.positions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(fens) < args.count:
        raise SystemExit(f"need {args.count} positions, found {len(fens)}")
    base_options = load_options(args.options)
    engines = {
        size: chess.engine.SimpleEngine.popen_uci(args.engine) for size in sizes
    }
    for size, engine in engines.items():
        configure(engine, base_options | {"Hash": size, "Threads": 1})
    path_engine = chess.engine.SimpleEngine.popen_uci(args.path_engine)
    oracle = chess.engine.SimpleEngine.popen_uci(args.oracle)
    configure(path_engine, {"Hash": 64, "Threads": 1})
    configure(oracle, {"Hash": 64, "Threads": 1})

    rows: list[dict[str, Any]] = []
    try:
        for index, fen in enumerate(fens[: args.count]):
            root = chess.Board(fen)
            board = root.copy(stack=False)
            clear(path_engine)
            path_game = object()
            actual_moves = []
            for _ in range(args.path_plies):
                if board.is_game_over(claim_draw=True):
                    break
                result = search(
                    path_engine, board, args.path_choice_nodes, path_game
                )
                move = chess.Move.from_uci(result["move"])
                actual_moves.append(move)
                board.push(move)
            if len(actual_moves) != args.path_plies or board.is_game_over(claim_draw=True):
                continue
            target = board
            order = sizes[index % len(sizes) :] + sizes[: index % len(sizes)]
            results: dict[int, dict[str, Any]] = {}
            for size in order:
                engine = engines[size]
                clear(engine)
                game = object()
                current = root.copy(stack=False)
                prefix = []
                for move in actual_moves:
                    prefix.append(search(engine, current, args.prefix_nodes, game))
                    current.push(move)
                target_result = search(engine, current, args.target_nodes, game)
                target_result["prefix"] = prefix
                results[size] = target_result

            clear(oracle)
            oracle_move = search(oracle, target, args.deep_nodes, object())["move"]
            unique_moves = sorted(
                {oracle_move, *(result["move"] for result in results.values())}
            )
            values = {
                move: forced_score(oracle, target, move, args.deep_nodes)
                for move in unique_moves
            }
            best = max(values.values())
            regrets = {
                size: best - values[result["move"]] for size, result in results.items()
            }
            rows.append(
                {
                    "index": index,
                    "root_fen": fen,
                    "target_fen": target.fen(),
                    "actual_moves": [move.uci() for move in actual_moves],
                    "results": {str(size): result for size, result in results.items()},
                    "oracle_move": oracle_move,
                    "oracle_values_cp": values,
                    "regrets_cp": {str(size): regret for size, regret in regrets.items()},
                }
            )
            print(
                len(rows),
                " ".join(
                    f"{size}M:{results[size]['move']}@{results[size]['depth']}"
                    for size in sizes
                ),
                flush=True,
            )
    finally:
        for engine in [*engines.values(), path_engine, oracle]:
            engine.quit()

    reference = str(args.reference_hash)
    summaries = {}
    for size in sizes:
        key = str(size)
        wall_ratios = [
            row["results"][reference]["wall_ms"] / row["results"][key]["wall_ms"]
            for row in rows
        ]
        regret_deltas = [
            row["regrets_cp"][reference] - row["regrets_cp"][key] for row in rows
        ]
        summaries[key] = {
            "median_speedup_vs_reference": statistics.median(wall_ratios),
            "mean_depth_advantage_vs_reference": statistics.mean(
                row["results"][key]["depth"] - row["results"][reference]["depth"]
                for row in rows
            ),
            "mean_seldepth_advantage_vs_reference": statistics.mean(
                row["results"][key]["seldepth"]
                - row["results"][reference]["seldepth"]
                for row in rows
            ),
            "move_agreement_with_reference": sum(
                row["results"][key]["move"] == row["results"][reference]["move"]
                for row in rows
            )
            / len(rows),
            "mean_regret_advantage_cp": statistics.mean(regret_deltas),
            "wins": sum(delta > 0 for delta in regret_deltas),
            "losses": sum(delta < 0 for delta in regret_deltas),
            "ties": sum(delta == 0 for delta in regret_deltas),
        }
    payload = {
        "schema": "LV_PERSISTENCE_CAPACITY_V1",
        "settings": {
            "hash_sizes_mb": sizes,
            "reference_hash_mb": args.reference_hash,
            "positions": len(rows),
            "path_plies": args.path_plies,
            "path_choice_nodes": args.path_choice_nodes,
            "prefix_nodes": args.prefix_nodes,
            "target_nodes": args.target_nodes,
            "deep_nodes_per_forced_move": args.deep_nodes,
        },
        "summaries": summaries,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"schema": payload["schema"], "settings": payload["settings"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()

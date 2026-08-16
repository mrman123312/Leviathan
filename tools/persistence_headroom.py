#!/usr/bin/env python3
"""Measure existing cross-move search reuse before inventing a persistent forest."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine


MATE = 100000


def load_options(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("options JSON must be an object")
    return data


def configure(engine: chess.engine.SimpleEngine, options: dict[str, Any]) -> None:
    unknown = sorted(set(options) - set(engine.options))
    if unknown:
        raise ValueError(f"engine does not expose options: {unknown}")
    engine.configure(options)


def clear_hash(engine: chess.engine.SimpleEngine) -> None:
    if "Clear Hash" in engine.options:
        engine.configure({"Clear Hash": None})


def clear_tt_only(engine: chess.engine.SimpleEngine) -> None:
    if "Clear TT Only" not in engine.options:
        raise ValueError("diagnostic engine does not expose Clear TT Only")
    engine.configure({"Clear TT Only": None})


def cp(info: dict[str, Any], pov: chess.Color) -> int:
    value = info["score"].pov(pov).score(mate_score=MATE)
    return int(value if value is not None else 0)


def search(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    game: object,
    root_moves: list[chess.Move] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    info = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        game=game,
        root_moves=root_moves,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    return {
        "move": info["pv"][0].uci(),
        "score_cp": cp(info, board.turn),
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "nodes": int(info.get("nodes", 0)),
        "engine_time_ms": float(info.get("time", 0.0)) * 1000.0,
        "wall_ms": wall_ms,
    }


def forced_score(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move_uci: str,
    nodes: int,
) -> int:
    clear_hash(engine)
    return search(engine, board, nodes, object(), [chess.Move.from_uci(move_uci)])["score_cp"]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--path-engine", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--path-plies", type=int, default=2)
    parser.add_argument("--path-nodes", type=int, default=12000)
    parser.add_argument("--target-nodes", type=int, default=24000)
    parser.add_argument("--deep-nodes", type=int, default=200000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fens = [line.strip() for line in args.positions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(fens) < args.count:
        raise SystemExit(f"need {args.count} positions, found {len(fens)}")

    warm = chess.engine.SimpleEngine.popen_uci(args.engine)
    history = chess.engine.SimpleEngine.popen_uci(args.engine)
    cold = chess.engine.SimpleEngine.popen_uci(args.engine)
    path_engine = chess.engine.SimpleEngine.popen_uci(args.path_engine)
    oracle = chess.engine.SimpleEngine.popen_uci(args.oracle)
    options = load_options(args.options)
    configure(warm, options)
    configure(history, options)
    configure(cold, options)
    configure(path_engine, {"Threads": 1, "Hash": 64})
    configure(oracle, {"Threads": 1, "Hash": 64})

    rows = []
    try:
        for index, fen in enumerate(fens[: args.count]):
            root = chess.Board(fen)
            path_board = root.copy()
            clear_hash(path_engine)
            path_game = object()
            actual_moves = []
            for _ in range(args.path_plies):
                if path_board.is_game_over(claim_draw=True):
                    break
                result = search(path_engine, path_board, args.path_nodes, path_game)
                move = chess.Move.from_uci(result["move"])
                actual_moves.append(move)
                path_board.push(move)
            if len(actual_moves) != args.path_plies or path_board.is_game_over(claim_draw=True):
                continue
            target = path_board

            def warm_target() -> dict[str, Any]:
                clear_hash(warm)
                game = object()
                board = root.copy()
                prefix = []
                for move in actual_moves:
                    prefix.append(search(warm, board, args.target_nodes, game))
                    board.push(move)
                result = search(warm, board, args.target_nodes, game)
                result["prefix"] = prefix
                return result

            def cold_target() -> dict[str, Any]:
                clear_hash(cold)
                return search(cold, target, args.target_nodes, object())

            def history_target() -> dict[str, Any]:
                clear_hash(history)
                game = object()
                board = root.copy()
                prefix = []
                for move in actual_moves:
                    prefix.append(search(history, board, args.target_nodes, game))
                    board.push(move)
                clear_tt_only(history)
                result = search(history, board, args.target_nodes, game)
                result["prefix"] = prefix
                return result

            runners = [
                ("warm", warm_target),
                ("history", history_target),
                ("cold", cold_target),
            ]
            offset = index % len(runners)
            results = {}
            for name, runner in runners[offset:] + runners[:offset]:
                results[name] = runner()
            warm_result = results["warm"]
            history_result = results["history"]
            cold_result = results["cold"]

            clear_hash(oracle)
            oracle_game = object()
            oracle_result = search(oracle, target, args.deep_nodes, oracle_game)
            moves = sorted(
                {
                    warm_result["move"],
                    history_result["move"],
                    cold_result["move"],
                    oracle_result["move"],
                }
            )
            values = {
                move: forced_score(oracle, target, move, args.deep_nodes) for move in moves
            }
            best = max(values.values())
            warm_regret = best - values[warm_result["move"]]
            history_regret = best - values[history_result["move"]]
            cold_regret = best - values[cold_result["move"]]
            best_regret = min(warm_regret, history_regret, cold_regret)
            winners = [
                name
                for name, regret in {
                    "warm": warm_regret,
                    "history": history_regret,
                    "cold": cold_regret,
                }.items()
                if regret == best_regret
            ]
            row = {
                "index": index,
                "root_fen": fen,
                "actual_moves": [move.uci() for move in actual_moves],
                "target_fen": target.fen(),
                "warm": warm_result,
                "history": history_result,
                "cold": cold_result,
                "oracle_move": oracle_result["move"],
                "oracle_values_cp": values,
                "warm_regret_cp": warm_regret,
                "history_regret_cp": history_regret,
                "cold_regret_cp": cold_regret,
                "warm_regret_advantage_cp": cold_regret - warm_regret,
                "history_regret_advantage_cp": cold_regret - history_regret,
                "tt_regret_advantage_cp": history_regret - warm_regret,
                "winner": winners[0] if len(winners) == 1 else "tie",
            }
            rows.append(row)
            print(
                len(rows),
                row["winner"],
                row["warm_regret_advantage_cp"],
                warm_result["move"],
                history_result["move"],
                cold_result["move"],
                f"depth={warm_result['depth']}/{history_result['depth']}/{cold_result['depth']}",
                flush=True,
            )
    finally:
        for engine in (warm, history, cold, path_engine, oracle):
            engine.quit()

    def pairwise_counts(left: str, right: str) -> dict[str, int]:
        left_key = f"{left}_regret_cp"
        right_key = f"{right}_regret_cp"
        return {
            f"{left}_wins": sum(row[left_key] < row[right_key] for row in rows),
            f"{right}_wins": sum(row[right_key] < row[left_key] for row in rows),
            "ties": sum(row[left_key] == row[right_key] for row in rows),
        }

    payload = {
        "schema": "LV_PERSISTENCE_HEADROOM_V2",
        "positions": len(rows),
        "settings": {
            "path_plies": args.path_plies,
            "path_nodes": args.path_nodes,
            "target_nodes": args.target_nodes,
            "deep_nodes_per_search": args.deep_nodes,
        },
        "move_agreement_rate": sum(row["warm"]["move"] == row["cold"]["move"] for row in rows)
        / len(rows),
        "warm_history_move_agreement_rate": sum(
            row["warm"]["move"] == row["history"]["move"] for row in rows
        )
        / len(rows),
        "history_cold_move_agreement_rate": sum(
            row["history"]["move"] == row["cold"]["move"] for row in rows
        )
        / len(rows),
        "warm_wins": sum(row["winner"] == "warm" for row in rows),
        "history_wins": sum(row["winner"] == "history" for row in rows),
        "cold_wins": sum(row["winner"] == "cold" for row in rows),
        "ties": sum(row["winner"] == "tie" for row in rows),
        "pairwise": {
            "warm_vs_cold": pairwise_counts("warm", "cold"),
            "history_vs_cold": pairwise_counts("history", "cold"),
            "warm_vs_history": pairwise_counts("warm", "history"),
        },
        "mean_warm_regret_advantage_cp": mean(rows, "warm_regret_advantage_cp"),
        "mean_history_regret_advantage_cp": mean(rows, "history_regret_advantage_cp"),
        "mean_tt_regret_advantage_cp": mean(rows, "tt_regret_advantage_cp"),
        "median_warm_regret_advantage_cp": statistics.median(
            row["warm_regret_advantage_cp"] for row in rows
        ),
        "mean_depth_advantage": statistics.mean(
            row["warm"]["depth"] - row["cold"]["depth"] for row in rows
        ),
        "mean_seldepth_advantage": statistics.mean(
            row["warm"]["seldepth"] - row["cold"]["seldepth"] for row in rows
        ),
        "mean_history_depth_advantage": statistics.mean(
            row["history"]["depth"] - row["cold"]["depth"] for row in rows
        ),
        "mean_history_seldepth_advantage": statistics.mean(
            row["history"]["seldepth"] - row["cold"]["seldepth"] for row in rows
        ),
        "median_wall_speedup": statistics.median(
            row["cold"]["wall_ms"] / row["warm"]["wall_ms"] for row in rows
        ),
        "median_history_wall_speedup": statistics.median(
            row["cold"]["wall_ms"] / row["history"]["wall_ms"] for row in rows
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Counterfactual test of root-volatility allocation against native Stockfish TM.

The external adaptive arm is the W032 mechanism.  Its real comparator is not
only a uniform node budget: Stockfish already allocates more time when the root
best move changes.  This harness enables Stockfish's deterministic ``nodestime``
mode, calibrates a native clock allocation to the adaptive arm's aggregate node
count *before* oracle scoring, and then compares both decisions with one deep
oracle.  The calibration therefore tests allocation rather than wall-clock
noise and does not leak oracle outcomes into budget selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import chess
import chess.engine

from volatility_allocator import (
    adaptive_search,
    comparison,
    configure,
    fixed_search,
    load_options,
    oracle_scores,
    reset,
    summarize,
)


def internal_adaptive_search(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    cap_nodes: int,
) -> dict[str, Any]:
    result = fixed_search(engine, board, cap_nodes)
    return {
        **result,
        "trigger_nodes": None,
        "volatile": result["nodes"] >= cap_nodes,
        "last_moves": [],
    }


def native_tm_search(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    clock_nodes: int,
    remaining_moves: int,
) -> dict[str, Any]:
    # python-chess clock fields are seconds. With nodestime=1, each UCI
    # millisecond becomes one deterministic node of available game clock.
    clock_seconds = clock_nodes / 1000.0
    result = engine.play(
        board,
        chess.engine.Limit(
            white_clock=clock_seconds,
            black_clock=clock_seconds,
            remaining_moves=remaining_moves,
        ),
        game=reset(engine),
        info=chess.engine.INFO_ALL,
    )
    if result.move is None:
        raise SystemExit("native time manager emitted no move")
    return {
        "move": result.move,
        "nodes": int(result.info.get("nodes", 0)),
        "depth": int(result.info.get("depth", 0)),
        "seldepth": int(result.info.get("seldepth", 0)),
    }


def native_corpus(
    engine: chess.engine.SimpleEngine,
    boards: list[chess.Board],
    clock_nodes: int,
    remaining_moves: int,
) -> list[dict[str, Any]]:
    return [
        native_tm_search(engine, board, clock_nodes, remaining_moves)
        for board in boards
    ]


def calibrate_native_clock(
    engine: chess.engine.SimpleEngine,
    boards: list[chess.Board],
    target_nodes: int,
    low_clock_nodes: int,
    high_clock_nodes: int,
    remaining_moves: int,
    max_gap: float,
    iterations: int,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    cache: dict[int, list[dict[str, Any]]] = {}

    def measure(clock_nodes: int) -> tuple[int, list[dict[str, Any]]]:
        if clock_nodes not in cache:
            cache[clock_nodes] = native_corpus(
                engine, boards, clock_nodes, remaining_moves
            )
        rows = cache[clock_nodes]
        return sum(row["nodes"] for row in rows), rows

    low_total, _ = measure(low_clock_nodes)
    high_total, _ = measure(high_clock_nodes)
    if low_total >= high_total:
        raise SystemExit("native TM calibration is not monotonic at the bracket endpoints")
    for clock, total in ((low_clock_nodes, low_total), (high_clock_nodes, high_total)):
        if abs(total / target_nodes - 1.0) <= max_gap:
            rows = cache[clock]
            calibration = [
                {
                    "clock_nodes": measured_clock,
                    "total_nodes": sum(row["nodes"] for row in measured_rows),
                    "ratio_to_target": sum(row["nodes"] for row in measured_rows)
                    / target_nodes,
                }
                for measured_clock, measured_rows in sorted(cache.items())
            ]
            return clock, rows, calibration
    if not low_total <= target_nodes <= high_total:
        raise SystemExit(
            "native TM bracket does not contain target: "
            f"low={low_total} target={target_nodes} high={high_total}"
        )

    low_clock, high_clock = low_clock_nodes, high_clock_nodes
    for _ in range(iterations):
        closest_clock = min(cache, key=lambda value: abs(sum(r["nodes"] for r in cache[value]) - target_nodes))
        closest_rows = cache[closest_clock]
        closest_total = sum(row["nodes"] for row in closest_rows)
        if abs(closest_total / target_nodes - 1.0) <= max_gap:
            break

        estimate = round(
            low_clock
            + (target_nodes - low_total)
            * (high_clock - low_clock)
            / (high_total - low_total)
        )
        estimate = max(low_clock + 1, min(high_clock - 1, estimate))
        if estimate in cache:
            # Iteration-depth granularity can flatten nearby clock values.
            estimate = (low_clock + high_clock) // 2
        if estimate in cache or estimate <= low_clock or estimate >= high_clock:
            break

        total, _ = measure(estimate)
        if total < target_nodes:
            low_clock, low_total = estimate, total
        else:
            high_clock, high_total = estimate, total

    selected_clock = min(
        cache,
        key=lambda value: abs(sum(row["nodes"] for row in cache[value]) - target_nodes),
    )
    selected_rows = cache[selected_clock]
    calibration = [
        {
            "clock_nodes": clock,
            "total_nodes": sum(row["nodes"] for row in rows),
            "ratio_to_target": sum(row["nodes"] for row in rows) / target_nodes,
        }
        for clock, rows in sorted(cache.items())
    ]
    return selected_clock, selected_rows, calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument(
        "--adaptive-control", choices=("external", "internal"), default="external"
    )
    parser.add_argument("--probe-nodes", type=int, default=15000)
    parser.add_argument("--cap-nodes", type=int, default=100000)
    parser.add_argument("--history-depths", type=int, default=5)
    parser.add_argument("--native-low-clock-nodes", type=int, default=800000)
    parser.add_argument("--native-high-clock-nodes", type=int, default=5000000)
    parser.add_argument("--native-remaining-moves", type=int, default=50)
    parser.add_argument("--calibration-iterations", type=int, default=5)
    parser.add_argument("--oracle-nodes", type=int, default=500000)
    parser.add_argument("--forced-nodes", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=2026081621)
    parser.add_argument("--max-node-gap", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fens = [
        line.strip()
        for line in args.positions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_positions > 0:
        fens = fens[: args.max_positions]
    if not fens or len(set(fens)) != len(fens):
        raise SystemExit("position corpus must be non-empty and unique")
    boards = [chess.Board(fen) for fen in fens]

    options = {"Threads": 1, "Hash": 64} | load_options(args.options)
    adaptive_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    adaptive_replica_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    uniform_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    shuffled_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    native_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    oracle = chess.engine.SimpleEngine.popen_uci(args.oracle)
    adaptive_options = options
    if args.adaptive_control == "internal":
        adaptive_options = options | {
            "Leviathan Volatility Allocation": True,
            "Leviathan Volatility Probe Nodes": args.probe_nodes,
            "Leviathan Volatility Stable Depths": args.history_depths,
        }
    for engine in (adaptive_engine, adaptive_replica_engine):
        configure(engine, adaptive_options)
    for engine in (uniform_engine, shuffled_engine):
        configure(engine, options)
    configure(native_engine, options | {"nodestime": 1})
    configure(oracle, {"Threads": 1, "Hash": 64})

    try:
        if args.adaptive_control == "internal":
            adaptive = [
                internal_adaptive_search(adaptive_engine, board, args.cap_nodes)
                for board in boards
            ]
        else:
            adaptive = [
                adaptive_search(
                    adaptive_engine,
                    board,
                    args.probe_nodes,
                    args.cap_nodes,
                    args.history_depths,
                )
                for board in boards
            ]
        adaptive_total = sum(row["nodes"] for row in adaptive)
        if args.adaptive_control == "internal":
            adaptive_replica = [
                internal_adaptive_search(adaptive_replica_engine, board, args.cap_nodes)
                for board in boards
            ]
        else:
            adaptive_replica = [
                adaptive_search(
                    adaptive_replica_engine,
                    board,
                    args.probe_nodes,
                    args.cap_nodes,
                    args.history_depths,
                )
                for board in boards
            ]
        adaptive_replica_total = sum(row["nodes"] for row in adaptive_replica)
        allocation_deltas = [
            abs(left["nodes"] - right["nodes"]) / max(1, left["nodes"])
            for left, right in zip(adaptive, adaptive_replica)
        ]
        adaptive_reproducibility = {
            "moves_equal": all(
                left["move"] == right["move"]
                for left, right in zip(adaptive, adaptive_replica)
            ),
            "volatile_flags_equal": all(
                left["volatile"] == right["volatile"]
                for left, right in zip(adaptive, adaptive_replica)
            ),
            "root_histories_equal": all(
                left["last_moves"] == right["last_moves"]
                for left, right in zip(adaptive, adaptive_replica)
            ),
            "primary_total_nodes": adaptive_total,
            "replica_total_nodes": adaptive_replica_total,
            "replica_ratio": adaptive_replica_total / adaptive_total,
            "max_per_position_node_delta_ratio": max(allocation_deltas, default=0.0),
        }
        adaptive_reproducibility["valid"] = (
            adaptive_reproducibility["moves_equal"]
            and adaptive_reproducibility["volatile_flags_equal"]
            and adaptive_reproducibility["root_histories_equal"]
            and abs(adaptive_reproducibility["replica_ratio"] - 1.0) <= args.max_node_gap
        )

        uniform_budget = max(1, round(adaptive_total / len(boards)))
        uniform = [fixed_search(uniform_engine, board, uniform_budget) for board in boards]

        shuffled_budgets = [row["nodes"] for row in adaptive]
        random.Random(args.seed).shuffle(shuffled_budgets)
        shuffled = [
            fixed_search(shuffled_engine, board, budget)
            for board, budget in zip(boards, shuffled_budgets)
        ]

        native_clock, native, calibration = calibrate_native_clock(
            native_engine,
            boards,
            adaptive_total,
            args.native_low_clock_nodes,
            args.native_high_clock_nodes,
            args.native_remaining_moves,
            args.max_node_gap,
            args.calibration_iterations,
        )

        rows = []
        regrets: dict[str, list[int]] = {
            "adaptive": [],
            "uniform": [],
            "shuffled": [],
            "native_tm": [],
        }
        for index, (fen, board, arow, urow, srow, nrow) in enumerate(
            zip(fens, boards, adaptive, uniform, shuffled, native), start=1
        ):
            moves = {arow["move"], urow["move"], srow["move"], nrow["move"]}
            best, scores, oracle_best = oracle_scores(
                oracle, board, moves, args.oracle_nodes, args.forced_nodes
            )
            row_regrets = {
                "adaptive": best - scores[arow["move"]],
                "uniform": best - scores[urow["move"]],
                "shuffled": best - scores[srow["move"]],
                "native_tm": best - scores[nrow["move"]],
            }
            for name, value in row_regrets.items():
                regrets[name].append(value)
            rows.append(
                {
                    "index": index,
                    "fen": fen,
                    "volatile": arow["volatile"],
                    "last_moves": arow["last_moves"],
                    "adaptive_nodes": arow["nodes"],
                    "uniform_nodes": urow["nodes"],
                    "shuffled_nodes": srow["nodes"],
                    "native_tm_nodes": nrow["nodes"],
                    "adaptive_move": arow["move"].uci(),
                    "uniform_move": urow["move"].uci(),
                    "shuffled_move": srow["move"].uci(),
                    "native_tm_move": nrow["move"].uci(),
                    "oracle_best": oracle_best,
                    "oracle_best_cp": best,
                    "regret_cp": row_regrets,
                }
            )
    finally:
        for engine in (
            adaptive_engine,
            adaptive_replica_engine,
            uniform_engine,
            shuffled_engine,
            native_engine,
            oracle,
        ):
            engine.quit()

    totals = {
        name: sum(row[f"{name}_nodes"] for row in rows)
        for name in ("adaptive", "uniform", "shuffled", "native_tm")
    }
    ratios = {name: total / totals["adaptive"] for name, total in totals.items()}
    compute_valid = all(abs(ratio - 1.0) <= args.max_node_gap for ratio in ratios.values())
    adaptive_vs_native = comparison(
        regrets["native_tm"], regrets["adaptive"], args.seed + 1
    )
    native_vs_uniform = comparison(
        regrets["uniform"], regrets["native_tm"], args.seed + 2
    )
    adaptive_vs_uniform = comparison(
        regrets["uniform"], regrets["adaptive"], args.seed + 3
    )
    adaptive_vs_shuffled = comparison(
        regrets["shuffled"], regrets["adaptive"], args.seed + 4
    )

    native_ci = adaptive_vs_native["bootstrap_mean_95pct_ci"]
    uniform_ci = adaptive_vs_uniform["bootstrap_mean_95pct_ci"]
    shuffled_ci = adaptive_vs_shuffled["bootstrap_mean_95pct_ci"]
    if not adaptive_reproducibility["valid"]:
        status = "INVALID_ADAPTIVE_REPRODUCIBILITY"
    elif not compute_valid:
        status = "INVALID_COMPUTE_MATCH"
    elif (
        native_ci[0] > 0
        and uniform_ci[0] > 0
        and shuffled_ci[0] > 0
        and adaptive_vs_native["wins"] >= adaptive_vs_native["losses"]
        and adaptive_vs_uniform["wins"] >= adaptive_vs_uniform["losses"]
        and adaptive_vs_shuffled["wins"] >= adaptive_vs_shuffled["losses"]
    ):
        status = "ADAPTIVE_NOVELTY_PROVISIONAL"
    elif native_ci[1] < 0:
        status = "NATIVE_TM_SUPERIOR"
    else:
        status = "NATIVE_TM_NOT_BEATEN"

    payload = {
        "schema": "LV_VOLATILITY_NATIVE_TM_V1",
        "interpretation_guard": (
            "Matched-node deep-oracle evidence only; this is not game-strength evidence."
        ),
        "settings": {
            "positions": len(fens),
            "adaptive_control": args.adaptive_control,
            "probe_nodes": args.probe_nodes,
            "cap_nodes": args.cap_nodes,
            "history_depths": args.history_depths,
            "uniform_requested_nodes": uniform_budget,
            "native_selected_clock_nodes": native_clock,
            "native_remaining_moves": args.native_remaining_moves,
            "oracle_nodes": args.oracle_nodes,
            "forced_nodes": args.forced_nodes,
            "seed": args.seed,
        },
        "native_calibration": calibration,
        "adaptive_reproducibility": adaptive_reproducibility,
        "compute": {
            "totals": totals,
            "ratios_to_adaptive": ratios,
            "max_allowed_gap": args.max_node_gap,
            "valid": compute_valid,
        },
        "arms": {name: summarize(values) for name, values in regrets.items()},
        "comparisons": {
            "adaptive_vs_native_tm": adaptive_vs_native,
            "native_tm_vs_uniform": native_vs_uniform,
            "adaptive_vs_uniform": adaptive_vs_uniform,
            "adaptive_vs_shuffled": adaptive_vs_shuffled,
        },
        "status": status,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    if not adaptive_reproducibility["valid"]:
        raise SystemExit("adaptive allocation failed its independent reproducibility gate")
    if not compute_valid:
        raise SystemExit("aggregate node matching failed; quality conclusion is invalid")


if __name__ == "__main__":
    main()

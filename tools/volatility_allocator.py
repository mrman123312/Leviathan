#!/usr/bin/env python3
"""Matched-compute deep-oracle test for root-volatility allocation.

The adaptive arm stops after the first completed root iteration at or beyond
the probe budget when the last root moves are stable.  If they are not stable,
it continues to the cap.  The uniform arm receives the adaptive arm's observed
mean node budget without seeing oracle outcomes.  A shuffled-budget arm keeps
the same budget distribution while breaking its connection to volatility.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

import chess
import chess.engine


MATE_SCORE = 100000


def load_options(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("options JSON must be an object")
    return value


def configure(engine: chess.engine.SimpleEngine, options: dict[str, Any]) -> None:
    unknown = sorted(set(options) - set(engine.options))
    if unknown:
        raise SystemExit(f"engine does not expose options: {unknown}")
    engine.configure(options)


def reset(engine: chess.engine.SimpleEngine) -> object:
    if "Clear Hash" in engine.options:
        engine.configure({"Clear Hash": None})
    return object()


def score_cp(info: dict[str, Any], pov: chess.Color) -> int:
    value = info["score"].pov(pov).score(mate_score=MATE_SCORE)
    return int(value if value is not None else 0)


def adaptive_search(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    probe_nodes: int,
    cap_nodes: int,
    history_depths: int,
) -> dict[str, Any]:
    game = reset(engine)
    by_depth: dict[int, chess.Move] = {}
    last: dict[str, Any] | None = None
    trigger_nodes: int | None = None
    volatile: bool | None = None
    observed_nodes = 0
    with engine.analysis(
        board, chess.engine.Limit(nodes=cap_nodes), multipv=1, game=game
    ) as analysis:
        for info in analysis:
            if info.get("pv"):
                last = info
            if info.get("pv") and info.get("depth") is not None:
                by_depth[int(info["depth"])] = info["pv"][0]
            observed_nodes = max(observed_nodes, int(info.get("nodes", 0)))
            if (
                volatile is None
                and observed_nodes >= probe_nodes
                and len(by_depth) >= history_depths
            ):
                depths = sorted(by_depth)
                recent = [by_depth[depth] for depth in depths[-history_depths:]]
                volatile = len(set(recent)) >= 2
                trigger_nodes = observed_nodes
                if not volatile:
                    analysis.stop()
    if last is None or not last.get("pv"):
        raise SystemExit("adaptive search emitted no principal variation")
    depths = sorted(by_depth)
    return {
        "move": last["pv"][0],
        "nodes": observed_nodes,
        "trigger_nodes": trigger_nodes,
        "volatile": bool(volatile),
        "last_moves": [by_depth[depth].uci() for depth in depths[-history_depths:]],
        "depth": int(last.get("depth", 0)),
        "seldepth": int(last.get("seldepth", 0)),
    }


def fixed_search(
    engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int
) -> dict[str, Any]:
    result = engine.analyse(
        board, chess.engine.Limit(nodes=nodes), multipv=1, game=reset(engine)
    )
    info = result[0] if isinstance(result, list) else result
    if not info.get("pv"):
        raise SystemExit("fixed search emitted no principal variation")
    return {
        "move": info["pv"][0],
        "nodes": int(info.get("nodes", 0)),
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
    }


def oracle_scores(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    moves: set[chess.Move],
    oracle_nodes: int,
    forced_nodes: int,
) -> tuple[int, dict[chess.Move, int], list[str]]:
    infos = engine.analyse(
        board,
        chess.engine.Limit(nodes=oracle_nodes),
        multipv=min(8, board.legal_moves.count()),
        game=reset(engine),
    )
    rows = infos if isinstance(infos, list) else [infos]
    scores = {
        info["pv"][0]: score_cp(info, board.turn)
        for info in rows
        if info.get("pv")
    }
    if not scores:
        raise SystemExit("oracle emitted no principal variations")
    best = max(scores.values())
    oracle_best = sorted(move.uci() for move, value in scores.items() if value == best)
    for move in moves:
        if move not in scores:
            info = engine.analyse(
                board,
                chess.engine.Limit(nodes=forced_nodes),
                root_moves=[move],
                game=reset(engine),
            )
            scores[move] = score_cp(info, board.turn)
    return best, scores, oracle_best


def bootstrap_mean_ci(
    values: list[float], seed: int, samples: int = 30000
) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    means = [
        statistics.mean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    means.sort()
    return [means[int(samples * 0.025)], means[int(samples * 0.975)]]


def summarize(regrets: list[int]) -> dict[str, Any]:
    return {
        "mean_regret_cp": statistics.mean(regrets),
        "median_regret_cp": statistics.median(regrets),
        "oracle_agreement": sum(value == 0 for value in regrets) / len(regrets),
        "over_20cp": sum(value > 20 for value in regrets),
        "over_50cp": sum(value > 50 for value in regrets),
    }


def comparison(
    reference: list[int], candidate: list[int], seed: int
) -> dict[str, Any]:
    advantage = [left - right for left, right in zip(reference, candidate)]
    return {
        "mean_regret_advantage_cp": statistics.mean(advantage),
        "median_regret_advantage_cp": statistics.median(advantage),
        "bootstrap_mean_95pct_ci": bootstrap_mean_ci(advantage, seed),
        "wins": sum(value > 0 for value in advantage),
        "ties": sum(value == 0 for value in advantage),
        "losses": sum(value < 0 for value in advantage),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--probe-nodes", type=int, default=15000)
    parser.add_argument("--cap-nodes", type=int, default=100000)
    parser.add_argument("--history-depths", type=int, default=5)
    parser.add_argument("--oracle-nodes", type=int, default=500000)
    parser.add_argument("--forced-nodes", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=2026081618)
    parser.add_argument("--max-node-gap", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fens = [
        line.strip()
        for line in args.positions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not fens or len(set(fens)) != len(fens):
        raise SystemExit("position corpus must be non-empty and unique")

    options = {"Threads": 1, "Hash": 64} | load_options(args.options)
    adaptive_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    uniform_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    shuffled_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    oracle = chess.engine.SimpleEngine.popen_uci(args.oracle)
    for engine in (adaptive_engine, uniform_engine, shuffled_engine):
        configure(engine, options)
    configure(oracle, {"Threads": 1, "Hash": 64})

    try:
        boards = [chess.Board(fen) for fen in fens]
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
        uniform_budget = max(1, round(adaptive_total / len(boards)))
        uniform = [fixed_search(uniform_engine, board, uniform_budget) for board in boards]

        shuffled_budgets = [row["nodes"] for row in adaptive]
        random.Random(args.seed).shuffle(shuffled_budgets)
        shuffled = [
            fixed_search(shuffled_engine, board, budget)
            for board, budget in zip(boards, shuffled_budgets)
        ]

        rows = []
        regrets = {"adaptive": [], "uniform": [], "shuffled": []}
        for index, (fen, board, arow, urow, srow) in enumerate(
            zip(fens, boards, adaptive, uniform, shuffled), start=1
        ):
            moves = {arow["move"], urow["move"], srow["move"]}
            best, scores, oracle_best = oracle_scores(
                oracle, board, moves, args.oracle_nodes, args.forced_nodes
            )
            row_regrets = {
                "adaptive": best - scores[arow["move"]],
                "uniform": best - scores[urow["move"]],
                "shuffled": best - scores[srow["move"]],
            }
            for name, value in row_regrets.items():
                regrets[name].append(value)
            rows.append(
                {
                    "index": index,
                    "fen": fen,
                    "volatile": arow["volatile"],
                    "last_moves": arow["last_moves"],
                    "trigger_nodes": arow["trigger_nodes"],
                    "adaptive_nodes": arow["nodes"],
                    "uniform_nodes": urow["nodes"],
                    "shuffled_nodes": srow["nodes"],
                    "adaptive_move": arow["move"].uci(),
                    "uniform_move": urow["move"].uci(),
                    "shuffled_move": srow["move"].uci(),
                    "oracle_best": oracle_best,
                    "oracle_best_cp": best,
                    "regret_cp": row_regrets,
                }
            )
    finally:
        for engine in (adaptive_engine, uniform_engine, shuffled_engine, oracle):
            engine.quit()

    totals = {
        "adaptive": sum(row["adaptive_nodes"] for row in rows),
        "uniform": sum(row["uniform_nodes"] for row in rows),
        "shuffled": sum(row["shuffled_nodes"] for row in rows),
    }
    node_ratios = {
        name: value / totals["adaptive"] for name, value in totals.items()
    }
    compute_valid = all(
        abs(ratio - 1.0) <= args.max_node_gap for ratio in node_ratios.values()
    )
    adaptive_vs_uniform = comparison(
        regrets["uniform"], regrets["adaptive"], args.seed + 1
    )
    adaptive_vs_shuffled = comparison(
        regrets["shuffled"], regrets["adaptive"], args.seed + 2
    )
    provisional = (
        compute_valid
        and adaptive_vs_uniform["bootstrap_mean_95pct_ci"][0] > 0
        and adaptive_vs_shuffled["bootstrap_mean_95pct_ci"][0] > 0
        and adaptive_vs_uniform["wins"] >= adaptive_vs_uniform["losses"]
    )
    payload = {
        "schema": "LV_VOLATILITY_ALLOCATOR_V1",
        "interpretation_guard": (
            "This is a matched-aggregate-node deep-oracle screen, not game evidence."
        ),
        "settings": {
            "positions": len(fens),
            "probe_nodes": args.probe_nodes,
            "cap_nodes": args.cap_nodes,
            "history_depths": args.history_depths,
            "uniform_requested_nodes": uniform_budget,
            "oracle_nodes": args.oracle_nodes,
            "forced_nodes": args.forced_nodes,
            "seed": args.seed,
        },
        "compute": {
            "totals": totals,
            "ratios_to_adaptive": node_ratios,
            "max_allowed_gap": args.max_node_gap,
            "valid": compute_valid,
        },
        "volatile_positions": sum(row["volatile"] for row in rows),
        "arms": {name: summarize(values) for name, values in regrets.items()},
        "comparisons": {
            "adaptive_vs_uniform": adaptive_vs_uniform,
            "adaptive_vs_shuffled": adaptive_vs_shuffled,
        },
        "status": "PROVISIONAL_WIN" if provisional else "REJECT_OR_REDESIGN",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    if not compute_valid:
        raise SystemExit("aggregate node matching failed; quality conclusion is invalid")


if __name__ == "__main__":
    main()

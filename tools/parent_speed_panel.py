#!/usr/bin/env python3
"""Calibrated fixed-node speed panel across behavior-changing engine revisions.

This harness deliberately separates throughput evidence from lossless evidence.
Only declared equivalence pairs must reproduce an exact per-position transcript;
other engines are compared at the same requested node budget and their behavioral
differences remain explicit in the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine


MATE_SCORE = 100000


def mappings(items: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name or not value or name in result:
            raise SystemExit(f"invalid or duplicate {label} mapping: {item}")
        result[name] = value
    return result


def load_options(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"options file must contain an object: {path}")
    return value


def configure(engine: chess.engine.SimpleEngine, options: dict[str, Any]) -> None:
    unknown = sorted(set(options) - set(engine.options))
    if unknown:
        raise SystemExit(f"engine does not expose options: {unknown}")
    engine.configure(options)


def clear(engine: chess.engine.SimpleEngine) -> None:
    if "Clear Hash" in engine.options:
        engine.configure({"Clear Hash": None})


def score_cp(info: dict[str, Any], pov: chess.Color) -> int:
    value = info["score"].pov(pov).score(mate_score=MATE_SCORE)
    return int(value if value is not None else 0)


def run_corpus(
    engine: chess.engine.SimpleEngine, fens: list[str], nodes: int
) -> dict[str, Any]:
    clear(engine)
    game = object()
    entries = []
    started = time.perf_counter_ns()
    for index, fen in enumerate(fens):
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=game)
        entries.append(
            {
                "index": index,
                "fen": fen,
                "nodes": int(info.get("nodes", 0)),
                "depth": int(info.get("depth", 0)),
                "seldepth": int(info.get("seldepth", 0)),
                "score_cp": score_cp(info, board.turn),
                "lowerbound": bool(info.get("lowerbound", False)),
                "upperbound": bool(info.get("upperbound", False)),
                "pv": [move.uci() for move in info.get("pv", [])],
            }
        )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "elapsed_ms": elapsed_ms,
        "nodes": sum(entry["nodes"] for entry in entries),
        "behavior_sha256": hashlib.sha256(encoded).hexdigest(),
        "positions": len(entries),
        "entries": entries,
    }


def signature(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": result["nodes"],
        "behavior_sha256": result["behavior_sha256"],
        "positions": result["positions"],
    }


def bootstrap_median_ci(
    values: list[float], seed: int, samples: int = 30000
) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    medians = [
        statistics.median(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    medians.sort()
    return [medians[int(samples * 0.025)], medians[int(samples * 0.975)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", action="append", required=True)
    parser.add_argument("--options", action="append", default=[])
    parser.add_argument("--equivalent", action="append", default=[])
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=50)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--nodes", type=int, default=60000)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2026081611)
    parser.add_argument("--calibration-tolerance", type=float, default=0.008)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = mappings(args.engine, "engine")
    if not {"control_a", "control_b"} <= set(paths):
        raise SystemExit("control_a and control_b engine mappings are required")
    option_paths = mappings(args.options, "options")
    unknown_options = sorted(set(option_paths) - set(paths))
    if unknown_options:
        raise SystemExit(f"options supplied for unknown engines: {unknown_options}")
    equivalents = mappings(args.equivalent, "equivalence")
    for candidate, reference in equivalents.items():
        if candidate not in paths or reference not in paths:
            raise SystemExit(f"unknown equivalence pair: {candidate}={reference}")

    all_fens = [
        line.strip()
        for line in args.positions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fens = all_fens[args.offset : args.offset + args.count]
    if len(fens) != args.count:
        raise SystemExit(
            f"requested {args.count} positions at offset {args.offset}, loaded {len(fens)}"
        )

    engines = {
        name: chess.engine.SimpleEngine.popen_uci(path) for name, path in paths.items()
    }
    for name, engine in engines.items():
        configure(engine, {"Hash": 64, "Threads": 1})
        configure(engine, load_options(option_paths.get(name)))

    try:
        strict = {name: run_corpus(engine, fens, args.nodes) for name, engine in engines.items()}
        signatures = {name: signature(result) for name, result in strict.items()}
        if signatures["control_a"] != signatures["control_b"]:
            raise SystemExit(
                "A/A FUNCTIONAL DIVERGENCE: "
                f"{signatures['control_a']} != {signatures['control_b']}"
            )
        for candidate, reference in equivalents.items():
            if signatures[candidate] != signatures[reference]:
                raise SystemExit(
                    f"DECLARED EQUIVALENCE FAILURE {candidate}!={reference}: "
                    f"{signatures[candidate]} != {signatures[reference]}"
                )

        for engine in engines.values():
            run_corpus(engine, fens[:5], max(5000, args.nodes // 4))

        names = [name for name in paths if name != "control_a"]
        observations: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        rng = random.Random(args.seed)
        for round_index in range(args.rounds):
            order = names[:]
            rng.shuffle(order)
            if round_index % 2:
                order.reverse()
            for name in order:
                first = "control_a" if round_index % 2 == 0 else "control_b"
                second = "control_b" if round_index % 2 == 0 else "control_a"
                before = run_corpus(engines[first], fens, args.nodes)
                middle = run_corpus(engines[name], fens, args.nodes)
                after = run_corpus(engines[second], fens, args.nodes)
                if signature(before) != signatures[first] or signature(after) != signatures[second]:
                    raise SystemExit("reference transcript changed during timing")
                if signature(middle) != signatures[name]:
                    raise SystemExit(f"{name} transcript changed during timing")
                wall_ratio = math.sqrt(before["elapsed_ms"] * after["elapsed_ms"]) / middle[
                    "elapsed_ms"
                ]
                reference_nps = math.sqrt(
                    (before["nodes"] / before["elapsed_ms"])
                    * (after["nodes"] / after["elapsed_ms"])
                )
                candidate_nps = middle["nodes"] / middle["elapsed_ms"]
                observations[name].append(
                    {
                        "round": round_index,
                        "reference_first": first,
                        "reference_first_ms": before["elapsed_ms"],
                        "candidate_ms": middle["elapsed_ms"],
                        "reference_second": second,
                        "reference_second_ms": after["elapsed_ms"],
                        "fixed_budget_wall_ratio": wall_ratio,
                        "node_throughput_ratio": candidate_nps / reference_nps,
                    }
                )
    finally:
        for engine in engines.values():
            engine.quit()

    results = {}
    for index, (name, rows) in enumerate(observations.items()):
        wall = [row["fixed_budget_wall_ratio"] for row in rows]
        throughput = [row["node_throughput_ratio"] for row in rows]
        median = statistics.median(wall)
        interval = bootstrap_median_ci(wall, args.seed + 1000 + index)
        faster = sum(value > 1.0 for value in wall)
        if name == "control_b":
            status = (
                "CALIBRATION_PASS"
                if abs(median - 1.0) <= args.calibration_tolerance
                and interval[0] <= 1.0 <= interval[1]
                else "CALIBRATION_FAIL"
            )
        elif interval[0] > 1.005 and faster >= math.ceil(args.rounds * 0.75):
            status = "OBSERVED_FASTER_FIXED_NODE_BUDGET"
        elif interval[1] < 0.995:
            status = "OBSERVED_SLOWER_FIXED_NODE_BUDGET"
        else:
            status = "INCONCLUSIVE"
        results[name] = {
            "rounds": args.rounds,
            "median_fixed_budget_wall_ratio": median,
            "bootstrap_median_wall_95pct_ci": interval,
            "median_node_throughput_ratio": statistics.median(throughput),
            "faster_rounds": faster,
            "status": status,
            "lossless_vs_parent": signatures[name] == signatures["control_a"],
            "observations": rows,
        }
    if results["control_b"]["status"] != "CALIBRATION_PASS":
        for name in results:
            if name != "control_b":
                results[name]["status"] = "INVALID_CALIBRATION"

    payload = {
        "schema": "LV_PARENT_SPEED_PANEL_V1",
        "interpretation_guard": (
            "Non-equivalent candidates are throughput comparisons at a fixed requested "
            "node budget, not lossless speedups and not strength evidence."
        ),
        "settings": {
            "offset": args.offset,
            "positions": args.count,
            "nodes_per_position": args.nodes,
            "rounds": args.rounds,
            "seed": args.seed,
            "calibration_tolerance": args.calibration_tolerance,
        },
        "declared_equivalences": equivalents,
        "signatures": signatures,
        "transcripts": {name: result["entries"] for name, result in strict.items()},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"transcripts"}
            },
            indent=2,
        )
    )
    if results["control_b"]["status"] != "CALIBRATION_PASS":
        raise SystemExit("A/A calibration failed; conclusions are invalid")


if __name__ == "__main__":
    main()

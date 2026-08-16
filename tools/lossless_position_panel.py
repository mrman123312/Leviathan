#!/usr/bin/env python3
"""Strict search-equivalent speed panel over a fixed set of game positions."""

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


def parse_engines(items: list[str]) -> dict[str, str]:
    engines: dict[str, str] = {}
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in engines:
            raise SystemExit(f"invalid or duplicate engine mapping: {item}")
        engines[name] = path
    required = {"control_a", "control_b"}
    if not required <= set(engines):
        raise SystemExit("control_a and control_b are required")
    if len(engines) < 3:
        raise SystemExit("at least one candidate is required")
    return engines


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


def exact_signature(result: dict[str, Any]) -> dict[str, Any]:
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
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--nodes", type=int, default=60000)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2026081607)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = parse_engines(args.engine)
    options = load_options(args.options)
    fens = [
        line.strip()
        for line in args.positions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.count]
    if len(fens) != args.count:
        raise SystemExit(f"requested {args.count} positions but loaded {len(fens)}")
    engines = {
        name: chess.engine.SimpleEngine.popen_uci(path) for name, path in paths.items()
    }
    for engine in engines.values():
        configure(engine, options)

    try:
        strict = {name: run_corpus(engine, fens, args.nodes) for name, engine in engines.items()}
        signatures = {name: exact_signature(result) for name, result in strict.items()}
        reference = signatures["control_a"]
        divergent = {name: value for name, value in signatures.items() if value != reference}
        if divergent:
            raise SystemExit(
                f"FUNCTIONAL DIVERGENCE: reference={reference} divergent={divergent}"
            )

        for engine in engines.values():
            run_corpus(engine, fens[:5], max(5000, args.nodes // 4))

        candidate_names = [name for name in paths if name != "control_a"]
        observations = {name: [] for name in candidate_names}
        rng = random.Random(args.seed)
        for round_index in range(args.rounds):
            order = candidate_names[:]
            rng.shuffle(order)
            if round_index % 2:
                order.reverse()
            for name in order:
                first = "control_a" if round_index % 2 == 0 else "control_b"
                second = "control_b" if round_index % 2 == 0 else "control_a"
                before = run_corpus(engines[first], fens, args.nodes)
                middle = run_corpus(engines[name], fens, args.nodes)
                after = run_corpus(engines[second], fens, args.nodes)
                trio = [exact_signature(item) for item in (before, middle, after)]
                if not trio[0] == trio[1] == trio[2]:
                    raise SystemExit(
                        f"FUNCTIONAL DIVERGENCE during timing: {name} {trio}"
                    )
                ratio = math.sqrt(before["elapsed_ms"] * after["elapsed_ms"]) / middle[
                    "elapsed_ms"
                ]
                observations[name].append(
                    {
                        "round": round_index,
                        "reference_first": first,
                        "reference_first_ms": before["elapsed_ms"],
                        "candidate_ms": middle["elapsed_ms"],
                        "reference_second": second,
                        "reference_second_ms": after["elapsed_ms"],
                        "sandwich_speedup": ratio,
                    }
                )
    finally:
        for engine in engines.values():
            engine.quit()

    results = {}
    for index, (name, rows) in enumerate(observations.items()):
        ratios = [row["sandwich_speedup"] for row in rows]
        median = statistics.median(ratios)
        interval = bootstrap_median_ci(ratios, args.seed + 1000 + index)
        faster = sum(ratio > 1.0 for ratio in ratios)
        if name == "control_b":
            status = (
                "CALIBRATION_PASS"
                if abs(median - 1.0) <= 0.008 and interval[0] <= 1.0 <= interval[1]
                else "CALIBRATION_FAIL"
            )
        elif interval[0] > 1.005 and faster >= math.ceil(args.rounds * 0.75):
            status = "PROVISIONAL_WIN"
        elif interval[1] < 0.995:
            status = "REJECT_REGRESSION"
        else:
            status = "RETEST_INCONCLUSIVE"
        results[name] = {
            "rounds": args.rounds,
            "median_speedup": median,
            "mean_speedup": statistics.mean(ratios),
            "geometric_mean_speedup": math.exp(
                statistics.mean(math.log(ratio) for ratio in ratios)
            ),
            "bootstrap_median_95pct_ci": interval,
            "faster_rounds": faster,
            "status": status,
            "observations": rows,
        }
    if results["control_b"]["status"] != "CALIBRATION_PASS":
        for name in results:
            if name != "control_b":
                results[name]["status"] = "INVALID_CALIBRATION"

    payload = {
        "schema": "LV_LOSSLESS_POSITION_PANEL_V1",
        "settings": {
            "positions": args.count,
            "nodes_per_position": args.nodes,
            "rounds": args.rounds,
            "seed": args.seed,
        },
        "signatures": signatures,
        "reference_transcript": strict["control_a"]["entries"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "settings": payload["settings"],
                "signatures": signatures,
                "results": results,
            },
            indent=2,
        )
    )
    if results["control_b"]["status"] != "CALIBRATION_PASS":
        raise SystemExit("A/A calibration failed; position-panel conclusions are invalid")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Calibrated P18 warm-hit advantage benchmark.

For each prospective row where the opponent reply is labeled true, compare:

    A1: cold authoritative P09 with own_ms only
    X : warm authoritative P09 after ponder_ms already spent on the same branch
    A2: cold authoritative P09 with own_ms only

A1/A2 are identical-binary controls around X.  The benchmark promotes a result only
when the warm advantage exceeds the same-runner A/A drift guard.  This is meant to
answer the P18 blocker: does correct opponent-clock work become useful authoritative
search before Leviathan's own clock starts, or is the apparent gain measurement noise?
"""
from __future__ import annotations

import argparse
import json
import queue
import statistics
import time
from pathlib import Path
from typing import Any

try:
    from leviathan_hybrid_uci import EngineProcess, append_move_to_position, parse_info
except ImportError:  # pragma: no cover - package-relative execution
    from .leviathan_hybrid_uci import EngineProcess, append_move_to_position, parse_info


def init_engine(path: str, label: str, threads: int, hash_mb: int) -> EngineProcess:
    engine = EngineProcess(path, label)
    engine.send("uci")
    engine.wait_for("uciok", 20)
    engine.initialized = True
    engine.send(f"setoption name Threads value {threads}")
    engine.send(f"setoption name Hash value {hash_mb}")
    engine.send("isready")
    engine.wait_for("readyok", 20)
    return engine


def clear_engine(engine: EngineProcess) -> None:
    engine.drain()
    engine.send("stop")
    engine.drain()
    engine.send("ucinewgame")
    engine.send("setoption name Clear Hash")
    engine.send("isready")
    engine.wait_for("readyok", 20)


def go(engine: EngineProcess, position_cmd: str, movetime_ms: int) -> dict[str, Any]:
    engine.drain()
    engine.send(position_cmd)
    start = time.monotonic()
    engine.send(f"go movetime {movetime_ms}")
    best_move = None
    last_info: dict[str, Any] = {}
    deadline = start + max(10.0, movetime_ms / 1000.0 + 10.0)
    while time.monotonic() < deadline:
        try:
            line = engine.read(1)
        except queue.Empty:
            continue
        info = parse_info(line)
        if info:
            last_info = info
        if line.startswith("bestmove"):
            parts = line.split()
            best_move = parts[1] if len(parts) > 1 else None
            break
    elapsed_ms = int(round((time.monotonic() - start) * 1000.0))
    return {
        "best_move": best_move,
        "depth": int(last_info.get("depth", 0)),
        "seldepth": int(last_info.get("seldepth", 0)),
        "nodes": int(last_info.get("nodes", 0)),
        "elapsed_ms": elapsed_ms,
    }


def prospective_rows(path: str, limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if float(row.get("reply_label") or 0.0) < 0.5:
            continue
        if not row.get("deep_move") or not row.get("reply") or not row.get("position"):
            continue
        group = str(row.get("group_id") or row.get("position"))
        if group in seen:
            continue
        seen.add(group)
        row["_line_number"] = line_number
        out.append(row)
        if limit and len(out) >= limit:
            break
    return out


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def frac(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def run_triplet(
    *,
    index: int,
    row: dict[str, Any],
    cold_a: EngineProcess,
    warm: EngineProcess,
    cold_b: EngineProcess,
    own_ms: int,
    ponder_ms: int,
    reverse_controls: bool,
) -> dict[str, Any]:
    branch = append_move_to_position(row["position"], row["reply"])
    oracle = row["deep_move"]

    clear_engine(cold_a)
    clear_engine(warm)
    clear_engine(cold_b)

    first_engine = cold_b if reverse_controls else cold_a
    second_engine = cold_a if reverse_controls else cold_b

    before = go(first_engine, branch, own_ms)
    _ponder = go(warm, branch, ponder_ms)
    warm_result = go(warm, branch, own_ms)
    after = go(second_engine, branch, own_ms)

    cold_depth_mid = (before["depth"] + after["depth"]) / 2.0
    cold_seldepth_mid = (before["seldepth"] + after["seldepth"]) / 2.0
    cold_nodes_mid = (before["nodes"] + after["nodes"]) / 2.0
    cold_elapsed_mid = (before["elapsed_ms"] + after["elapsed_ms"]) / 2.0

    return {
        "index": index,
        "dataset_line": row.get("_line_number"),
        "group_id": row.get("group_id"),
        "reply": row["reply"],
        "oracle": oracle,
        "reverse_controls": reverse_controls,
        "cold_before": before,
        "warm": warm_result,
        "cold_after": after,
        "cold_before_hit": before["best_move"] == oracle,
        "warm_hit": warm_result["best_move"] == oracle,
        "cold_after_hit": after["best_move"] == oracle,
        "control_hit_delta": int(after["best_move"] == oracle) - int(before["best_move"] == oracle),
        "warm_hit_delta_vs_mid": int(warm_result["best_move"] == oracle)
        - ((int(before["best_move"] == oracle) + int(after["best_move"] == oracle)) / 2.0),
        "control_depth_delta": after["depth"] - before["depth"],
        "warm_depth_delta_vs_mid": warm_result["depth"] - cold_depth_mid,
        "control_seldepth_delta": after["seldepth"] - before["seldepth"],
        "warm_seldepth_delta_vs_mid": warm_result["seldepth"] - cold_seldepth_mid,
        "control_nodes_frac_delta": frac(after["nodes"] - before["nodes"], max(1.0, cold_nodes_mid)),
        "warm_nodes_frac_delta_vs_mid": frac(warm_result["nodes"] - cold_nodes_mid, max(1.0, cold_nodes_mid)),
        "control_elapsed_frac_delta": frac(after["elapsed_ms"] - before["elapsed_ms"], max(1.0, cold_elapsed_mid)),
        "warm_elapsed_frac_delta_vs_mid": frac(warm_result["elapsed_ms"] - cold_elapsed_mid, max(1.0, cold_elapsed_mid)),
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n = len(rows)
    control_depth_abs = [abs(float(r["control_depth_delta"])) for r in rows]
    control_nodes_abs = [abs(float(r["control_nodes_frac_delta"])) for r in rows]
    warm_hit_delta = sum(float(r["warm_hit_delta_vs_mid"]) for r in rows)
    warm_depth_delta = [float(r["warm_depth_delta_vs_mid"]) for r in rows]
    warm_nodes_delta = [float(r["warm_nodes_frac_delta_vs_mid"]) for r in rows]

    calibration_clean = bool(
        n
        and mean(control_depth_abs) <= args.max_control_depth_abs_mean
        and median(control_nodes_abs) <= args.max_control_nodes_abs_median
    )
    warm_advantage = bool(
        warm_hit_delta >= args.min_warm_hit_delta
        and mean(warm_depth_delta) >= args.min_warm_depth_delta_mean
        and median(warm_nodes_delta) >= args.min_warm_nodes_frac_median
    )
    return {
        "positions": n,
        "own_ms": args.own_ms,
        "ponder_ms": args.ponder_ms,
        "threads": args.threads,
        "hash_mb": args.hash,
        "limits": {
            "max_control_depth_abs_mean": args.max_control_depth_abs_mean,
            "max_control_nodes_abs_median": args.max_control_nodes_abs_median,
            "min_warm_hit_delta": args.min_warm_hit_delta,
            "min_warm_depth_delta_mean": args.min_warm_depth_delta_mean,
            "min_warm_nodes_frac_median": args.min_warm_nodes_frac_median,
        },
        "control": {
            "mean_abs_depth_delta": mean(control_depth_abs),
            "median_abs_nodes_frac_delta": median(control_nodes_abs),
            "hit_delta_sum": sum(int(r["control_hit_delta"]) for r in rows),
        },
        "warm_vs_control_midpoint": {
            "oracle_hit_delta_sum": warm_hit_delta,
            "mean_depth_delta": mean(warm_depth_delta),
            "mean_seldepth_delta": mean([float(r["warm_seldepth_delta_vs_mid"]) for r in rows]),
            "median_nodes_frac_delta": median(warm_nodes_delta),
            "median_elapsed_frac_delta": median([float(r["warm_elapsed_frac_delta_vs_mid"]) for r in rows]),
        },
        "calibration_clean": calibration_clean,
        "warm_advantage": warm_advantage,
        "pass": bool(calibration_clean and warm_advantage),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ponder-ms", type=int, default=2000)
    parser.add_argument("--own-ms", type=int, default=250)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--max-control-depth-abs-mean", type=float, default=0.25)
    parser.add_argument("--max-control-nodes-abs-median", type=float, default=0.08)
    parser.add_argument("--min-warm-hit-delta", type=float, default=1.0)
    parser.add_argument("--min-warm-depth-delta-mean", type=float, default=0.25)
    parser.add_argument("--min-warm-nodes-frac-median", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = prospective_rows(args.dataset, args.limit)
    cold_a = init_engine(args.engine, "cold-a", args.threads, args.hash)
    warm = init_engine(args.engine, "warm", args.threads, args.hash)
    cold_b = init_engine(args.engine, "cold-b", args.threads, args.hash)
    results: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(data):
            result = run_triplet(
                index=index,
                row=row,
                cold_a=cold_a,
                warm=warm,
                cold_b=cold_b,
                own_ms=args.own_ms,
                ponder_ms=args.ponder_ms,
                reverse_controls=bool(index % 2),
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        cold_a.close()
        warm.close()
        cold_b.close()

    summary = summarize(results, args)
    payload = {"summary": summary, "rows": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["pass"] else 5


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Calibrated throughput comparison for engines whose search behavior may differ.

Unlike lossless_pair_benchmark.py, this harness intentionally does NOT require
identical node transcripts. It compares milliseconds per searched node using
sandwiched A/B measurements and preserves an A/A control to detect runner drift.
It is appropriate for search-policy mutations, not for proving functional
identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
from pathlib import Path


def uci(options: dict) -> str:
    out = []
    for k, v in options.items():
        if isinstance(v, bool):
            v = str(v).lower()
        out.append(f"setoption name {k} value {v}")
    return "\n".join(out) + "\n"


def run(binary: str, options: dict, command: str) -> dict:
    p = subprocess.run(
        [binary],
        input=uci(options) + command + "\nquit\n",
        text=True,
        capture_output=True,
        check=True,
    )
    o = p.stdout + p.stderr
    try:
        ms = int(re.findall(r"Total time \(ms\)\s*:\s*(\d+)", o)[-1])
        nodes = int(re.findall(r"Nodes searched\s*:\s*(\d+)", o)[-1])
        nps = int(re.findall(r"Nodes/second\s*:\s*(\d+)", o)[-1])
    except IndexError as exc:
        raise SystemExit(f"Could not parse bench output from {binary}\n{o[-4000:]}") from exc
    if ms <= 0 or nodes <= 0:
        raise SystemExit(f"Invalid bench result ms={ms} nodes={nodes} from {binary}")
    return {
        "ms": ms,
        "nodes": nodes,
        "nps": nps,
        "ns_per_node": ms * 1_000_000.0 / nodes,
    }


def bootstrap_median(vals: list[float], seed: int, samples: int = 30000) -> list[float]:
    rng = random.Random(seed)
    n = len(vals)
    medians = [statistics.median(vals[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    medians.sort()
    return [medians[int(samples * 0.025)], medians[int(samples * 0.975)]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-a", required=True)
    ap.add_argument("--control-b", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--options", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--command", default="bench 16 1 50000 default nodes")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    opts = json.loads(args.options.read_text())
    eng = {
        "control_a": args.control_a,
        "control_b": args.control_b,
        "candidate": args.candidate,
    }

    # Warm all binaries twice so cold startup/network page faults do not dominate.
    warm = {name: [] for name in eng}
    for name, binary in eng.items():
        for _ in range(2):
            warm[name].append(run(binary, opts, args.command))

    obs = {"control_b": [], "candidate": []}
    rng = random.Random(2026081701)
    for i in range(args.rounds):
        order = ["control_b", "candidate"]
        rng.shuffle(order)
        if i % 2:
            order.reverse()
        for name in order:
            first = "control_a" if i % 2 == 0 else "control_b"
            second = "control_b" if i % 2 == 0 else "control_a"
            before = run(eng[first], opts, args.command)
            mid = run(eng[name], opts, args.command)
            after = run(eng[second], opts, args.command)

            # Compare cost per searched node, not raw wall time, because search
            # policy mutations legitimately visit a different number of nodes.
            ref_ns = math.sqrt(before["ns_per_node"] * after["ns_per_node"])
            throughput_ratio = ref_ns / mid["ns_per_node"]
            obs[name].append(
                {
                    "round": i,
                    "reference_first": first,
                    "reference_first_ms": before["ms"],
                    "reference_first_nodes": before["nodes"],
                    "measured_ms": mid["ms"],
                    "measured_nodes": mid["nodes"],
                    "reference_second": second,
                    "reference_second_ms": after["ms"],
                    "reference_second_nodes": after["nodes"],
                    "throughput_ratio": throughput_ratio,
                }
            )

    results = {}
    for ix, (name, rows) in enumerate(obs.items()):
        ratios = [r["throughput_ratio"] for r in rows]
        med = statistics.median(ratios)
        interval = bootstrap_median(ratios, 52000 + ix)
        faster = sum(x > 1 for x in ratios)
        if name == "control_b":
            status = (
                "CALIBRATION_PASS"
                if abs(med - 1) <= 0.006 and interval[0] <= 1 <= interval[1]
                else "CALIBRATION_FAIL"
            )
        else:
            status = (
                "CANDIDATE_FASTER"
                if interval[0] > 1.002
                else "CANDIDATE_SLOWER"
                if interval[1] < 0.998
                else "THROUGHPUT_INCONCLUSIVE"
            )
        results[name] = {
            "rounds": args.rounds,
            "median_throughput_ratio": med,
            "mean_throughput_ratio": statistics.mean(ratios),
            "geometric_mean_throughput_ratio": math.exp(statistics.mean(math.log(x) for x in ratios)),
            "bootstrap_median_95pct_ci": interval,
            "faster_rounds": faster,
            "status": status,
            "observations": rows,
        }

    if results["control_b"]["status"] != "CALIBRATION_PASS":
        results["candidate"]["status"] = "INVALID_CALIBRATION"

    payload = {
        "schema": "LV_DIVERGENT_PAIR_THROUGHPUT_V1",
        "meaning": "ratio > 1 means candidate processes searched nodes faster than P01 baseline; search quality is intentionally out of scope",
        "command": args.command,
        "runner": {
            "image": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
            "arch": os.uname().machine,
        },
        "binary_sha256": {
            name: hashlib.sha256(Path(binary).read_bytes()).hexdigest()
            for name, binary in eng.items()
        },
        "warmup": warm,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if results["control_b"]["status"] != "CALIBRATION_PASS":
        raise SystemExit("A/A calibration failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

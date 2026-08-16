#!/usr/bin/env python3
"""Strict transcript-equivalent A/A/B speed screen for one lossless candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import resource
import statistics
import subprocess
from pathlib import Path
from typing import Any


def uci_commands(options: dict[str, Any]) -> str:
    lines = []
    for name, value in options.items():
        if isinstance(value, bool):
            value = str(value).lower()
        lines.append(f"setoption name {name} value {value}")
    return "\n".join(lines) + "\n"


def normalize_search_transcript(output: str) -> str:
    normalized = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Position:") or line.startswith("bestmove "):
            normalized.append(line)
        elif line.startswith("info ") and not line.startswith("info string "):
            line = re.sub(r"\s+nps\s+\d+", "", line)
            line = re.sub(r"\s+time\s+\d+", "", line)
            normalized.append(line)
    return "\n".join(normalized)


def run_bench(
    binary: str,
    options: dict[str, Any],
    command: str = "bench",
    repeats: int = 1,
) -> dict[str, Any]:
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    process = subprocess.run(
        [binary],
        input=uci_commands(options) + (command + "\n") * repeats + "quit\n",
        text=True,
        capture_output=True,
        check=True,
    )
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_ms = 1000.0 * (
        usage_after.ru_utime
        + usage_after.ru_stime
        - usage_before.ru_utime
        - usage_before.ru_stime
    )
    output = process.stdout + process.stderr
    transcript = normalize_search_transcript(output)
    if not transcript:
        raise SystemExit(f"no normalized search transcript emitted by {binary}")
    times = [int(value) for value in re.findall(r"Total time \(ms\)\s*:\s*(\d+)", output)]
    nodes = [int(value) for value in re.findall(r"Nodes searched\s*:\s*(\d+)", output)]
    if len(times) != repeats or len(nodes) != repeats:
        raise SystemExit(
            f"expected {repeats} bench summaries from {binary}, got {len(times)}/{len(nodes)}"
        )
    return {
        "ms": sum(times),
        "cpu_ms": cpu_ms,
        "nodes": sum(nodes),
        "nps": sum(nodes) * 1000 // max(1, sum(times)),
        "behavior_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
        "behavior_lines": len(transcript.splitlines()),
    }


def bootstrap_median_ci(values: list[float], seed: int, samples: int = 30000) -> list[float]:
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
    parser.add_argument("--control-a", required=True)
    parser.add_argument("--control-b", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--corpus-repeats", type=int, default=1)
    parser.add_argument("--metric", choices=("wall", "cpu"), default="wall")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    options = json.loads(args.options.read_text(encoding="utf-8"))
    engines = {
        "control_a": args.control_a,
        "control_b": args.control_b,
        "candidate": args.candidate,
    }

    signatures = {name: {} for name in engines}
    commands = {
        "default": "bench",
        "depth11": "bench 16 1 11 default depth",
        "nodes50k": "bench 16 1 50000 default nodes",
    }
    exact_keys = ("nodes", "behavior_sha256", "behavior_lines")
    for name, binary in engines.items():
        for label, command in commands.items():
            result = run_bench(binary, options, command, args.corpus_repeats)
            signatures[name][label] = {key: result[key] for key in exact_keys}
    reference_signature = signatures["control_a"]
    divergent = {name: value for name, value in signatures.items() if value != reference_signature}
    if divergent:
        raise SystemExit(
            f"FUNCTIONAL DIVERGENCE: reference={reference_signature} divergent={divergent}"
        )

    for binary in engines.values():
        run_bench(binary, options, repeats=args.corpus_repeats)

    observations = {"control_b": [], "candidate": []}
    metric_key = "ms" if args.metric == "wall" else "cpu_ms"
    rng = random.Random(2026081604)
    for round_index in range(args.rounds):
        order = ["control_b", "candidate"]
        rng.shuffle(order)
        if round_index % 2:
            order.reverse()
        for name in order:
            first = "control_a" if round_index % 2 == 0 else "control_b"
            second = "control_b" if round_index % 2 == 0 else "control_a"
            before = run_bench(
                engines[first], options, repeats=args.corpus_repeats
            )
            middle = run_bench(
                engines[name], options, repeats=args.corpus_repeats
            )
            after = run_bench(
                engines[second], options, repeats=args.corpus_repeats
            )
            if not all(before[key] == middle[key] == after[key] for key in exact_keys):
                raise SystemExit(
                    f"FUNCTIONAL DIVERGENCE during timing: {name} {before} {middle} {after}"
                )
            ratio = math.sqrt(before[metric_key] * after[metric_key]) / middle[metric_key]
            observations[name].append(
                {
                    "round": round_index,
                    "reference_first": first,
                    "reference_first_ms": before["ms"],
                    "reference_first_cpu_ms": before["cpu_ms"],
                    "candidate_ms": middle["ms"],
                    "candidate_cpu_ms": middle["cpu_ms"],
                    "reference_second": second,
                    "reference_second_ms": after["ms"],
                    "reference_second_cpu_ms": after["cpu_ms"],
                    "sandwich_speedup": ratio,
                }
            )

    results = {}
    for index, (name, rows) in enumerate(observations.items()):
        ratios = [row["sandwich_speedup"] for row in rows]
        median = statistics.median(ratios)
        ci = bootstrap_median_ci(ratios, 41000 + index)
        faster = sum(ratio > 1.0 for ratio in ratios)
        if name == "control_b":
            status = (
                "CALIBRATION_PASS"
                if abs(median - 1.0) <= 0.004 and ci[0] <= 1.0 <= ci[1]
                else "CALIBRATION_FAIL"
            )
        elif ci[0] > 1.002 and faster >= math.ceil(args.rounds * 0.75):
            status = "PROVISIONAL_WIN"
        elif ci[1] < 0.998:
            status = "REJECT_REGRESSION"
        else:
            status = "RETEST_INCONCLUSIVE"
        results[name] = {
            "rounds": args.rounds,
            "median_speedup": median,
            "mean_speedup": statistics.mean(ratios),
            "geometric_mean_speedup": math.exp(statistics.mean(math.log(x) for x in ratios)),
            "bootstrap_median_95pct_ci": ci,
            "faster_rounds": faster,
            "status": status,
            "observations": rows,
        }

    if results["control_b"]["status"] != "CALIBRATION_PASS":
        results["candidate"]["status"] = "INVALID_CALIBRATION"

    payload = {
        "schema": "LV_LOSSLESS_PAIR_V1",
        "runner": {
            "image": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
            "arch": os.uname().machine,
            "corpus_repeats": args.corpus_repeats,
            "timing_metric": args.metric,
        },
        "signatures": signatures,
        "binary_sha256": {
            name: hashlib.sha256(Path(binary).read_bytes()).hexdigest()
            for name, binary in engines.items()
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if results["control_b"]["status"] != "CALIBRATION_PASS":
        raise SystemExit("A/A calibration failed; candidate speed conclusion is invalid")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail closed unless two engines match three normalized search transcripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lossless_pair_benchmark import run_bench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    options: dict[str, Any] = json.loads(args.options.read_text(encoding="utf-8"))
    workloads = {
        "default": "bench",
        "depth11": "bench 16 1 11 default depth",
        "nodes50k": "bench 16 1 50000 default nodes",
    }
    exact_fields = ("nodes", "behavior_sha256", "behavior_lines")
    signatures: dict[str, dict[str, dict[str, Any]]] = {}
    for name, binary in (("reference", args.reference), ("candidate", args.candidate)):
        signatures[name] = {}
        for workload, command in workloads.items():
            result = run_bench(binary, options, command)
            signatures[name][workload] = {
                field: result[field] for field in exact_fields
            }

    mismatches = []
    for workload in workloads:
        for field in exact_fields:
            if signatures["reference"][workload][field] != signatures["candidate"][workload][field]:
                mismatches.append(f"{workload}.{field}")

    payload = {
        "schema": "LV_EXACT_BEHAVIOR_GATE_V1",
        "reference": args.reference,
        "candidate": args.candidate,
        "signatures": signatures,
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if mismatches:
        raise SystemExit("default-off search behavior diverged")


if __name__ == "__main__":
    main()

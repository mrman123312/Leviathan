"""Repeated Stockfish bench runner with robust NPS summaries."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

NODES_RE = re.compile(r"Nodes searched\s*:\s*(\d+)")
TIME_RE = re.compile(r"Total time \(ms\)\s*:\s*(\d+)")
NPS_RE = re.compile(r"Nodes/second\s*:\s*(\d+)")


def one(binary: str, uci_prefix: str) -> dict[str, int]:
    if uci_prefix:
        proc = subprocess.run(
            [binary], input=uci_prefix + "\nbench\nquit\n", text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        )
    else:
        proc = subprocess.run([binary, "bench"], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=True)
    text = proc.stdout
    def get(rx: re.Pattern[str], name: str) -> int:
        m = rx.search(text)
        if not m:
            raise RuntimeError(f"missing {name} in bench output")
        return int(m.group(1))
    return {"nodes": get(NODES_RE, "nodes"), "time_ms": get(TIME_RE, "time"), "nps": get(NPS_RE, "nps")}


def summarize(rows: list[dict[str, int]]) -> dict[str, float | int]:
    nps = [r["nps"] for r in rows]
    times = [r["time_ms"] for r in rows]
    return {
        "runs": len(rows),
        "nodes": rows[0]["nodes"],
        "median_nps": statistics.median(nps),
        "mean_nps": statistics.mean(nps),
        "stdev_nps": statistics.stdev(nps) if len(nps) > 1 else 0.0,
        "median_time_ms": statistics.median(times),
        "min_nps": min(nps),
        "max_nps": max(nps),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--uci", type=Path, help="optional UCI commands before bench")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    prefix = args.uci.read_text(encoding="utf-8") if args.uci else ""
    rows = [one(args.binary, prefix) for _ in range(args.runs)]
    payload = {"binary": args.binary, "summary": summarize(rows), "runs": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

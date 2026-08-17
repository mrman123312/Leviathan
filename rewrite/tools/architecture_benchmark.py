#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

INFO_RE = re.compile(r"info depth (\d+) score cp (-?\d+) nodes (\d+)")


class Engine:
    def __init__(self, path: str):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1
        )

    def send(self, text: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def read(self) -> str:
        assert self.proc.stdout
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"engine ended unexpectedly: {err}")
        return line.strip()

    def go(self, command: str) -> dict:
        self.send("ucinewgame")
        self.send("position startpos")
        start = time.perf_counter()
        self.send(command)
        latest = None
        while True:
            line = self.read()
            match = INFO_RE.search(line)
            if match:
                latest = {
                    "depth": int(match.group(1)),
                    "score_cp": int(match.group(2)),
                    "nodes": int(match.group(3)),
                }
            if line.startswith("bestmove "):
                elapsed = time.perf_counter() - start
                if latest is None:
                    raise RuntimeError(f"no info line before {line}")
                latest["wall_ms"] = elapsed * 1000.0
                latest["bestmove"] = line.split()[1]
                return latest

    def perft(self, depth: int) -> dict:
        self.send("position startpos")
        start = time.perf_counter()
        self.send(f"perft {depth}")
        line = self.read()
        elapsed = time.perf_counter() - start
        if not line.startswith("nodes "):
            raise RuntimeError(f"bad perft response: {line}")
        return {"nodes": int(line.split()[1]), "wall_ms": elapsed * 1000.0}

    def close(self) -> None:
        try:
            self.send("quit")
        except Exception:
            pass
        self.proc.wait(timeout=5)


def median_record(records: list[dict]) -> dict:
    out = {"runs": records}
    for key in ("wall_ms", "nodes", "depth", "score_cp"):
        vals = [r[key] for r in records if key in r]
        if vals:
            out[f"median_{key}"] = statistics.median(vals)
    moves = [r.get("bestmove") for r in records if "bestmove" in r]
    if moves:
        out["bestmoves"] = moves
    return out


def benchmark(path: str, repeats: int) -> dict:
    engine = Engine(path)
    try:
        fixed = [engine.go("go depth 5") for _ in range(repeats)]
        timed = [engine.go("go movetime 150") for _ in range(repeats)]
        perft = [engine.perft(5) for _ in range(repeats)]
        return {
            "fixed_depth_5": median_record(fixed),
            "movetime_150ms": median_record(timed),
            "perft_5": median_record(perft),
        }
    finally:
        engine.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--base-ref", default="489e154b231b0922702892c76ab44efddf26bef5")
    ap.add_argument("--current-ref", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    result = {
        "schema_version": 1,
        "base_ref": args.base_ref,
        "current_ref": args.current_ref,
        "repeats": args.repeats,
        "base": benchmark(args.base, args.repeats),
        "current": benchmark(args.current, args.repeats),
    }

    b = result["base"]
    c = result["current"]
    result["derived"] = {
        "fixed_depth_wall_speedup": b["fixed_depth_5"]["median_wall_ms"] / c["fixed_depth_5"]["median_wall_ms"],
        "perft_wall_speedup": b["perft_5"]["median_wall_ms"] / c["perft_5"]["median_wall_ms"],
        "timed_depth_delta": c["movetime_150ms"]["median_depth"] - b["movetime_150ms"]["median_depth"],
        "timed_node_ratio": c["movetime_150ms"]["median_nodes"] / max(1, b["movetime_150ms"]["median_nodes"]),
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

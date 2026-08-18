#!/usr/bin/env python3
"""Label FEN corpora with multiple UCI teachers while preserving disagreement."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TeacherResult:
    bestmove: str
    score_type: str | None
    score: int | None
    depth: int | None
    nodes: int | None


class UciTeacher:
    def __init__(self, name: str, executable: str):
        self.name = name
        self.executable = executable
        self.proc = subprocess.Popen(
            [executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1
        )
        self._send("uci")
        self._until("uciok")
        self._send("isready")
        self._until("readyok")

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _until(self, token: str) -> list[str]:
        assert self.proc.stdout is not None
        lines: list[str] = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                err = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"{self.name} exited while waiting for {token}: {err}")
            line = line.rstrip("\r\n")
            lines.append(line)
            if line == token or line.startswith(token + " "):
                return lines

    def label(self, fen: str, movetime: int, depth: int | None) -> TeacherResult:
        self._send("position fen " + fen)
        self._send(f"go depth {depth}" if depth else f"go movetime {movetime}")
        lines = self._until("bestmove")
        score_type = None
        score = None
        seen_depth = None
        nodes = None
        bestmove = "0000"
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "info":
                if "depth" in parts:
                    try: seen_depth = int(parts[parts.index("depth") + 1])
                    except (ValueError, IndexError): pass
                if "nodes" in parts:
                    try: nodes = int(parts[parts.index("nodes") + 1])
                    except (ValueError, IndexError): pass
                if "score" in parts:
                    try:
                        i = parts.index("score")
                        if parts[i + 1] in ("cp", "mate"):
                            score_type = parts[i + 1]
                            score = int(parts[i + 2])
                    except (ValueError, IndexError):
                        pass
            elif parts[0] == "bestmove" and len(parts) > 1:
                bestmove = parts[1]
        return TeacherResult(bestmove, score_type, score, seen_depth, nodes)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send("quit")
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()


def parse_teacher(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("teacher must be NAME=/path/to/engine")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("teacher must be NAME=/path/to/engine")
    return name, path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen-file", required=True)
    ap.add_argument("--teacher", action="append", required=True, type=parse_teacher)
    ap.add_argument("--movetime", type=int, default=250)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--source", default="unspecified")
    ap.add_argument("--license", dest="license_name", default="unspecified")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.movetime <= 0 or (args.depth is not None and args.depth <= 0):
        ap.error("movetime/depth must be positive")

    fens = []
    for raw in Path(args.fen_file).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            fens.append(line)
    if not fens:
        raise SystemExit("no FENs found")

    teachers: list[UciTeacher] = []
    try:
        for name, executable in args.teacher:
            teachers.append(UciTeacher(name, executable))

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as out:
            for index, fen in enumerate(fens):
                teacher_payload = {}
                cp_scores: list[int] = []
                candidates: list[str] = []
                for teacher in teachers:
                    result = teacher.label(fen, args.movetime, args.depth)
                    teacher_payload[teacher.name] = {
                        "engine": teacher.executable,
                        "bestmove": result.bestmove,
                        "score_type": result.score_type,
                        "score": result.score,
                        "depth": result.depth,
                        "nodes": result.nodes,
                    }
                    if result.score_type == "cp" and result.score is not None:
                        cp_scores.append(result.score)
                    if result.bestmove != "0000" and result.bestmove not in candidates:
                        candidates.append(result.bestmove)

                record = {
                    "schema_version": 1,
                    "index": index,
                    "fen": fen,
                    "source": args.source,
                    "license": args.license_name,
                    "teachers": teacher_payload,
                    "candidate_moves": candidates,
                    "consensus_cp": round(statistics.mean(cp_scores), 3) if cp_scores else None,
                    "disagreement_cp": round(statistics.pstdev(cp_scores), 3) if len(cp_scores) > 1 else None,
                    "bestmove_disagreement": len(candidates) > 1,
                    "tags": [],
                }
                if record["bestmove_disagreement"]:
                    record["tags"].append("teacher_disagreement")
                if record["disagreement_cp"] is not None and record["disagreement_cp"] >= 75:
                    record["tags"].append("high_score_disagreement")
                out.write(json.dumps(record, sort_keys=True) + "\n")
                print(f"labeled {index + 1}/{len(fens)}", file=sys.stderr)
    finally:
        for teacher in teachers:
            teacher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

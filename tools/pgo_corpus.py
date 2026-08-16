"""Game-shaped PGO workload for Leviathan release builds.

Stockfish's standard profile-build uses `bench`. This corpus supplements that
shape with repeated UCI searches from opening, middlegame and small-endgame
positions so branch/layout optimization sees more of actual engine control flow.
It is compiler profiling, not model training.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


POSITIONS = [
    "startpos",
    "startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7",
    "startpos moves d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5",
    "startpos moves e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "startpos moves c2c4 e7e5 b1c3 g8f6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6",
    "startpos moves g1f3 d7d5 g2g3 c7c5 f1g2 b8c6 e1g1 e7e5 d2d3 g8f6",
    "startpos moves e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 e4e5 f6d7 f2f4 c7c5",
    "startpos moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c4d5 e6d5 c1g5 f8e7",
    "startpos moves e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2 c6c5",
    "startpos moves d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8",
    "fen 8/8/8/3k4/3P4/3K4/8/8 w - - 0 1",
    "fen 8/8/3k4/8/3P4/4K3/8/8 b - - 0 1",
    "fen 8/5pk1/6p1/7p/4P3/5P2/6PP/6K1 w - - 0 1",
    "fen 8/8/2k5/8/2P5/2K5/8/8 w - - 0 1",
    "fen 8/8/8/2k5/8/2K5/3R4/8 w - - 0 1",
]


def wait_for(proc: subprocess.Popen[str], token: str) -> None:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"engine exited before {token}")
        if line.startswith(token):
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--nodes", type=int, default=120_000)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    proc = subprocess.Popen([args.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdin is not None
    try:
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        wait_for(proc, "uciok")
        proc.stdin.write("setoption name Threads value 1\nsetoption name Hash value 32\nisready\n")
        proc.stdin.flush()
        wait_for(proc, "readyok")

        for _ in range(args.rounds):
            for position in POSITIONS:
                proc.stdin.write("ucinewgame\n")
                proc.stdin.write(f"position {position}\n")
                proc.stdin.write(f"go nodes {args.nodes}\n")
                proc.stdin.flush()
                wait_for(proc, "bestmove")
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        rc = proc.wait(timeout=10)
        if rc:
            raise RuntimeError(f"engine exited {rc}")
    finally:
        if proc.poll() is None:
            proc.kill()
    print(f"pgo-corpus positions={len(POSITIONS) * args.rounds} nodes_each={args.nodes}")


if __name__ == "__main__":
    main()

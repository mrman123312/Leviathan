#!/usr/bin/env python3
"""Deterministic correctness audit against python-chess.

This is deliberately independent of Leviathan's move generator. It compares
exact legal move sets on every sampled position and depth-2 perft on a subset.
Any divergence is a hard failure with the seed/FEN printed for reproduction.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass

import chess


@dataclass
class Engine:
    proc: subprocess.Popen

    @classmethod
    def start(cls, path: str) -> "Engine":
        proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return cls(proc)

    def command(self, text: str) -> str:
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"engine ended after {text!r}: {err}")
        return line.strip()

    def position(self, fen: str) -> None:
        # position emits nothing on success.
        assert self.proc.stdin is not None
        self.proc.stdin.write(f"position fen {fen}\n")
        self.proc.stdin.flush()

    def legal(self, fen: str) -> set[str]:
        self.position(fen)
        line = self.command("legal")
        if not line.startswith("legal"):
            raise RuntimeError(f"unexpected legal response: {line}")
        return set(line.split()[1:])

    def perft(self, fen: str, depth: int) -> int:
        self.position(fen)
        line = self.command(f"perft {depth}")
        parts = line.split()
        if len(parts) != 2 or parts[0] != "nodes":
            raise RuntimeError(f"unexpected perft response: {line}")
        return int(parts[1])

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
        finally:
            self.proc.wait(timeout=5)


def py_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += py_perft(board, depth - 1)
        board.pop()
    return total


def targeted_positions() -> list[chess.Board]:
    fens = [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",  # checkmate
        "r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N5/PPP2PPP/R3K2R w KQkq - 0 10",
    ]
    return [chess.Board(fen) for fen in fens]


def generated_positions(seed: int, count: int) -> list[chess.Board]:
    rng = random.Random(seed)
    board = chess.Board()
    out: list[chess.Board] = []
    seen: set[str] = set()

    attempts = 0
    while len(out) < count and attempts < count * 20:
        attempts += 1
        if board.is_game_over(claim_draw=False) or board.ply() > 180:
            board.reset()

        legal = list(board.legal_moves)
        if not legal:
            board.reset()
            continue
        board.push(rng.choice(legal))

        # Sample a broad range of game phases instead of only long playout tails.
        if board.ply() >= 4 and (rng.random() < 0.35 or board.ply() % 11 == 0):
            fen = board.fen(en_passant="fen")
            identity = " ".join(fen.split()[:4])
            if identity not in seen:
                seen.add(identity)
                out.append(board.copy(stack=False))

        if rng.random() < 0.08:
            board.reset()

    if len(out) != count:
        raise RuntimeError(f"generated only {len(out)} of {count} requested positions")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--seed", type=int, default=8910)
    parser.add_argument("--positions", type=int, default=160)
    parser.add_argument("--perft-positions", type=int, default=80)
    parser.add_argument("--perft-depth", type=int, default=2)
    args = parser.parse_args()

    positions = targeted_positions() + generated_positions(args.seed, args.positions)
    engine = Engine.start(args.engine)
    legal_checked = 0
    perft_checked = 0

    try:
        for i, board in enumerate(positions):
            fen = board.fen(en_passant="fen")
            expected_moves = {m.uci() for m in board.legal_moves}
            got_moves = engine.legal(fen)
            legal_checked += 1
            if got_moves != expected_moves:
                missing = sorted(expected_moves - got_moves)
                extra = sorted(got_moves - expected_moves)
                print(f"LEGAL DIVERGENCE index={i} seed={args.seed}", file=sys.stderr)
                print(f"FEN: {fen}", file=sys.stderr)
                print(f"missing: {missing}", file=sys.stderr)
                print(f"extra: {extra}", file=sys.stderr)
                return 2

            if i < args.perft_positions or i < len(targeted_positions()):
                expected_nodes = py_perft(board, args.perft_depth)
                got_nodes = engine.perft(fen, args.perft_depth)
                perft_checked += 1
                if got_nodes != expected_nodes:
                    print(f"PERFT DIVERGENCE index={i} seed={args.seed} depth={args.perft_depth}", file=sys.stderr)
                    print(f"FEN: {fen}", file=sys.stderr)
                    print(f"expected={expected_nodes} got={got_nodes}", file=sys.stderr)
                    return 3
    finally:
        engine.close()

    print(
        f"differential audit ok seed={args.seed} legal_positions={legal_checked} "
        f"perft_positions={perft_checked} depth={args.perft_depth}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded proof-style mate specialist for Leviathan experiments.

This is deliberately separate from the hot Stockfish node loop. It performs an
AND/OR proof search: the attacker needs one proving move; the defender must fail
on every legal reply. The router may later invoke a compiled equivalent only if
this specialist demonstrates useful marginal value on held-out positions.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import chess


@dataclass
class Result:
    proven: bool
    pv: list[str]
    nodes: int
    exhausted: bool = False


class Prover:
    def __init__(self, max_nodes: int):
        self.max_nodes = max_nodes
        self.nodes = 0
        self.memo: dict[tuple[str, int, bool], tuple[bool, tuple[str, ...]]] = {}

    def solve(self, board: chess.Board, attacker: chess.Color, plies: int) -> tuple[bool, list[str]]:
        if self.nodes >= self.max_nodes:
            return False, []
        self.nodes += 1
        key = (board.fen(), plies, attacker)
        cached = self.memo.get(key)
        if cached is not None:
            return cached[0], list(cached[1])

        if board.is_checkmate():
            ok = board.turn != attacker
            self.memo[key] = (ok, ())
            return ok, []
        if plies <= 0 or board.is_game_over(claim_draw=True):
            self.memo[key] = (False, ())
            return False, []

        legal = list(board.legal_moves)
        # Move ordering only; completeness is preserved because all moves remain.
        legal.sort(key=lambda m: (board.gives_check(m), board.is_capture(m), bool(m.promotion)), reverse=True)

        if board.turn == attacker:
            for move in legal:
                board.push(move)
                ok, child = self.solve(board, attacker, plies - 1)
                board.pop()
                if ok:
                    pv = [move.uci(), *child]
                    self.memo[key] = (True, tuple(pv))
                    return True, pv
            self.memo[key] = (False, ())
            return False, []

        # Defender node: every legal reply must still allow a forced mate.
        longest: list[str] = []
        for move in legal:
            board.push(move)
            ok, child = self.solve(board, attacker, plies - 1)
            board.pop()
            if not ok:
                self.memo[key] = (False, ())
                return False, []
            candidate = [move.uci(), *child]
            if len(candidate) > len(longest):
                longest = candidate
        self.memo[key] = (True, tuple(longest))
        return True, longest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", help="single FEN")
    ap.add_argument("--fens", type=Path, help="one FEN per line")
    ap.add_argument("--plies", type=int, default=7)
    ap.add_argument("--max-nodes", type=int, default=250000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if bool(args.fen) == bool(args.fens):
        raise SystemExit("provide exactly one of --fen or --fens")
    fens = [args.fen] if args.fen else [x.strip() for x in args.fens.read_text().splitlines() if x.strip()]
    rows = []
    for fen in fens:
        board = chess.Board(fen)
        prover = Prover(args.max_nodes)
        attacker = board.turn
        proven, pv = prover.solve(board, attacker, args.plies)
        rows.append({
            "fen": fen,
            "attacker": "white" if attacker else "black",
            "plies": args.plies,
            "proven": proven,
            "pv": pv,
            "nodes": prover.nodes,
            "exhausted": prover.nodes >= args.max_nodes,
        })
    text = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

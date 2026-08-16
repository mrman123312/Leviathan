"""Bounded exact minimax solver for small chess endgames.

The solver is intentionally conservative: it reports PROVEN only when the full
reachable game tree inside the requested limits is resolved. A node-budget hit
returns UNKNOWN, never a guessed WDL. This is primarily an Atlas exact-island
generator for positions too small to justify general search but where Syzygy
files may not be installed locally.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import chess


UNKNOWN = 2


@dataclass
class Budget:
    max_nodes: int
    max_plies: int
    nodes: int = 0
    aborted: bool = False


def terminal_value(board: chess.Board, root: chess.Color) -> int | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0
    return 1 if outcome.winner == root else -1


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    moves = list(board.legal_moves)
    moves.sort(
        key=lambda m: (
            bool(m.promotion),
            board.gives_check(m),
            board.is_capture(m),
            chess.square_rank(m.to_square) if board.turn == chess.WHITE else 7 - chess.square_rank(m.to_square),
        ),
        reverse=True,
    )
    return moves


def solve(board: chess.Board, root: chess.Color, budget: Budget, ply: int, pv: list[str]) -> tuple[int, list[str]]:
    budget.nodes += 1
    if budget.nodes > budget.max_nodes or ply > budget.max_plies:
        budget.aborted = True
        return UNKNOWN, []

    tv = terminal_value(board, root)
    if tv is not None:
        return tv, []

    moves = ordered_moves(board)
    maximizing = board.turn == root
    best = -2 if maximizing else 2
    best_line: list[str] = []

    for move in moves:
        board.push(move)
        value, child = solve(board, root, budget, ply + 1, pv)
        board.pop()
        if value == UNKNOWN:
            return UNKNOWN, []

        if maximizing:
            if value > best:
                best, best_line = value, [move.uci(), *child]
            if best == 1:
                # A winning root move is enough at a maximizing node.
                break
        else:
            if value < best:
                best, best_line = value, [move.uci(), *child]
            if best == -1:
                # One refutation is enough at an opponent node.
                break

    return best, best_line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True)
    ap.add_argument("--max-pieces", type=int, default=7)
    ap.add_argument("--max-nodes", type=int, default=2_000_000)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    board = chess.Board(args.fen)
    pieces = len(board.piece_map())
    if pieces > args.max_pieces:
        result = {"status": "UNKNOWN", "reason": "piece_limit", "pieces": pieces}
    else:
        root = board.turn
        budget = Budget(args.max_nodes, args.max_plies)
        value, line = solve(board, root, budget, 0, [])
        status = "UNKNOWN" if value == UNKNOWN or budget.aborted else "PROVEN"
        result = {
            "status": status,
            "wdl": None if status == "UNKNOWN" else value,
            "root_side": "white" if root == chess.WHITE else "black",
            "nodes": budget.nodes,
            "pieces": pieces,
            "pv": line,
            "fen": args.fen,
            "source": "leviathan-exact-endgame-v1",
        }

    text = json.dumps(result, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

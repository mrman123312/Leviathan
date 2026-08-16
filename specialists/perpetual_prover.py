"""Proof-sound, incomplete forced-perpetual specialist.

The root side is allowed only checking moves; the defender may play every legal
reply. Therefore a PROVEN result is a real forced draw-or-win through the
checking corridor, while failure does not imply no perpetual exists. Repetition
and fifty-move claims use python-chess's actual move stack.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import chess

SUCCESS = 1
FAIL = 0
UNKNOWN = -1


@dataclass
class Budget:
    max_nodes: int
    max_plies: int
    nodes: int = 0


def terminal(board: chess.Board, root: chess.Color) -> int | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None or outcome.winner == root:
        return SUCCESS
    return FAIL


def ordered(board: chess.Board, attacker_turn: bool) -> list[chess.Move]:
    moves = list(board.legal_moves)
    if attacker_turn:
        moves = [m for m in moves if board.gives_check(m)]
    moves.sort(
        key=lambda m: (board.is_capture(m), bool(m.promotion), board.gives_check(m)),
        reverse=True,
    )
    return moves


def prove(board: chess.Board, root: chess.Color, budget: Budget, ply: int) -> tuple[int, list[str]]:
    budget.nodes += 1
    if budget.nodes > budget.max_nodes or ply > budget.max_plies:
        return UNKNOWN, []

    tv = terminal(board, root)
    if tv is not None:
        return tv, []

    attacker_turn = board.turn == root
    moves = ordered(board, attacker_turn)
    if not moves:
        return FAIL, []

    if attacker_turn:
        saw_unknown = False
        for move in moves:
            board.push(move)
            result, child = prove(board, root, budget, ply + 1)
            board.pop()
            if result == SUCCESS:
                return SUCCESS, [move.uci(), *child]
            if result == UNKNOWN:
                saw_unknown = True
        return (UNKNOWN, []) if saw_unknown else (FAIL, [])

    # Defender node: every legal defense must remain inside the forced
    # draw-or-win corridor. One refutation disproves this checking proof.
    saw_unknown = False
    longest: list[str] = []
    for move in moves:
        board.push(move)
        result, child = prove(board, root, budget, ply + 1)
        board.pop()
        if result == FAIL:
            return FAIL, [move.uci(), *child]
        if result == UNKNOWN:
            saw_unknown = True
        elif len(child) + 1 > len(longest):
            longest = [move.uci(), *child]
    return (UNKNOWN, []) if saw_unknown else (SUCCESS, longest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True)
    ap.add_argument("--max-nodes", type=int, default=250_000)
    ap.add_argument("--max-plies", type=int, default=40)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    board = chess.Board(args.fen)
    root = board.turn
    budget = Budget(args.max_nodes, args.max_plies)
    result, line = prove(board, root, budget, 0)
    status = {SUCCESS: "PROVEN", FAIL: "NOT_PROVEN_IN_CHECKING_MODEL", UNKNOWN: "UNKNOWN"}[result]
    payload = {
        "status": status,
        "goal": "forced_draw_or_win_by_checks",
        "root_side": "white" if root == chess.WHITE else "black",
        "nodes": budget.nodes,
        "pv": line,
        "fen": args.fen,
        "verification": "all_defender_replies_expanded",
        "source": "leviathan-perpetual-prover-v1",
    }
    text = json.dumps(payload, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

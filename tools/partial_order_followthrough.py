#!/usr/bin/env python3
"""Test whether bounded commuting quiet pairs are reached by engine play.

The structural census can overstate useful headroom: two moves may commute yet
the engine may never want to play the other move after the first. This second
gate asks for each fully commuting pair whether the opponent selects the same
reply and whether the deferred move remains among the engine's top continuations.
It is diagnostic only and grants no pruning authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine


def ranked_moves(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    count: int,
) -> list[chess.Move]:
    engine.configure({"Clear Hash": None})
    result = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=min(count, board.legal_moves.count()),
        game=object(),
    )
    infos = result if isinstance(result, list) else [result]
    return [info["pv"][0] for info in infos if info.get("pv")]


def rank_of(moves: list[chess.Move], target: chess.Move) -> int | None:
    try:
        return moves.index(target) + 1
    except ValueError:
        return None


def position_after(board: chess.Board, *moves: chess.Move) -> chess.Board:
    result = board.copy(stack=False)
    for move in moves:
        if move not in result.legal_moves:
            raise ValueError(f"illegal path move {move.uci()} from {result.fen()}")
        result.push(move)
    return result


def inspect(
    engine: chess.engine.SimpleEngine,
    fen: str,
    a_uci: str,
    b_uci: str,
    nodes: int,
    top: int,
) -> dict[str, Any]:
    root = chess.Board(fen)
    a = chess.Move.from_uci(a_uci)
    b = chess.Move.from_uci(b_uci)
    after_a = position_after(root, a)
    after_b = position_after(root, b)
    reply_a = ranked_moves(engine, after_a, nodes, 1)[0]
    reply_b = ranked_moves(engine, after_b, nodes, 1)[0]
    continuations_a = ranked_moves(
        engine, position_after(root, a, reply_a), nodes, top
    )
    continuations_b = ranked_moves(
        engine, position_after(root, b, reply_b), nodes, top
    )
    b_rank_after_a = rank_of(continuations_a, b)
    a_rank_after_b = rank_of(continuations_b, a)
    same_reply = reply_a == reply_b
    return {
        "fen": fen,
        "a": a_uci,
        "b": b_uci,
        "reply_after_a": reply_a.uci(),
        "reply_after_b": reply_b.uci(),
        "same_best_reply": same_reply,
        "b_rank_after_a_reply": b_rank_after_a,
        "a_rank_after_b_reply": a_rank_after_b,
        "a_then_b_survives": b_rank_after_a is not None,
        "b_then_a_survives": a_rank_after_b is not None,
        "bidirectional_survival": b_rank_after_a is not None
        and a_rank_after_b is not None,
        "same_reply_bidirectional": same_reply
        and b_rank_after_a is not None
        and a_rank_after_b is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    census = json.loads(args.census.read_text(encoding="utf-8"))
    candidates = [
        (row["fen"], pair["a"], pair["b"])
        for row in census["rows"]
        for pair in row["pairs"]
        if pair["all_replies_commute"]
    ]
    if args.limit:
        candidates = candidates[: args.limit]

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 64})
    rows = []
    try:
        for index, (fen, a, b) in enumerate(candidates):
            row = inspect(engine, fen, a, b, args.nodes, args.top)
            rows.append(row)
            print(
                index + 1,
                row["same_best_reply"],
                row["b_rank_after_a_reply"],
                row["a_rank_after_b_reply"],
                flush=True,
            )
    finally:
        engine.quit()

    count = len(rows)
    totals = {
        "pairs": count,
        "same_best_reply": sum(row["same_best_reply"] for row in rows),
        "a_then_b_survives": sum(row["a_then_b_survives"] for row in rows),
        "b_then_a_survives": sum(row["b_then_a_survives"] for row in rows),
        "bidirectional_survival": sum(row["bidirectional_survival"] for row in rows),
        "same_reply_bidirectional": sum(
            row["same_reply_bidirectional"] for row in rows
        ),
    }
    payload = {
        "schema": "LV_PARTIAL_ORDER_FOLLOWTHROUGH_V1",
        "settings": {
            "census": str(args.census),
            "nodes": args.nodes,
            "top": args.top,
            "limit": args.limit,
        },
        "totals": totals,
        "rates": {
            key: value / count if count else None
            for key, value in totals.items()
            if key != "pairs"
        },
        "interpretation_guard": (
            "This measures realized relevance of bounded diamonds. It does not "
            "erase repetition history or prove that either move is forced."
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline census of exact three-ply quiet-move commutativity diamonds.

For same-side quiet moves A and B, test A-R-B versus B-R-A.  A board-state
diamond exists when the complete FENs match.  A much stronger bounded condition
requires the opponent's entire legal reply sets to match and every reply to
commute.  Neither condition erases repetition-history differences, so this tool
measures headroom only and grants no pruning authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine


def quiet_nonrights_move(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    return bool(
        piece
        and piece.piece_type not in (chess.PAWN, chess.KING, chess.ROOK)
        and not board.is_capture(move)
        and not board.gives_check(move)
        and not move.promotion
    )


def final_after(
    board: chess.Board, first: chess.Move, reply: chess.Move, second: chess.Move
) -> chess.Board | None:
    result = board.copy(stack=False)
    for move in (first, reply, second):
        if move not in result.legal_moves:
            return None
        result.push(move)
    return result


def same_complete_fen(left: chess.Board, right: chess.Board) -> bool:
    return left.fen(en_passant="fen") == right.fen(en_passant="fen")


def inspect_pair(
    board: chess.Board, first: chess.Move, second: chess.Move
) -> dict[str, Any]:
    after_first = board.copy(stack=False)
    after_first.push(first)
    after_second = board.copy(stack=False)
    after_second.push(second)
    replies_first = {move.uci(): move for move in after_first.legal_moves}
    replies_second = {move.uci(): move for move in after_second.legal_moves}
    common = sorted(set(replies_first) & set(replies_second))
    diamonds = []
    for uci in common:
        reply = chess.Move.from_uci(uci)
        left = final_after(board, first, reply, second)
        right = final_after(board, second, reply, first)
        if left is not None and right is not None and same_complete_fen(left, right):
            diamonds.append(uci)
    reply_sets_equal = set(replies_first) == set(replies_second)
    all_replies_commute = reply_sets_equal and len(diamonds) == len(replies_first)
    return {
        "a": first.uci(),
        "b": second.uci(),
        "replies_after_a": len(replies_first),
        "replies_after_b": len(replies_second),
        "common_replies": len(common),
        "diamond_replies": diamonds,
        "reply_sets_equal": reply_sets_equal,
        "all_replies_commute": all_replies_commute,
    }


def selected_root_moves(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    top_moves: int,
) -> list[chess.Move]:
    engine.configure({"Clear Hash": None})
    result = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=min(top_moves, board.legal_moves.count()),
    )
    infos = result if isinstance(result, list) else [result]
    moves = []
    for info in infos:
        if info.get("pv") and quiet_nonrights_move(board, info["pv"][0]):
            moves.append(info["pv"][0])
    return moves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--max-root-moves", type=int, default=24)
    parser.add_argument("--engine")
    parser.add_argument("--top-moves", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=20000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fens = [
        line.strip()
        for line in args.positions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.count]
    engine = chess.engine.SimpleEngine.popen_uci(args.engine) if args.engine else None
    if engine:
        engine.configure({"Threads": 1, "Hash": 64})
    rows = []
    totals = {
        "positions": 0,
        "root_move_pairs": 0,
        "pairs_with_diamond": 0,
        "total_diamond_replies": 0,
        "pairs_with_equal_reply_sets": 0,
        "fully_commuting_pairs": 0,
    }
    examples = []
    try:
        for index, fen in enumerate(fens):
            board = chess.Board(fen)
            if board.is_check() or board.ep_square is not None or board.halfmove_clock >= 70:
                continue
            if engine:
                moves = selected_root_moves(
                    engine, board, args.nodes, args.top_moves
                )
            else:
                moves = sorted(
                    (move for move in board.legal_moves if quiet_nonrights_move(board, move)),
                    key=lambda move: move.uci(),
                )[: args.max_root_moves]
            pairs = []
            for left_index, first in enumerate(moves):
                for second in moves[left_index + 1 :]:
                    result = inspect_pair(board, first, second)
                    pairs.append(result)
                    totals["root_move_pairs"] += 1
                    if result["diamond_replies"]:
                        totals["pairs_with_diamond"] += 1
                        totals["total_diamond_replies"] += len(result["diamond_replies"])
                        if len(examples) < 30:
                            examples.append(
                                {
                                    "fen": fen,
                                    "a": result["a"],
                                    "b": result["b"],
                                    "reply": result["diamond_replies"][0],
                                    "reply_sets_equal": result["reply_sets_equal"],
                                    "all_replies_commute": result["all_replies_commute"],
                                }
                            )
                    if result["reply_sets_equal"]:
                        totals["pairs_with_equal_reply_sets"] += 1
                    if result["all_replies_commute"]:
                        totals["fully_commuting_pairs"] += 1
            rows.append(
                {
                    "index": index,
                    "fen": fen,
                    "quiet_root_moves": [move.uci() for move in moves],
                    "pairs": pairs,
                }
            )
            totals["positions"] += 1
            print(
                totals["positions"],
                len(moves),
                sum(bool(pair["diamond_replies"]) for pair in pairs),
                sum(pair["all_replies_commute"] for pair in pairs),
                flush=True,
            )
    finally:
        if engine:
            engine.quit()

    pairs = totals["root_move_pairs"]
    payload = {
        "schema": "LV_PARTIAL_ORDER_CENSUS_V1",
        "settings": {
            "requested_positions": args.count,
            "max_root_moves": args.max_root_moves,
            "engine_filtered": bool(args.engine),
            "top_moves": args.top_moves if args.engine else None,
            "nodes": args.nodes if args.engine else None,
            "domain": "quiet non-pawn, non-king, non-rook, non-capture, non-check moves",
        },
        "totals": totals,
        "rates": {
            "pair_has_any_board_state_diamond": totals["pairs_with_diamond"] / pairs
            if pairs
            else None,
            "pair_has_equal_reply_sets": totals["pairs_with_equal_reply_sets"] / pairs
            if pairs
            else None,
            "pair_fully_commutes_for_all_replies": totals["fully_commuting_pairs"] / pairs
            if pairs
            else None,
        },
        "repetition_history_warning": "Even full-FEN diamonds reached through different paths have different repetition histories; no pruning proof is claimed.",
        "examples": examples,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

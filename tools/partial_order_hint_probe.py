#!/usr/bin/env python3
"""Measure the all-in cost of preloading a deferred commuting plan move."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import chess
import chess.engine


MATE_SCORE = 100000


def analyse(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    game: object,
    root_moves: list[chess.Move] | None = None,
) -> dict[str, Any]:
    info = engine.analyse(
        board,
        limit,
        game=game,
        root_moves=root_moves,
    )
    score = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
    return {
        "nodes": int(info.get("nodes", 0)),
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "score_cp": int(score if score is not None else 0),
        "pv": [move.uci() for move in info.get("pv", [])],
    }


def clear(engine: chess.engine.SimpleEngine) -> None:
    engine.configure({"Clear Hash": None})


def board_after(fen: str, *moves: str) -> chess.Board:
    board = chess.Board(fen)
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal path move {uci} from {board.fen()}")
        board.push(move)
    return board


def stable_signature(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "depth": result["depth"],
        "score_cp": result["score_cp"],
        "pv": result["pv"],
    }


def probe(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    target: chess.Move,
    target_rank: int,
    depth: int,
    hint_nodes: int,
    label: str,
) -> dict[str, Any]:
    clear(engine)
    cold_first = analyse(engine, board, chess.engine.Limit(depth=depth), object())

    clear(engine)
    hinted_game = object()
    hint = analyse(
        engine,
        board,
        chess.engine.Limit(nodes=hint_nodes),
        hinted_game,
        root_moves=[target],
    )
    full_after_hint = analyse(
        engine,
        board,
        chess.engine.Limit(depth=depth),
        hinted_game,
    )

    clear(engine)
    cold_second = analyse(engine, board, chess.engine.Limit(depth=depth), object())
    cold_nodes = math.sqrt(cold_first["nodes"] * cold_second["nodes"])
    all_in_hint_nodes = hint["nodes"] + full_after_hint["nodes"]
    controls_match = stable_signature(cold_first) == stable_signature(cold_second)
    hinted_matches = stable_signature(cold_first) == stable_signature(full_after_hint)
    return {
        "label": label,
        "fen": board.fen(en_passant="fen"),
        "target": target.uci(),
        "target_prior_rank": target_rank,
        "cold_first": cold_first,
        "hint": hint,
        "full_after_hint": full_after_hint,
        "cold_second": cold_second,
        "cold_controls_match": controls_match,
        "hinted_result_matches_cold": hinted_matches,
        "all_in_node_ratio": cold_nodes / all_in_hint_nodes
        if all_in_hint_nodes
        else None,
        "full_only_node_ratio": cold_nodes / full_after_hint["nodes"]
        if full_after_hint["nodes"]
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--followthrough", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--hint-nodes", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.followthrough.read_text(encoding="utf-8"))
    pairs = [row for row in source["rows"] if row["same_reply_bidirectional"]]
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 64})
    rows = []
    try:
        for pair_index, pair in enumerate(pairs):
            reply = pair["reply_after_a"]
            after_a = board_after(pair["fen"], pair["a"], reply)
            after_b = board_after(pair["fen"], pair["b"], reply)
            directions = (
                (
                    after_a,
                    chess.Move.from_uci(pair["b"]),
                    pair["b_rank_after_a_reply"],
                    f"{pair_index}:A-R-hint-B",
                ),
                (
                    after_b,
                    chess.Move.from_uci(pair["a"]),
                    pair["a_rank_after_b_reply"],
                    f"{pair_index}:B-R-hint-A",
                ),
            )
            for board, target, rank, label in directions:
                result = probe(
                    engine,
                    board,
                    target,
                    int(rank),
                    args.depth,
                    args.hint_nodes,
                    label,
                )
                rows.append(result)
                print(
                    label,
                    rank,
                    result["cold_controls_match"],
                    result["hinted_result_matches_cold"],
                    result["all_in_node_ratio"],
                    flush=True,
                )
    finally:
        engine.quit()

    valid = [
        row
        for row in rows
        if row["cold_controls_match"] and row["hinted_result_matches_cold"]
    ]
    opportunities = [row for row in valid if row["target_prior_rank"] > 1]

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        ratios = [row["all_in_node_ratio"] for row in selected]
        full_only = [row["full_only_node_ratio"] for row in selected]
        return {
            "count": len(selected),
            "median_all_in_node_ratio": statistics.median(ratios) if ratios else None,
            "mean_all_in_node_ratio": statistics.mean(ratios) if ratios else None,
            "all_in_wins": sum(value > 1.0 for value in ratios),
            "median_full_only_node_ratio": statistics.median(full_only)
            if full_only
            else None,
        }

    payload = {
        "schema": "LV_PARTIAL_ORDER_HINT_PROBE_V1",
        "settings": {
            "depth": args.depth,
            "hint_nodes": args.hint_nodes,
            "pairs": len(pairs),
            "directions": len(rows),
        },
        "stable_directions": summarize(valid),
        "stable_rank_gt_one_directions": summarize(opportunities),
        "unstable_controls": sum(not row["cold_controls_match"] for row in rows),
        "hint_changed_result": sum(
            row["cold_controls_match"] and not row["hinted_result_matches_cold"]
            for row in rows
        ),
        "interpretation_guard": (
            "The all-in ratio charges the restricted hint search. Full-only savings "
            "must never be reported without that setup cost."
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

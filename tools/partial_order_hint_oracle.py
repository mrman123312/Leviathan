#!/usr/bin/env python3
"""Equal-total-node deep-oracle screen for deferred commuting-plan hints."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import chess
import chess.engine


MATE_SCORE = 100000


def clear(engine: chess.engine.SimpleEngine) -> None:
    engine.configure({"Clear Hash": None})


def analyse(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    game: object,
    root_moves: list[chess.Move] | None = None,
) -> dict[str, Any]:
    info = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        game=game,
        root_moves=root_moves,
    )
    value = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
    return {
        "move": info["pv"][0].uci(),
        "score_cp": int(value if value is not None else 0),
        "nodes": int(info.get("nodes", 0)),
        "depth": int(info.get("depth", 0)),
        "pv": [move.uci() for move in info.get("pv", [])],
    }


def board_after(fen: str, *moves: str) -> chess.Board:
    board = chess.Board(fen)
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal path move {uci} from {board.fen()}")
        board.push(move)
    return board


def decision_cold(
    engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int
) -> dict[str, Any]:
    clear(engine)
    return analyse(engine, board, nodes, object())


def decision_hint(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    target: chess.Move,
    total_nodes: int,
    hint_nodes: int,
) -> dict[str, Any]:
    clear(engine)
    game = object()
    hint = analyse(engine, board, hint_nodes, game, root_moves=[target])
    full = analyse(engine, board, total_nodes - hint_nodes, game)
    return {**full, "hint": hint, "all_in_nodes": hint["nodes"] + full["nodes"]}


def oracle(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    moves: set[str],
    nodes: int,
) -> dict[str, Any]:
    clear(engine)
    best = analyse(engine, board, nodes, object())
    forced: dict[str, int] = {}
    for uci in sorted(moves | {best["move"]}):
        clear(engine)
        move = chess.Move.from_uci(uci)
        result = analyse(engine, board, nodes, object(), root_moves=[move])
        forced[uci] = result["score_cp"]
    best_move = max(forced, key=forced.get)
    best_score = forced[best_move]
    return {
        "unrestricted_best_move": best["move"],
        "best_move": best_move,
        "best_score_cp": best_score,
        "forced_scores_cp": forced,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--followthrough", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--decision-nodes", type=int, default=20000)
    parser.add_argument("--hint-nodes", type=int, default=1000)
    parser.add_argument("--oracle-nodes", type=int, default=300000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.hint_nodes < args.decision_nodes:
        raise SystemExit("hint nodes must be between zero and total decision nodes")

    source = json.loads(args.followthrough.read_text(encoding="utf-8"))
    pairs = [row for row in source["rows"] if row["same_reply_bidirectional"]]
    cases = []
    for pair_index, pair in enumerate(pairs):
        reply = pair["reply_after_a"]
        cases.extend(
            [
                {
                    "label": f"{pair_index}:A-R-hint-B",
                    "board": board_after(pair["fen"], pair["a"], reply),
                    "target": chess.Move.from_uci(pair["b"]),
                    "rank": pair["b_rank_after_a_reply"],
                },
                {
                    "label": f"{pair_index}:B-R-hint-A",
                    "board": board_after(pair["fen"], pair["b"], reply),
                    "target": chess.Move.from_uci(pair["a"]),
                    "rank": pair["a_rank_after_b_reply"],
                },
            ]
        )

    decision_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    oracle_engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    for engine in (decision_engine, oracle_engine):
        engine.configure({"Threads": 1, "Hash": 64})
    rows = []
    try:
        for index, case in enumerate(cases):
            board = case["board"]
            cold = decision_cold(decision_engine, board, args.decision_nodes)
            hinted = decision_hint(
                decision_engine,
                board,
                case["target"],
                args.decision_nodes,
                args.hint_nodes,
            )
            deep = oracle(
                oracle_engine,
                board,
                {cold["move"], hinted["move"]},
                args.oracle_nodes,
            )
            cold_regret = max(
                0, deep["best_score_cp"] - deep["forced_scores_cp"][cold["move"]]
            )
            hinted_regret = max(
                0,
                deep["best_score_cp"]
                - deep["forced_scores_cp"][hinted["move"]],
            )
            row = {
                "label": case["label"],
                "fen": board.fen(en_passant="fen"),
                "target": case["target"].uci(),
                "target_prior_rank": case["rank"],
                "cold": cold,
                "hinted": hinted,
                "oracle": deep,
                "cold_regret_cp": cold_regret,
                "hinted_regret_cp": hinted_regret,
                "regret_advantage_cp": cold_regret - hinted_regret,
            }
            rows.append(row)
            print(
                index + 1,
                cold["move"],
                hinted["move"],
                cold_regret,
                hinted_regret,
                flush=True,
            )
    finally:
        decision_engine.quit()
        oracle_engine.quit()

    advantages = [row["regret_advantage_cp"] for row in rows]
    payload = {
        "schema": "LV_PARTIAL_ORDER_HINT_ORACLE_V1",
        "settings": {
            "decision_nodes_each": args.decision_nodes,
            "hint_nodes_within_budget": args.hint_nodes,
            "full_nodes_after_hint": args.decision_nodes - args.hint_nodes,
            "oracle_nodes_per_forced_move": args.oracle_nodes,
            "directions": len(rows),
        },
        "summary": {
            "hint_wins": sum(value > 0 for value in advantages),
            "ties": sum(value == 0 for value in advantages),
            "hint_losses": sum(value < 0 for value in advantages),
            "mean_regret_advantage_cp": statistics.mean(advantages)
            if advantages
            else None,
            "median_regret_advantage_cp": statistics.median(advantages)
            if advantages
            else None,
            "same_decision": sum(
                row["cold"]["move"] == row["hinted"]["move"] for row in rows
            ),
            "cold_oracle_agreement": sum(
                row["cold"]["move"] == row["oracle"]["best_move"] for row in rows
            ),
            "hint_oracle_agreement": sum(
                row["hinted"]["move"] == row["oracle"]["best_move"] for row in rows
            ),
        },
        "interpretation_guard": (
            "Discovery-set mechanistic screen only. Any positive fragment requires "
            "a new corpus and games; any negative result can reject this hint form."
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

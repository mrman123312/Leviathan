"""Generate root-level MetaSearch / value-of-computation training data.

This is the first concrete transplant from the Cognitive Foundry idea of a
metacognitive controller that chooses an information-gathering operation.

For each FEN we search at several low node budgets and once at a much deeper
teacher budget. We then re-search each distinct low-budget selected move at the
teacher budget so we can measure *decision regret*: how much minimax value was
lost because the shallow computation stopped on that move.

The live engine is not modified. This dataset is intended to train/adjudicate a
future stop/deepen controller before it receives any authority over search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine


def score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    value = score.pov(turn).score(mate_score=100_000)
    return 0 if value is None else int(value)


def pv_uci(info: dict[str, Any], limit: int = 12) -> list[str]:
    return [move.uci() for move in info.get("pv", ())[:limit]]


def prefix_match(left: list[str], right: list[str]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def summarize(info: dict[str, Any], turn: chess.Color) -> dict[str, Any]:
    pv = pv_uci(info)
    return {
        "best_move": pv[0] if pv else None,
        "score_cp": score_to_cp(info["score"], turn),
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "nodes": int(info.get("nodes", 0)),
        "nps": int(info.get("nps", 0)),
        "hashfull": int(info.get("hashfull", 0)),
        "pv": pv,
    }


def analyse_multipv(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    multipv: int,
) -> list[dict[str, Any]]:
    legal = board.legal_moves.count()
    if legal == 0:
        return []
    raw = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=min(multipv, legal),
    )
    if isinstance(raw, dict):
        return [raw]
    return list(raw)


def root_gap(rows: list[dict[str, Any]], turn: chess.Color) -> int | None:
    if len(rows) < 2:
        return None
    return score_to_cp(rows[0]["score"], turn) - score_to_cp(rows[1]["score"], turn)


def parse_budgets(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or values[0] <= 0:
        raise ValueError("--budgets must contain positive node counts")
    return values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--fens", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budgets", default="1000,4000,16000")
    ap.add_argument("--teacher-nodes", type=int, default=200_000)
    ap.add_argument("--multipv", type=int, default=3)
    ap.add_argument("--regret-threshold-cp", type=int, default=25)
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash-mb", type=int, default=64)
    args = ap.parse_args()

    budgets = parse_budgets(args.budgets)
    if args.teacher_nodes <= budgets[-1]:
        raise ValueError("--teacher-nodes must exceed every low search budget")
    if args.multipv <= 0:
        raise ValueError("--multipv must be positive")

    fens = [x.strip() for x in args.fens.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.max_positions:
        fens = fens[: args.max_positions]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    try:
        available = engine.options
        config: dict[str, Any] = {}
        if "Threads" in available:
            config["Threads"] = args.threads
        if "Hash" in available:
            config["Hash"] = args.hash_mb
        if config:
            engine.configure(config)

        with args.out.open("w", encoding="utf-8") as out:
            for position_id, fen in enumerate(fens):
                board = chess.Board(fen)
                if board.is_game_over():
                    continue
                turn = board.turn

                # Clear mutable engine state between positions when supported so
                # examples do not accidentally depend on the previous FEN.
                if "Clear Hash" in available:
                    engine.configure({"Clear Hash": None})

                low_raw: dict[int, list[dict[str, Any]]] = {}
                for budget in budgets:
                    low_raw[budget] = analyse_multipv(engine, board, budget, args.multipv)

                teacher_raw = analyse_multipv(engine, board, args.teacher_nodes, args.multipv)
                if not teacher_raw:
                    continue
                teacher = summarize(teacher_raw[0], turn)
                teacher_best = teacher["best_move"]
                teacher_gap = root_gap(teacher_raw, turn)

                # A low-budget move can look good merely because it was searched
                # shallowly. Re-evaluate every distinct selected move with the
                # teacher budget to obtain counterfactual value on comparable
                # compute.
                chosen_moves = {
                    summarize(rows[0], turn)["best_move"]
                    for rows in low_raw.values()
                    if rows and summarize(rows[0], turn)["best_move"] is not None
                }
                counterfactual: dict[str, int] = {}
                for move_uci in sorted(chosen_moves):
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        continue
                    info = engine.analyse(
                        board,
                        chess.engine.Limit(nodes=args.teacher_nodes),
                        root_moves=[move],
                    )
                    counterfactual[move_uci] = score_to_cp(info["score"], turn)

                teacher_best_cp = int(teacher["score_cp"])
                for budget in budgets:
                    rows = low_raw[budget]
                    if not rows:
                        continue
                    low = summarize(rows[0], turn)
                    selected = low["best_move"]
                    selected_teacher_cp = (
                        counterfactual.get(selected, teacher_best_cp)
                        if selected is not None
                        else teacher_best_cp
                    )
                    regret_cp = max(0, teacher_best_cp - selected_teacher_cp)
                    low_gap = root_gap(rows, turn)
                    changed = selected != teacher_best

                    record = {
                        "position_id": str(position_id),
                        "fen": fen,
                        "budget_nodes": budget,
                        "teacher_nodes": args.teacher_nodes,
                        "low": low,
                        "teacher": teacher,
                        "low_root_gap_cp": low_gap,
                        "teacher_root_gap_cp": teacher_gap,
                        "selected_move_teacher_cp": selected_teacher_cp,
                        "regret_cp": regret_cp,
                        "decision_changed": changed,
                        "pv_prefix_match_plies": prefix_match(low["pv"], teacher["pv"]),
                        "need_more_search": bool(
                            changed or regret_cp >= args.regret_threshold_cp
                        ),
                    }
                    out.write(json.dumps(record, separators=(",", ":")) + "\n")

                print(
                    f"position={position_id + 1}/{len(fens)} "
                    f"teacher={teacher_best} budgets={budgets}"
                )
    finally:
        engine.quit()


if __name__ == "__main__":
    main()

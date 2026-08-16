#!/usr/bin/env python3
"""Direct candidate/reference disagreement test with a deeper third-engine oracle."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

import chess
import chess.engine


MATE = 100000


def load_options(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("options JSON must be an object")
    return data


def configure(engine: chess.engine.SimpleEngine, options: dict[str, Any]) -> None:
    unknown = sorted(set(options) - set(engine.options))
    if unknown:
        raise ValueError(f"engine does not expose options: {unknown}")
    if options:
        engine.configure(options)


def clear(engine: chess.engine.SimpleEngine) -> None:
    if "Clear Hash" in engine.options:
        engine.configure({"Clear Hash": None})


def cp(info: dict[str, Any], pov: chess.Color) -> int:
    value = info["score"].pov(pov).score(mate_score=MATE)
    return int(value if value is not None else 0)


def analyse(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    multipv: int = 1,
    root_moves: list[chess.Move] | None = None,
) -> list[dict[str, Any]]:
    result = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=multipv,
        root_moves=root_moves,
    )
    return result if isinstance(result, list) else [result]


def selected_move(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int) -> str:
    clear(engine)
    info = analyse(engine, board, nodes)[0]
    return info["pv"][0].uci()


def forced_score(
    engine: chess.engine.SimpleEngine, board: chess.Board, move_uci: str, nodes: int
) -> int:
    clear(engine)
    move = chess.Move.from_uci(move_uci)
    return cp(analyse(engine, board, nodes, root_moves=[move])[0], board.turn)


def deep_oracle(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    candidate_move: str,
    reference_move: str,
) -> tuple[dict[str, int], str, int]:
    clear(engine)
    discovery = analyse(engine, board, nodes)[0]
    oracle_move = discovery["pv"][0].uci()
    moves = sorted({candidate_move, reference_move, oracle_move})
    values = {move: forced_score(engine, board, move, nodes) for move in moves}
    best = max(values.values())
    return values, oracle_move, best


def generation_step(
    engine: chess.engine.SimpleEngine, board: chess.Board, rng: random.Random
) -> chess.Move | None:
    infos = analyse(engine, board, 1000, min(5, board.legal_moves.count()))
    choices = []
    for info in infos:
        if info.get("pv"):
            choices.append((info["pv"][0], cp(info, board.turn)))
    if not choices:
        return None
    best = choices[0][1]
    viable = [item for item in choices if best - item[1] <= 110] or choices[:1]
    return rng.choices(
        [move for move, _ in viable], weights=list(range(len(viable), 0, -1)), k=1
    )[0]


def features(board: chess.Board) -> dict[str, int | bool]:
    return {
        "legal_moves": board.legal_moves.count(),
        "in_check": board.is_check(),
        "pieces": len(board.piece_map()),
        "pawns": len(board.pieces(chess.PAWN, chess.WHITE))
        + len(board.pieces(chess.PAWN, chess.BLACK)),
        "queens": len(board.pieces(chess.QUEEN, chess.WHITE))
        + len(board.pieces(chess.QUEEN, chess.BLACK)),
        "halfmove": board.halfmove_clock,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--options-candidate", type=Path)
    parser.add_argument("--options-reference", type=Path)
    parser.add_argument("--disagreements", type=int, default=40)
    parser.add_argument("--max-positions", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=773031)
    parser.add_argument("--shallow", type=int, default=24000)
    parser.add_argument("--deep", type=int, default=300000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidate = chess.engine.SimpleEngine.popen_uci(args.candidate)
    reference = chess.engine.SimpleEngine.popen_uci(args.reference)
    oracle = chess.engine.SimpleEngine.popen_uci(args.oracle)
    generator = chess.engine.SimpleEngine.popen_uci(args.oracle)
    base = {"Threads": 1, "Hash": 64}
    configure(candidate, base | load_options(args.options_candidate))
    configure(reference, base | load_options(args.options_reference))
    configure(oracle, base)
    configure(generator, base)

    rng = random.Random(args.seed)
    rows = []
    tested = 0
    seen = set()
    try:
        while len(rows) < args.disagreements and tested < args.max_positions:
            board = chess.Board()
            for _ in range(rng.randint(8, 34)):
                if board.is_game_over(claim_draw=True):
                    break
                move = generation_step(generator, board, rng)
                if move is None:
                    break
                board.push(move)
            if board.is_game_over(claim_draw=True):
                continue
            key = " ".join(board.fen().split()[:4])
            if key in seen:
                continue
            seen.add(key)
            tested += 1

            candidate_move = selected_move(candidate, board, args.shallow)
            reference_move = selected_move(reference, board, args.shallow)
            if candidate_move == reference_move:
                continue

            values, oracle_move, best = deep_oracle(
                oracle, board, args.deep, candidate_move, reference_move
            )
            candidate_regret = best - values[candidate_move]
            reference_regret = best - values[reference_move]
            delta = reference_regret - candidate_regret
            row = {
                "fen": board.fen(),
                "candidate_move": candidate_move,
                "reference_move": reference_move,
                "oracle_move": oracle_move,
                "oracle_values_cp": values,
                "candidate_regret": candidate_regret,
                "reference_regret": reference_regret,
                "regret_advantage_cp": delta,
                "winner": "candidate" if delta > 0 else ("reference" if delta < 0 else "tie"),
                "features": features(board),
            }
            rows.append(row)
            print(len(rows), row["winner"], delta, candidate_move, reference_move)
    finally:
        for engine in (candidate, reference, oracle, generator):
            engine.quit()

    wins = sum(row["winner"] == "candidate" for row in rows)
    losses = sum(row["winner"] == "reference" for row in rows)
    ties = len(rows) - wins - losses
    deltas = [row["regret_advantage_cp"] for row in rows]
    payload = {
        "schema": "LV_DISAGREEMENT_ORACLE_V2",
        "tested_positions": tested,
        "disagreements": len(rows),
        "candidate_wins": wins,
        "reference_wins": losses,
        "ties": ties,
        "candidate_nonloss_rate": (wins + ties) / len(rows) if rows else None,
        "candidate_win_rate_ex_ties": wins / (wins + losses) if wins + losses else None,
        "mean_regret_advantage_cp": statistics.mean(deltas) if deltas else None,
        "median_regret_advantage_cp": statistics.median(deltas) if deltas else None,
        "settings": {
            "seed": args.seed,
            "shallow_nodes": args.shallow,
            "deep_nodes_per_forced_move": args.deep,
            "target_disagreements": args.disagreements,
            "max_positions": args.max_positions,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    if len(rows) < args.disagreements:
        raise SystemExit("insufficient disagreements for the predeclared sample")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare search candidates on one fixed generated corpus with a deep oracle.

All candidates see the same positions.  Oracle work is performed when any
candidate disagrees with the reference, avoiding the selection bias caused by
stopping after a candidate-specific number of disagreements.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import chess
import chess.engine


MATE_SCORE = 100000


def parse_engines(items: list[str]) -> dict[str, str]:
    engines: dict[str, str] = {}
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in engines:
            raise SystemExit(f"invalid or duplicate engine mapping: {item}")
        engines[name] = path
    return engines


def load_options(path: Path) -> dict[str, Any]:
    options = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(options, dict):
        raise SystemExit("options JSON must be an object")
    return options


def configure(engine: chess.engine.SimpleEngine, options: dict[str, Any]) -> None:
    unknown = sorted(set(options) - set(engine.options))
    if unknown:
        raise SystemExit(f"engine does not expose options: {unknown}")
    engine.configure(options)


def clear(engine: chess.engine.SimpleEngine) -> None:
    if "Clear Hash" in engine.options:
        engine.configure({"Clear Hash": None})


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


def score_cp(info: dict[str, Any], pov: chess.Color) -> int:
    value = info["score"].pov(pov).score(mate_score=MATE_SCORE)
    return int(value if value is not None else 0)


def selected_move(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int) -> str:
    clear(engine)
    info = analyse(engine, board, nodes)[0]
    return info["pv"][0].uci()


def forced_score(
    engine: chess.engine.SimpleEngine, board: chess.Board, move_uci: str, nodes: int
) -> int:
    clear(engine)
    move = chess.Move.from_uci(move_uci)
    info = analyse(engine, board, nodes, root_moves=[move])[0]
    return score_cp(info, board.turn)


def generation_step(
    engine: chess.engine.SimpleEngine, board: chess.Board, rng: random.Random
) -> chess.Move | None:
    infos = analyse(engine, board, 1000, min(5, board.legal_moves.count()))
    choices = [(info["pv"][0], score_cp(info, board.turn)) for info in infos if info.get("pv")]
    if not choices:
        return None
    best = choices[0][1]
    viable = [item for item in choices if best - item[1] <= 110] or choices[:1]
    return rng.choices(
        [move for move, _ in viable],
        weights=list(range(len(viable), 0, -1)),
        k=1,
    )[0]


def generated_position(
    generator: chess.engine.SimpleEngine, rng: random.Random
) -> chess.Board | None:
    board = chess.Board()
    for _ in range(rng.randint(8, 34)):
        if board.is_game_over(claim_draw=True):
            return None
        move = generation_step(generator, board, rng)
        if move is None:
            return None
        board.push(move)
    return None if board.is_game_over(claim_draw=True) else board


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


def sign_test_two_sided(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if not trials:
        return None
    tail = sum(math.comb(trials, k) for k in range(min(wins, losses) + 1)) / (2**trials)
    return min(1.0, 2.0 * tail)


def bootstrap_mean_ci(values: list[int], seed: int, samples: int = 30000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    size = len(values)
    means = [
        statistics.mean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    means.sort()
    return [means[int(samples * 0.025)], means[int(samples * 0.975)]]


def summarize(rows: list[dict[str, Any]], name: str, seed: int) -> dict[str, Any]:
    disagreements = [row for row in rows if row["moves"][name] != row["moves"]["reference"]]
    deltas = [row["outcomes"][name]["regret_advantage_cp"] for row in disagreements]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    ties = len(deltas) - wins - losses
    return {
        "positions": len(rows),
        "agreements": len(rows) - len(disagreements),
        "disagreements": len(disagreements),
        "agreement_rate": (len(rows) - len(disagreements)) / len(rows),
        "candidate_wins": wins,
        "reference_wins": losses,
        "ties": ties,
        "candidate_win_rate_ex_ties": wins / (wins + losses) if wins + losses else None,
        "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
        "mean_regret_advantage_cp": statistics.mean(deltas) if deltas else None,
        "median_regret_advantage_cp": statistics.median(deltas) if deltas else None,
        "bootstrap_mean_regret_95pct_ci": bootstrap_mean_ci(deltas, seed),
        "whole_corpus_mean_regret_advantage_cp": sum(deltas) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", action="append", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=240)
    parser.add_argument("--seed", type=int, default=991073)
    parser.add_argument("--shallow", type=int, default=24000)
    parser.add_argument("--deep", type=int, default=300000)
    parser.add_argument("--min-disagreements", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = parse_engines(args.engine)
    if "reference" not in paths or len(paths) < 2:
        raise SystemExit("one engine must be named reference and at least one candidate is required")
    candidate_names = [name for name in paths if name != "reference"]
    options = {"Threads": 1, "Hash": 64} | load_options(args.options)
    engines = {name: chess.engine.SimpleEngine.popen_uci(path) for name, path in paths.items()}
    oracle = chess.engine.SimpleEngine.popen_uci(args.oracle)
    generator = chess.engine.SimpleEngine.popen_uci(args.oracle)
    for engine in engines.values():
        configure(engine, options)
    configure(oracle, {"Threads": 1, "Hash": 64})
    configure(generator, {"Threads": 1, "Hash": 64})

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    try:
        while len(rows) < args.positions and attempts < args.positions * 10:
            attempts += 1
            board = generated_position(generator, rng)
            if board is None:
                continue
            key = " ".join(board.fen().split()[:4])
            if key in seen:
                continue
            seen.add(key)
            moves = {
                name: selected_move(engine, board, args.shallow)
                for name, engine in engines.items()
            }
            unique = set(moves.values())
            row: dict[str, Any] = {
                "index": len(rows),
                "fen": board.fen(),
                "features": features(board),
                "moves": moves,
                "outcomes": {},
            }
            if len(unique) > 1:
                clear(oracle)
                oracle_move = analyse(oracle, board, args.deep)[0]["pv"][0].uci()
                unique.add(oracle_move)
                values = {
                    move: forced_score(oracle, board, move, args.deep)
                    for move in sorted(unique)
                }
                best = max(values.values())
                reference_regret = best - values[moves["reference"]]
                row["oracle_move"] = oracle_move
                row["oracle_values_cp"] = values
                row["reference_regret_cp"] = reference_regret
                for name in candidate_names:
                    regret = best - values[moves[name]]
                    delta = reference_regret - regret
                    row["outcomes"][name] = {
                        "regret_cp": regret,
                        "regret_advantage_cp": delta,
                        "winner": "candidate" if delta > 0 else ("reference" if delta < 0 else "tie"),
                    }
            else:
                for name in candidate_names:
                    row["outcomes"][name] = {
                        "regret_cp": None,
                        "regret_advantage_cp": 0,
                        "winner": "tie",
                    }
            rows.append(row)
            if len(rows) % 10 == 0:
                counts = {
                    name: sum(r["moves"][name] != r["moves"]["reference"] for r in rows)
                    for name in candidate_names
                }
                print(len(rows), counts, flush=True)
    finally:
        for engine in [*engines.values(), oracle, generator]:
            engine.quit()

    if len(rows) != args.positions:
        raise SystemExit(f"generated only {len(rows)} of {args.positions} predeclared positions")
    summaries = {
        name: summarize(rows, name, args.seed + 1000 + index)
        for index, name in enumerate(candidate_names)
    }
    payload = {
        "schema": "LV_MULTI_CANDIDATE_REGRET_PANEL_V1",
        "settings": {
            "positions": args.positions,
            "generation_attempts": attempts,
            "seed": args.seed,
            "shallow_nodes": args.shallow,
            "deep_nodes_per_search": args.deep,
            "min_disagreements": args.min_disagreements,
        },
        "summaries": summaries,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"schema": payload["schema"], "settings": payload["settings"], "summaries": summaries}, indent=2))
    sparse = {name: value["disagreements"] for name, value in summaries.items() if value["disagreements"] < args.min_disagreements}
    if sparse:
        raise SystemExit(f"insufficient disagreements for predeclared candidates: {sparse}")


if __name__ == "__main__":
    main()

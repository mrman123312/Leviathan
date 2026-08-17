#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

OPENINGS = [
    ("Open Game", ["e2e4", "e7e5", "g1f3", "b8c6"]),
    ("Queen's Gambit", ["d2d4", "d7d5", "c2c4", "e7e6"]),
    ("English", ["c2c4", "e7e5", "b1c3", "g8f6"]),
    ("Reti", ["g1f3", "d7d5", "g2g3", "c7c5"]),
    ("Sicilian", ["e2e4", "c7c5", "g1f3", "d7d6"]),
    ("King's Indian", ["d2d4", "g8f6", "c2c4", "g7g6"]),
    ("French", ["e2e4", "e7e6", "d2d4", "d7d5"]),
    ("Caro-Kann", ["e2e4", "c7c6", "d2d4", "d7d5"]),
]


def opening_board(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal opening move {uci} in {board.fen()}")
        board.push(move)
    return board


def score_from_result(result: str, leviathan_color: chess.Color) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if leviathan_color == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if leviathan_color == chess.BLACK else 0.0
    return 0.5


def approx_elo(score: float) -> float | None:
    if score <= 0.0 or score >= 1.0:
        return None
    return 400.0 * math.log10(score / (1.0 - score))


def play_game(
    leviathan: chess.engine.SimpleEngine,
    stockfish: chess.engine.SimpleEngine,
    opening_name: str,
    opening_moves: list[str],
    leviathan_color: chess.Color,
    move_time: float,
    max_plies: int,
    game_index: int,
) -> tuple[chess.pgn.Game, dict]:
    board = opening_board(opening_moves)
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Leviathan v4 vs Stockfish pinned control"
    game.headers["Round"] = str(game_index)
    game.headers["Opening"] = opening_name
    game.headers["LeviathanColor"] = "White" if leviathan_color == chess.WHITE else "Black"
    game.headers["MoveTimeMs"] = str(round(move_time * 1000))
    game.headers["White"] = "Leviathan Rewrite v4" if leviathan_color == chess.WHITE else "Stockfish"
    game.headers["Black"] = "Leviathan Rewrite v4" if leviathan_color == chess.BLACK else "Stockfish"

    node = game.end()
    timings = {"leviathan_ms": [], "stockfish_ms": []}
    error = None
    termination = None

    for _ in range(max_plies):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            termination = outcome.termination.name
            break

        engine = leviathan if board.turn == leviathan_color else stockfish
        key = "leviathan_ms" if board.turn == leviathan_color else "stockfish_ms"
        started = time.perf_counter()
        try:
            response = engine.play(board, chess.engine.Limit(time=move_time), game=game_index)
        except Exception as exc:  # preserve crash/timeout as match evidence
            error = f"{type(exc).__name__}: {exc}"
            termination = "engine-error"
            break
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings[key].append(elapsed_ms)
        if response.move is None or response.move not in board.legal_moves:
            error = f"invalid/no move: {response.move}"
            termination = "invalid-move"
            break
        board.push(response.move)
        node = node.add_variation(response.move)
    else:
        termination = "ply-limit-draw"

    outcome = board.outcome(claim_draw=True)
    if error:
        failed_is_leviathan = board.turn == leviathan_color
        result = "0-1" if (failed_is_leviathan and leviathan_color == chess.WHITE) or (not failed_is_leviathan and leviathan_color == chess.BLACK) else "1-0"
    elif outcome is not None:
        result = outcome.result()
    else:
        result = "1/2-1/2"

    game.headers["Result"] = result
    game.headers["Termination"] = termination or "unknown"

    record = {
        "game": game_index,
        "opening": opening_name,
        "opening_moves": opening_moves,
        "leviathan_color": "white" if leviathan_color == chess.WHITE else "black",
        "result": result,
        "leviathan_score": score_from_result(result, leviathan_color),
        "termination": termination,
        "error": error,
        "plies_played_after_opening": len(board.move_stack) - len(opening_moves),
        "final_fen": board.fen(),
        "timing": {
            "leviathan_moves": len(timings["leviathan_ms"]),
            "stockfish_moves": len(timings["stockfish_ms"]),
            "leviathan_mean_ms": sum(timings["leviathan_ms"]) / len(timings["leviathan_ms"]) if timings["leviathan_ms"] else None,
            "stockfish_mean_ms": sum(timings["stockfish_ms"]) / len(timings["stockfish_ms"]) if timings["stockfish_ms"] else None,
        },
    }
    return game, record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leviathan", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--move-time-ms", type=int, default=50)
    ap.add_argument("--max-plies", type=int, default=180)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    move_time = args.move_time_ms / 1000.0

    records: list[dict] = []
    games: list[chess.pgn.Game] = []
    leviathan = chess.engine.SimpleEngine.popen_uci(args.leviathan, timeout=20.0)
    stockfish = chess.engine.SimpleEngine.popen_uci(args.stockfish, timeout=20.0)
    try:
        sf_opts = {}
        if "Threads" in stockfish.options:
            sf_opts["Threads"] = 1
        if "Hash" in stockfish.options:
            sf_opts["Hash"] = 64
        if sf_opts:
            stockfish.configure(sf_opts)

        game_index = 0
        for opening_name, opening_moves in OPENINGS:
            for leviathan_color in (chess.WHITE, chess.BLACK):
                game_index += 1
                game, record = play_game(
                    leviathan, stockfish, opening_name, opening_moves,
                    leviathan_color, move_time, args.max_plies, game_index,
                )
                games.append(game)
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        leviathan.quit()
        stockfish.quit()

    with (out / "games.pgn").open("w", encoding="utf-8") as f:
        for game in games:
            print(game, file=f, end="\n\n")

    wins = sum(1 for r in records if r["leviathan_score"] == 1.0)
    draws = sum(1 for r in records if r["leviathan_score"] == 0.5)
    losses = sum(1 for r in records if r["leviathan_score"] == 0.0)
    score = (wins + 0.5 * draws) / len(records)
    errors = [r for r in records if r["error"]]
    lev_times = [r["timing"]["leviathan_mean_ms"] for r in records if r["timing"]["leviathan_mean_ms"] is not None]
    sf_times = [r["timing"]["stockfish_mean_ms"] for r in records if r["timing"]["stockfish_mean_ms"] is not None]

    summary = {
        "games": len(records),
        "paired_openings": len(OPENINGS),
        "move_time_ms_per_engine": args.move_time_ms,
        "max_plies_after_opening": args.max_plies,
        "leviathan": {"wins": wins, "draws": draws, "losses": losses, "score_fraction": score},
        "approx_logistic_elo_difference": approx_elo(score),
        "engine_errors": errors,
        "mean_observed_move_ms": {
            "leviathan_game_means": sum(lev_times) / len(lev_times) if lev_times else None,
            "stockfish_game_means": sum(sf_times) / len(sf_times) if sf_times else None,
        },
        "note": "Approximate Elo is undefined at a 0% or 100% score and is not a statistically reliable rating estimate for this small diagnostic match.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "games.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
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

INFO = chess.engine.INFO_BASIC | chess.engine.INFO_SCORE | chess.engine.INFO_PV


def board_from_opening(moves: list[str]) -> chess.Board:
    b = chess.Board()
    for u in moves:
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            raise RuntimeError(f"illegal seed move {u}: {b.fen()}")
        b.push(m)
    return b


def configure(engine: chess.engine.SimpleEngine, fundamentals: bool) -> dict:
    opts = {}
    def set_if(name: str, value):
        if name in engine.options:
            opts[name] = value
    set_if("Threads", 1)
    set_if("Hash", 64)
    if fundamentals:
        set_if("Leviathan Fundamentals", True)
        set_if("Leviathan Fundamentals Authority", 1)
        # The old competitive Fundamentals control explicitly disabled this
        # unrelated high-authority organ. Keep that contract when the option exists.
        set_if("Leviathan Quiet Overdrive", 0)
    if opts:
        engine.configure(opts)
    return opts


def result_score(result: str, candidate_color: chess.Color) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if candidate_color == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if candidate_color == chess.BLACK else 0.0
    raise RuntimeError(result)


def score_to_elo(s: float) -> float | None:
    if s <= 0.0 or s >= 1.0:
        return None
    return 400.0 * math.log10(s / (1.0 - s))


def safe_mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def play_one(candidate, opponent, opening_name, opening_moves, candidate_color, move_ms, max_plies, idx):
    b = board_from_opening(opening_moves)
    game = chess.pgn.Game.from_board(b)
    game.headers["Event"] = "Fundamentals Ultra candidate screen"
    game.headers["Round"] = str(idx)
    game.headers["Opening"] = opening_name
    game.headers["CandidateColor"] = "White" if candidate_color else "Black"
    game.headers["MoveTimeMs"] = str(move_ms)
    node = game.end()

    cand_times: list[float] = []
    opp_times: list[float] = []
    cand_depths: list[float] = []
    opp_depths: list[float] = []
    cand_nodes: list[float] = []
    opp_nodes: list[float] = []
    error = None
    termination = None

    for _ in range(max_plies):
        outcome = b.outcome(claim_draw=True)
        if outcome:
            termination = outcome.termination.name
            break
        cand_turn = b.turn == candidate_color
        eng = candidate if cand_turn else opponent
        t0 = time.perf_counter()
        try:
            r = eng.play(b, chess.engine.Limit(time=move_ms / 1000.0), info=INFO)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            termination = "ENGINE_ERROR"
            break
        dt = (time.perf_counter() - t0) * 1000.0
        info = r.info or {}
        if cand_turn:
            cand_times.append(dt)
            if isinstance(info.get("depth"), int): cand_depths.append(float(info["depth"]))
            if isinstance(info.get("nodes"), int): cand_nodes.append(float(info["nodes"]))
        else:
            opp_times.append(dt)
            if isinstance(info.get("depth"), int): opp_depths.append(float(info["depth"]))
            if isinstance(info.get("nodes"), int): opp_nodes.append(float(info["nodes"]))
        if r.move is None or r.move not in b.legal_moves:
            error = f"invalid/no move: {r.move}"
            termination = "INVALID_MOVE"
            break
        b.push(r.move)
        node = node.add_variation(r.move)
    else:
        termination = "PLY_LIMIT_DRAW"

    outcome = b.outcome(claim_draw=True)
    if error:
        failed_candidate = b.turn == candidate_color
        if failed_candidate:
            result = "0-1" if candidate_color == chess.WHITE else "1-0"
        else:
            result = "1-0" if candidate_color == chess.WHITE else "0-1"
    elif outcome:
        result = outcome.result()
    else:
        result = "1/2-1/2"

    game.headers["Result"] = result
    game.headers["Termination"] = termination or "unknown"
    return game, {
        "game": idx,
        "opening": opening_name,
        "candidate_color": "white" if candidate_color == chess.WHITE else "black",
        "result": result,
        "candidate_score": result_score(result, candidate_color),
        "termination": termination,
        "error": error,
        "plies_after_seed": len(b.move_stack) - len(opening_moves),
        "candidate_mean_ms": safe_mean(cand_times),
        "opponent_mean_ms": safe_mean(opp_times),
        "candidate_mean_depth": safe_mean(cand_depths),
        "opponent_mean_depth": safe_mean(opp_depths),
        "candidate_mean_nodes": safe_mean(cand_nodes),
        "opponent_mean_nodes": safe_mean(opp_nodes),
        "final_fen": b.fen(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--candidate-label", required=True)
    ap.add_argument("--opponent-label", default="Stockfish")
    ap.add_argument("--candidate-fundamentals", action="store_true")
    ap.add_argument("--opponent-fundamentals", action="store_true")
    ap.add_argument("--move-time-ms", type=int, default=50)
    ap.add_argument("--max-plies", type=int, default=180)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate = chess.engine.SimpleEngine.popen_uci(args.candidate, timeout=30.0)
    opponent = chess.engine.SimpleEngine.popen_uci(args.opponent, timeout=30.0)
    records = []
    games = []
    try:
        cand_opts = configure(candidate, args.candidate_fundamentals)
        opp_opts = configure(opponent, args.opponent_fundamentals)
        idx = 0
        for opening_name, opening_moves in OPENINGS:
            for color in (chess.WHITE, chess.BLACK):
                idx += 1
                game, record = play_one(
                    candidate, opponent, opening_name, opening_moves,
                    color, args.move_time_ms, args.max_plies, idx,
                )
                game.headers["White"] = args.candidate_label if color == chess.WHITE else args.opponent_label
                game.headers["Black"] = args.candidate_label if color == chess.BLACK else args.opponent_label
                games.append(game)
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                candidate.ucinewgame()
                opponent.ucinewgame()
    finally:
        candidate.quit()
        opponent.quit()

    with (out / "games.pgn").open("w", encoding="utf-8") as f:
        for g in games:
            print(g, file=f, end="\n\n")
    (out / "games.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")

    wins = sum(r["candidate_score"] == 1.0 for r in records)
    draws = sum(r["candidate_score"] == 0.5 for r in records)
    losses = sum(r["candidate_score"] == 0.0 for r in records)
    score = (wins + 0.5 * draws) / len(records)
    errs = [r for r in records if r["error"]]
    summary = {
        "candidate": args.candidate_label,
        "opponent": args.opponent_label,
        "games": len(records),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_fraction": score,
        "naive_logistic_elo": score_to_elo(score),
        "candidate_options": cand_opts,
        "opponent_options": opp_opts,
        "move_time_ms": args.move_time_ms,
        "errors": errs,
        "candidate_mean_ms": safe_mean([r["candidate_mean_ms"] for r in records if r["candidate_mean_ms"] is not None]),
        "opponent_mean_ms": safe_mean([r["opponent_mean_ms"] for r in records if r["opponent_mean_ms"] is not None]),
        "candidate_mean_depth": safe_mean([r["candidate_mean_depth"] for r in records if r["candidate_mean_depth"] is not None]),
        "opponent_mean_depth": safe_mean([r["opponent_mean_depth"] for r in records if r["opponent_mean_depth"] is not None]),
        "candidate_mean_nodes": safe_mean([r["candidate_mean_nodes"] for r in records if r["candidate_mean_nodes"] is not None]),
        "opponent_mean_nodes": safe_mean([r["opponent_mean_nodes"] for r in records if r["opponent_mean_nodes"] is not None]),
        "warning": "16 games is a screening sample, not a reliable Elo estimate.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if not errs else 2


if __name__ == "__main__":
    raise SystemExit(main())

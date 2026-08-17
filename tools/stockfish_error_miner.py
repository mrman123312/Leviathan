#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine

PIECE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def score_cp(info: dict, turn: chess.Color) -> int:
    score = info["score"].pov(turn).score(mate_score=100000)
    return 0 if score is None else int(score)


def material_cp(board: chess.Board) -> int:
    white = 0
    black = 0
    for piece in board.piece_map().values():
        value = PIECE_CP[piece.piece_type]
        if piece.color == chess.WHITE:
            white += value
        else:
            black += value
    return white - black


def board_features(board: chess.Board) -> dict:
    legal = list(board.legal_moves)
    captures = sum(board.is_capture(m) for m in legal)
    checks = sum(board.gives_check(m) for m in legal)
    promotions = sum(m.promotion is not None for m in legal)
    piece_count = len(board.piece_map())
    pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))
    nonpawn_count = piece_count - pawn_count - 2  # remove kings
    transition_moves = []
    if piece_count == 8:
        for move in legal:
            child = board.copy(stack=False)
            child.push(move)
            if len(child.piece_map()) <= 7:
                transition_moves.append(move.uci())

    mat_white = material_cp(board)
    mat_stm = mat_white if board.turn == chess.WHITE else -mat_white

    return {
        "piece_count": piece_count,
        "pawn_count": pawn_count,
        "nonpawn_nonking_count": nonpawn_count,
        "material_cp_white": mat_white,
        "material_cp_stm": mat_stm,
        "material_abs_cp": abs(mat_white),
        "legal_move_count": len(legal),
        "capture_count": captures,
        "checking_move_count": checks,
        "promotion_move_count": promotions,
        "rule50_halfmove_clock": board.halfmove_clock,
        "fullmove_number": board.fullmove_number,
        "in_check": board.is_check(),
        "eight_to_seven_transition_count": len(transition_moves),
        "eight_to_seven_transition_moves": transition_moves,
    }


def analyse_once(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int, game_token) -> dict:
    t0 = time.perf_counter()
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=game_token)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not info.get("pv"):
        raise RuntimeError("engine returned no PV")
    return {
        "move": info["pv"][0],
        "score_cp": score_cp(info, board.turn),
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "nodes": int(info.get("nodes", 0)),
        "nps": int(info.get("nps", 0)),
        "elapsed_ms": elapsed_ms,
        "pv": [m.uci() for m in info.get("pv", [])[:12]],
    }


def grade_move(engine: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move, nodes: int, game_token) -> dict:
    t0 = time.perf_counter()
    info = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        root_moves=[move],
        game=game_token,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "score_cp": score_cp(info, board.turn),
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "nodes": int(info.get("nodes", 0)),
        "elapsed_ms": elapsed_ms,
    }


def move_kind(board: chess.Board, move: chess.Move) -> str:
    if board.is_capture(move):
        return "capture"
    if move.promotion is not None:
        return "promotion"
    if board.gives_check(move):
        return "quiet_check"
    return "quiet"


def summarize(rows: list[dict], fast_nodes: int, deep_nodes: int, verify_nodes: int, threshold: int) -> dict:
    disagreements = [r for r in rows if r["fast_deep_disagree"]]
    candidates = [r for r in rows if r["candidate_error_preverify"]]
    verified = [r for r in rows if r["verified_error"]]
    by_horizon: dict[str, dict] = {}
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("horizon", "unknown"))].append(row)
    for horizon, group in groups.items():
        errs = [r for r in group if r["verified_error"]]
        by_horizon[horizon] = {
            "positions": len(group),
            "raw_disagreements": sum(r["fast_deep_disagree"] for r in group),
            "verified_errors": len(errs),
            "verified_error_rate": len(errs) / len(group) if group else 0.0,
            "mean_verified_regret_cp": statistics.mean(r["verified_regret_cp"] for r in errs) if errs else 0.0,
        }

    return {
        "schema": "LV_STOCKFISH_FINITE_COMPUTE_ERROR_MINER_V1",
        "positions": len(rows),
        "fast_nodes": fast_nodes,
        "deep_nodes": deep_nodes,
        "deep_multiplier": deep_nodes / fast_nodes,
        "verify_nodes": verify_nodes,
        "regret_threshold_cp": threshold,
        "raw_disagreements": len(disagreements),
        "raw_disagreement_rate": len(disagreements) / len(rows) if rows else 0.0,
        "candidate_errors_preverify": len(candidates),
        "verified_errors": len(verified),
        "verified_error_rate": len(verified) / len(rows) if rows else 0.0,
        "verified_deep_move_stable_count": sum(r.get("verified_deep_move_stable", False) for r in candidates),
        "mean_verified_regret_cp": statistics.mean(r["verified_regret_cp"] for r in verified) if verified else 0.0,
        "median_verified_regret_cp": statistics.median(r["verified_regret_cp"] for r in verified) if verified else 0.0,
        "max_verified_regret_cp": max((r["verified_regret_cp"] for r in verified), default=0),
        "eight_piece_positions": sum(r["features"]["piece_count"] == 8 for r in rows),
        "eight_to_seven_proximity_positions": sum(r["features"]["eight_to_seven_transition_count"] > 0 for r in rows),
        "fast_iteration_bestmove_flip_count": sum(r["quarter_fast_bestmove"] != r["fast_bestmove"] for r in rows),
        "by_horizon": by_horizon,
        "interpretation_rule": (
            "A verified finite-compute error requires fast/deep best-move disagreement, "
            "deep regret >= threshold, the deeper verification best move to remain stable, "
            "and verified regret to remain >= threshold."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--fast-nodes", type=int, default=50000)
    ap.add_argument("--deep-nodes", type=int, default=1600000)
    ap.add_argument("--verify-nodes", type=int, default=3200000)
    ap.add_argument("--regret-threshold-cp", type=int, default=25)
    args = ap.parse_args()

    if args.fast_nodes < 1000:
        raise SystemExit("fast-nodes too small")
    if args.deep_nodes <= args.fast_nodes:
        raise SystemExit("deep-nodes must exceed fast-nodes")
    if args.verify_nodes < args.deep_nodes:
        raise SystemExit("verify-nodes must be >= deep-nodes")

    source = json.loads(Path(args.input).read_text())
    if not isinstance(source, list) or not source:
        raise SystemExit("input must be a non-empty JSON list")

    rows: list[dict] = []
    engine = chess.engine.SimpleEngine.popen_uci(args.engine, timeout=30)
    config = {}
    if "Threads" in engine.options:
        config["Threads"] = 1
    if "Hash" in engine.options:
        config["Hash"] = 64
    if config:
        engine.configure(config)

    quarter_nodes = max(1000, args.fast_nodes // 4)

    try:
        for index, src in enumerate(source, 1):
            fen = src.get("fen") or src.get("final_fen")
            if not fen:
                raise RuntimeError(f"row {index} missing fen/final_fen")
            board = chess.Board(fen)
            features = board_features(board)

            q = analyse_once(engine, board, quarter_nodes, ("quarter", index, fen))
            fast = analyse_once(engine, board, args.fast_nodes, ("fast", index, fen))
            deep = analyse_once(engine, board, args.deep_nodes, ("deep", index, fen))

            fast_move = fast["move"]
            deep_move = deep["move"]
            disagree = fast_move != deep_move
            if disagree:
                graded_fast = grade_move(
                    engine,
                    board,
                    fast_move,
                    args.deep_nodes,
                    ("grade-fast", index, fen, fast_move.uci()),
                )
                regret = max(0, deep["score_cp"] - graded_fast["score_cp"])
            else:
                graded_fast = {
                    "score_cp": deep["score_cp"],
                    "depth": deep["depth"],
                    "seldepth": deep["seldepth"],
                    "nodes": 0,
                    "elapsed_ms": 0.0,
                }
                regret = 0

            candidate_error = disagree and regret >= args.regret_threshold_cp
            verified_regret = 0
            verified_deep_move = deep_move
            verified_deep_score = deep["score_cp"]
            verified_fast_score = graded_fast["score_cp"]
            verified_stable = False

            if candidate_error:
                verify_deep = analyse_once(
                    engine,
                    board,
                    args.verify_nodes,
                    ("verify-deep", index, fen),
                )
                verify_fast = grade_move(
                    engine,
                    board,
                    fast_move,
                    args.verify_nodes,
                    ("verify-fast", index, fen, fast_move.uci()),
                )
                verified_deep_move = verify_deep["move"]
                verified_deep_score = verify_deep["score_cp"]
                verified_fast_score = verify_fast["score_cp"]
                verified_regret = max(0, verified_deep_score - verified_fast_score)
                verified_stable = verified_deep_move == deep_move

            verified_error = bool(
                candidate_error
                and verified_stable
                and fast_move != verified_deep_move
                and verified_regret >= args.regret_threshold_cp
            )

            row = {
                "index": index,
                "source_name": src.get("name"),
                "horizon": src.get("horizon"),
                "fen": fen,
                "features": features,
                "quarter_fast_bestmove": q["move"].uci(),
                "quarter_fast_score_cp": q["score_cp"],
                "quarter_fast_depth": q["depth"],
                "fast_bestmove": fast_move.uci(),
                "fast_move_kind": move_kind(board, fast_move),
                "fast_score_cp": fast["score_cp"],
                "fast_depth": fast["depth"],
                "fast_seldepth": fast["seldepth"],
                "fast_nodes_actual": fast["nodes"],
                "fast_elapsed_ms": fast["elapsed_ms"],
                "deep_bestmove": deep_move.uci(),
                "deep_score_cp": deep["score_cp"],
                "deep_depth": deep["depth"],
                "deep_seldepth": deep["seldepth"],
                "deep_nodes_actual": deep["nodes"],
                "deep_elapsed_ms": deep["elapsed_ms"],
                "fast_deep_disagree": disagree,
                "deep_grade_fast_score_cp": graded_fast["score_cp"],
                "deep_regret_cp": regret,
                "candidate_error_preverify": candidate_error,
                "verified_deep_bestmove": verified_deep_move.uci(),
                "verified_deep_score_cp": verified_deep_score,
                "verified_fast_score_cp": verified_fast_score,
                "verified_regret_cp": verified_regret,
                "verified_deep_move_stable": verified_stable,
                "verified_error": verified_error,
                "score_shift_quarter_to_fast_cp": fast["score_cp"] - q["score_cp"],
                "bestmove_flip_quarter_to_fast": q["move"] != fast_move,
                "fast_pv": fast["pv"],
                "deep_pv": deep["pv"],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

            # Persist after every row so an interrupted local run can be salvaged.
            Path(args.output).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    finally:
        engine.quit()

    summary = summarize(
        rows,
        fast_nodes=args.fast_nodes,
        deep_nodes=args.deep_nodes,
        verify_nodes=args.verify_nodes,
        threshold=args.regret_threshold_cp,
    )
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

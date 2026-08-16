#!/usr/bin/env python3
"""Run one reproducible UCI game and emit PGN + machine-readable summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--white", required=True)
    ap.add_argument("--black", required=True)
    ap.add_argument("--white-name", default="Leviathan")
    ap.add_argument("--black-name", default="Stockfish")
    ap.add_argument("--movetime", type=int, default=150, help="milliseconds per move")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Leviathan single-game strength probe"
    game.headers["White"] = args.white_name
    game.headers["Black"] = args.black_name
    game.headers["TimeControl"] = f"{args.movetime}ms/move"
    node = game
    transcript = []

    white = chess.engine.SimpleEngine.popen_uci(args.white)
    black = chess.engine.SimpleEngine.popen_uci(args.black)
    try:
        for ply in range(args.max_plies):
            if board.is_game_over(claim_draw=True):
                break
            engine = white if board.turn == chess.WHITE else black
            info = engine.play(
                board,
                chess.engine.Limit(time=args.movetime / 1000.0),
                info=chess.engine.INFO_SCORE | chess.engine.INFO_DEPTH | chess.engine.INFO_NODES,
            )
            if info.move is None or info.move not in board.legal_moves:
                raise RuntimeError(f"illegal/null move from {'white' if board.turn else 'black'}: {info.move}")
            score = None
            if "score" in info.info:
                pov = info.info["score"].pov(board.turn)
                if pov.is_mate():
                    score = {"type": "mate", "value": pov.mate()}
                else:
                    score = {"type": "cp", "value": pov.score()}
            transcript.append({
                "ply": ply + 1,
                "side": "white" if board.turn == chess.WHITE else "black",
                "move": info.move.uci(),
                "score": score,
                "depth": info.info.get("depth"),
                "nodes": info.info.get("nodes"),
            })
            board.push(info.move)
            node = node.add_variation(info.move)

        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            termination = board.outcome(claim_draw=True).termination.name if board.outcome(claim_draw=True) else "unknown"
        else:
            result = "*"
            termination = "max_plies"
        game.headers["Result"] = result
        game.headers["Termination"] = termination

        pgn_path = Path(args.pgn)
        pgn_path.parent.mkdir(parents=True, exist_ok=True)
        pgn_path.write_text(str(game) + "\n", encoding="utf-8")
        summary = {
            "white": args.white_name,
            "black": args.black_name,
            "result": result,
            "termination": termination,
            "plies": len(transcript),
            "movetime_ms": args.movetime,
            "final_fen": board.fen(),
            "moves": transcript,
        }
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ("white", "black", "result", "termination", "plies", "movetime_ms")}))
    finally:
        white.quit()
        black.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

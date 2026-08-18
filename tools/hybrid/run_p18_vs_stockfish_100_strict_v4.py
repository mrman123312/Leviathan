#!/usr/bin/env python3
"""P18.6 strict match wrapper with exact failing-ply diagnostics.

Keeps the no-silent-retry policy from the strict harness and replaces play_game
with a diagnostic equivalent that reports the exact ply/FEN/side/engine if a UCI
play call fails. This makes any remaining protocol fault locally reproducible.
"""
from __future__ import annotations

try:
    import run_p18_vs_stockfish_100 as base
except ImportError:
    from . import run_p18_vs_stockfish_100 as base


def diagnostic_play_game(lev, sf, fen, leviathan_white, game_token, movetime_ms, max_plies):
    board = base.chess.Board(fen)
    limit = base.chess.engine.Limit(time=movetime_ms / 1000.0)
    moves = []

    for ply_index in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        leviathan_to_move = board.turn == (base.chess.WHITE if leviathan_white else base.chess.BLACK)
        eng = lev if leviathan_to_move else sf
        engine_name = "Leviathan" if leviathan_to_move else "Stockfish"

        try:
            result = eng.play(board, limit, game=game_token, ponder=True)
        except Exception as exc:
            print(base.json.dumps({
                "event": "PLAY_PROTOCOL_FAILURE_V4",
                "game_token": game_token,
                "ply": ply_index + 1,
                "engine": engine_name,
                "leviathan_white": leviathan_white,
                "fen": board.fen(),
                "moves_completed": len(moves),
                "last_moves": moves[-12:],
                "error": repr(exc),
            }, sort_keys=True), flush=True)
            raise

        move = result.move
        if move is None or move not in board.legal_moves:
            raise RuntimeError(
                f"illegal/no move from {engine_name} at ply {ply_index + 1} "
                f"fen={board.fen()}: {move}"
            )
        moves.append(move.uci())
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result_text = "1/2-1/2"
        termination = "max_plies"
    else:
        result_text = outcome.result()
        termination = outcome.termination.name

    return {
        "opening_fen": fen,
        "leviathan_white": leviathan_white,
        "result": result_text,
        "score_leviathan": base.score_from_result(result_text, leviathan_white),
        "termination": termination,
        "plies": len(moves),
        "moves": moves,
    }


base.play_game = diagnostic_play_game


def strict_gpu_game(args, threads, session_log, game_no, fen, leviathan_white):
    lev = sf = None
    try:
        lev, sf, cmd = base.open_hybrid_pair(args, threads, session_log, True)
        print(base.json.dumps({"event": "gpu_hybrid_command", "game": game_no, "command": cmd}), flush=True)
        row = diagnostic_play_game(
            lev, sf, fen, leviathan_white, f"p18-gpu-{game_no}",
            args.movetime_ms, args.max_plies,
        )
        row["game"] = game_no
        row["protocol_attempts"] = 1
        return row
    except Exception as exc:
        print(base.json.dumps({
            "event": "STRICT_PROTOCOL_FAILURE",
            "game": game_no,
            "side": "gpu",
            "error": repr(exc),
        }), flush=True)
        raise
    finally:
        base.safe_quit(lev)
        base.safe_quit(sf)


def strict_no_gpu(args, threads, gpu_row, no_gpu_log):
    game_no = int(gpu_row["game"])
    lev = sf = None
    try:
        lev, sf, cmd = base.open_hybrid_pair(args, threads, no_gpu_log, False)
        print(base.json.dumps({
            "event": "no_gpu_hybrid_command",
            "original_game": game_no,
            "command": cmd,
        }), flush=True)
        replay = diagnostic_play_game(
            lev, sf, str(gpu_row["opening_fen"]), bool(gpu_row["leviathan_white"]),
            f"p18-no-gpu-{game_no}", args.movetime_ms, args.max_plies,
        )
        gs = float(gpu_row["score_leviathan"])
        ns = float(replay["score_leviathan"])
        div = base.first_divergence(list(gpu_row.get("moves") or []), list(replay.get("moves") or []))
        return {
            "original_game": game_no,
            "opening_fen": gpu_row["opening_fen"],
            "leviathan_white": gpu_row["leviathan_white"],
            "gpu_result": gpu_row["result"],
            "gpu_score": gs,
            "no_gpu_result": replay["result"],
            "no_gpu_score": ns,
            "no_gpu_termination": replay["termination"],
            "no_gpu_plies": replay["plies"],
            "no_gpu_moves": replay["moves"],
            "no_gpu_protocol_attempts": 1,
            "outcome_comparison": base.outcome_label(gs, ns),
            **div,
        }
    except Exception as exc:
        print(base.json.dumps({
            "event": "STRICT_PROTOCOL_FAILURE",
            "original_game": game_no,
            "side": "no_gpu",
            "error": repr(exc),
        }), flush=True)
        raise
    finally:
        base.safe_quit(lev)
        base.safe_quit(sf)


base.run_gpu_game = strict_gpu_game
base.run_no_gpu_ablation = strict_no_gpu

if __name__ == "__main__":
    raise SystemExit(base.main())

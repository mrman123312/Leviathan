#!/usr/bin/env python3
"""Strict wrapper for the P18 100-game harness.

Unlike the exploratory harness, an engine timeout/crash is never silently retried
inside the same invocation. The run aborts at that game; already completed clean
games remain resumable on the next invocation. This makes protocol instability
visible instead of substituting a second nondeterministic attempt into statistics.
"""
from __future__ import annotations

try:
    import run_p18_vs_stockfish_100 as base
except ImportError:
    from . import run_p18_vs_stockfish_100 as base


def strict_gpu_game(args, threads, session_log, game_no, fen, leviathan_white):
    lev = sf = None
    try:
        lev, sf, cmd = base.open_hybrid_pair(args, threads, session_log, True)
        print(base.json.dumps({"event": "gpu_hybrid_command", "game": game_no, "command": cmd}), flush=True)
        row = base.play_game(lev, sf, fen, leviathan_white, f"p18-gpu-{game_no}", args.movetime_ms, args.max_plies)
        row["game"] = game_no
        row["protocol_attempts"] = 1
        return row
    except Exception as exc:
        print(base.json.dumps({"event": "STRICT_PROTOCOL_FAILURE", "game": game_no, "side": "gpu", "error": repr(exc)}), flush=True)
        raise
    finally:
        base.safe_quit(lev)
        base.safe_quit(sf)


def strict_no_gpu(args, threads, gpu_row, no_gpu_log):
    game_no = int(gpu_row["game"])
    lev = sf = None
    try:
        lev, sf, cmd = base.open_hybrid_pair(args, threads, no_gpu_log, False)
        print(base.json.dumps({"event": "no_gpu_hybrid_command", "original_game": game_no, "command": cmd}), flush=True)
        replay = base.play_game(
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
        print(base.json.dumps({"event": "STRICT_PROTOCOL_FAILURE", "original_game": game_no, "side": "no_gpu", "error": repr(exc)}), flush=True)
        raise
    finally:
        base.safe_quit(lev)
        base.safe_quit(sf)


base.run_gpu_game = strict_gpu_game
base.run_no_gpu_ablation = strict_no_gpu

if __name__ == "__main__":
    raise SystemExit(base.main())

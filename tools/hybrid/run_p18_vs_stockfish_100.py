#!/usr/bin/env python3
"""Resumable 100-game P18 hybrid vs Stockfish match with decisive no-GPU ablations.

Main match:
- P18 CPU+GPU hybrid vs frozen Stockfish
- paired openings with reversed colors
- equal CPU thread/hash budgets and equal own-move time
- UCI pondering enabled for both sides

Counterfactual after every hybrid WIN or LOSS:
- exact same opening and Leviathan color
- exact same P18 multi-reply/ponder architecture
- exact same Stockfish, CPU threads, hash, own-move time, and pondering
- learned GPU advisor disabled via --gpu-device off and no checkpoint
- built-in heuristic scorer remains so only the learned GPU/model feature is removed
- draws do not trigger a replay

A single replay is outcome-level evidence, not proof: threaded/pondering chess searches may be
nondeterministic. The harness records the first move divergence and whether the GPU-enabled
outcome was better, worse, or unchanged versus the otherwise-identical no-GPU hybrid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import chess
import chess.engine


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_quit(engine: chess.engine.SimpleEngine | None) -> None:
    if engine is None:
        return
    try:
        engine.quit()
    except Exception:
        try:
            engine.close()
        except Exception:
            pass


def configure_engine(engine: chess.engine.SimpleEngine, threads: int, hash_mb: int) -> None:
    # Ponder is managed by python-chess. engine.play(..., ponder=True) activates it.
    opts: dict[str, Any] = {}
    if "Threads" in engine.options:
        opts["Threads"] = threads
    if "Hash" in engine.options:
        opts["Hash"] = hash_mb
    if opts:
        engine.configure(opts)


def weighted_choice(infos: list[dict[str, Any]], board: chess.Board, rng: random.Random, temp_cp: float) -> chess.Move:
    candidates: list[tuple[chess.Move, float]] = []
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        score_obj = info.get("score")
        score = 0.0
        if score_obj is not None:
            val = score_obj.pov(board.turn).score(mate_score=100000)
            score = float(val if val is not None else 0)
        candidates.append((pv[0], score))
    if not candidates:
        return next(iter(board.legal_moves))
    best = max(s for _, s in candidates)
    weights = [math.exp(max(-20.0, min(0.0, (s - best) / max(1.0, temp_cp)))) for _, s in candidates]
    return rng.choices([m for m, _ in candidates], weights=weights, k=1)[0]


def generate_openings(opponent: str, out: Path, count: int, plies: int, seed: int, nodes: int) -> list[str]:
    if out.exists():
        fens = [x.strip() for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(fens) >= count:
            return fens[:count]
    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(opponent, timeout=30.0)
    fens: list[str] = []
    try:
        configure_engine(engine, 1, 32)
        for i in range(count):
            board = chess.Board()
            for _ in range(plies):
                if board.is_game_over(claim_draw=True):
                    break
                mpv = max(1, min(4, board.legal_moves.count()))
                info = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=mpv, game=f"opening-{i}")
                infos = info if isinstance(info, list) else [info]
                move = weighted_choice(infos, board, rng, 55.0)
                if move not in board.legal_moves:
                    break
                board.push(move)
            if board.is_game_over(claim_draw=True):
                board = chess.Board()
            fens.append(board.fen())
            print(json.dumps({"event": "opening", "index": i + 1, "fen": board.fen()}), flush=True)
    finally:
        safe_quit(engine)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(fens) + "\n", encoding="utf-8")
    return fens


def hybrid_command(args: argparse.Namespace, threads: int, log_path: Path, use_gpu_model: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(args.hybrid_script).resolve()),
        "--engine", str(Path(args.engine).resolve()),
        "--opponent-engine", str(Path(args.opponent_engine).resolve()),
        "--gpu-device", "auto" if use_gpu_model else "off",
        "--threads", str(threads),
        "--hash", str(args.hash),
        "--max-scouts", str(args.max_scouts),
        "--reply-nodes", str(args.reply_nodes),
        "--anneal-seconds", str(args.anneal_seconds),
        "--min-final-scouts", str(args.min_final_scouts),
        "--log", str(log_path.resolve()),
    ]
    if use_gpu_model:
        cmd.extend(["--model", str(Path(args.model).resolve())])
    return cmd


def open_hybrid_pair(args: argparse.Namespace, threads: int, log_path: Path, use_gpu_model: bool):
    cmd = hybrid_command(args, threads, log_path, use_gpu_model)
    lev = chess.engine.SimpleEngine.popen_uci(cmd, timeout=45.0)
    sf = chess.engine.SimpleEngine.popen_uci(str(Path(args.opponent_engine).resolve()), timeout=30.0)
    configure_engine(lev, threads, args.hash)
    configure_engine(sf, threads, args.hash)
    return lev, sf, cmd


def score_from_result(result: str, leviathan_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    if (result == "1-0" and leviathan_white) or (result == "0-1" and not leviathan_white):
        return 1.0
    return 0.0


def play_game(
    lev: chess.engine.SimpleEngine,
    sf: chess.engine.SimpleEngine,
    fen: str,
    leviathan_white: bool,
    game_token: str,
    movetime_ms: int,
    max_plies: int,
) -> dict[str, Any]:
    board = chess.Board(fen)
    limit = chess.engine.Limit(time=movetime_ms / 1000.0)
    moves: list[str] = []
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        leviathan_to_move = board.turn == (chess.WHITE if leviathan_white else chess.BLACK)
        eng = lev if leviathan_to_move else sf
        result = eng.play(board, limit, game=game_token, ponder=True)
        move = result.move
        if move is None or move not in board.legal_moves:
            raise RuntimeError(f"illegal/no move from {'Leviathan' if leviathan_to_move else 'Stockfish'} at {board.fen()}: {move}")
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
        "score_leviathan": score_from_result(result_text, leviathan_white),
        "termination": termination,
        "plies": len(moves),
        "moves": moves,
    }


def first_divergence(a: list[str], b: list[str]) -> dict[str, Any]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return {"first_divergence_ply": i + 1, "gpu_move": a[i], "no_gpu_move": b[i]}
    if len(a) != len(b):
        return {
            "first_divergence_ply": n + 1,
            "gpu_move": a[n] if len(a) > n else None,
            "no_gpu_move": b[n] if len(b) > n else None,
        }
    return {"first_divergence_ply": None, "gpu_move": None, "no_gpu_move": None}


def load_rows(path: Path, key_name: str = "game") -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            g = int(r.get(key_name, 0))
            if g > 0:
                rows[g] = r
        except Exception:
            continue
    return rows


def main_summary(rows: list[dict[str, Any]], args: argparse.Namespace, threads: int) -> dict[str, Any]:
    w = sum(float(r["score_leviathan"]) == 1.0 for r in rows)
    d = sum(float(r["score_leviathan"]) == 0.5 for r in rows)
    l = sum(float(r["score_leviathan"]) == 0.0 for r in rows)
    score = sum(float(r["score_leviathan"]) for r in rows) / len(rows) if rows else 0.0
    elo = None
    if 0.0 < score < 1.0:
        elo = -400.0 * math.log10(1.0 / score - 1.0)
    return {
        "games": len(rows),
        "wins_leviathan": w,
        "draws": d,
        "losses_leviathan": l,
        "score_leviathan": score,
        "naive_elo_from_score": elo,
        "movetime_ms_each": args.movetime_ms,
        "ponder": True,
        "threads_each": threads,
        "hash_mb_each": args.hash,
        "max_scouts": args.max_scouts,
        "anneal_seconds": args.anneal_seconds,
        "min_final_scouts": args.min_final_scouts,
    }


def outcome_label(gpu_score: float, no_gpu_score: float) -> str:
    if gpu_score > no_gpu_score:
        return "gpu_better_outcome"
    if gpu_score < no_gpu_score:
        return "gpu_worse_outcome"
    return "same_outcome"


def ablation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(r.get("outcome_comparison")) for r in rows]
    win_rows = [r for r in rows if float(r.get("gpu_score", -1)) == 1.0]
    loss_rows = [r for r in rows if float(r.get("gpu_score", -1)) == 0.0]

    def counts(rs: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "no_gpu_win": sum(float(r.get("no_gpu_score", -1)) == 1.0 for r in rs),
            "no_gpu_draw": sum(float(r.get("no_gpu_score", -1)) == 0.5 for r in rs),
            "no_gpu_loss": sum(float(r.get("no_gpu_score", -1)) == 0.0 for r in rs),
        }

    n = len(rows)
    return {
        "decisive_replays": n,
        "gpu_better_outcome": labels.count("gpu_better_outcome"),
        "gpu_worse_outcome": labels.count("gpu_worse_outcome"),
        "same_outcome": labels.count("same_outcome"),
        "mean_gpu_minus_no_gpu_score": (
            sum(float(r["gpu_score"]) - float(r["no_gpu_score"]) for r in rows) / n if n else 0.0
        ),
        "for_gpu_enabled_wins": {"replays": len(win_rows), **counts(win_rows)},
        "for_gpu_enabled_losses": {"replays": len(loss_rows), **counts(loss_rows)},
    }


def run_gpu_game(args: argparse.Namespace, threads: int, session_log: Path, game_no: int, fen: str, leviathan_white: bool):
    last_error = None
    for attempt in range(2):
        lev = sf = None
        try:
            lev, sf, cmd = open_hybrid_pair(args, threads, session_log, True)
            if attempt == 0:
                print(json.dumps({"event": "gpu_hybrid_command", "game": game_no, "command": cmd}), flush=True)
            row = play_game(lev, sf, fen, leviathan_white, f"p18-gpu-{game_no}", args.movetime_ms, args.max_plies)
            row["game"] = game_no
            return row
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                print(json.dumps({"event": "gpu_game_restart", "game": game_no, "error": repr(exc)}), flush=True)
        finally:
            safe_quit(lev)
            safe_quit(sf)
    raise RuntimeError(f"GPU-enabled game {game_no} failed twice: {last_error}")


def run_no_gpu_ablation(args: argparse.Namespace, threads: int, gpu_row: dict[str, Any], no_gpu_log: Path) -> dict[str, Any]:
    game_no = int(gpu_row["game"])
    last_error = None
    for attempt in range(2):
        lev = sf = None
        try:
            lev, sf, cmd = open_hybrid_pair(args, threads, no_gpu_log, False)
            if attempt == 0:
                print(json.dumps({"event": "no_gpu_hybrid_command", "original_game": game_no, "command": cmd}), flush=True)
            replay = play_game(
                lev,
                sf,
                str(gpu_row["opening_fen"]),
                bool(gpu_row["leviathan_white"]),
                f"p18-no-gpu-{game_no}",
                args.movetime_ms,
                args.max_plies,
            )
            gs = float(gpu_row["score_leviathan"])
            ns = float(replay["score_leviathan"])
            div = first_divergence(list(gpu_row.get("moves") or []), list(replay.get("moves") or []))
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
                "outcome_comparison": outcome_label(gs, ns),
                **div,
            }
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                print(json.dumps({"event": "no_gpu_replay_restart", "original_game": game_no, "error": repr(exc)}), flush=True)
        finally:
            safe_quit(lev)
            safe_quit(sf)
    raise RuntimeError(f"no-GPU replay for game {game_no} failed twice: {last_error}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="P09 CPU engine under both hybrid variants")
    ap.add_argument("--opponent-engine", required=True, help="frozen Stockfish baseline")
    ap.add_argument("--model", required=True, help="promoted P18.4 checkpoint used only by GPU-enabled variant")
    ap.add_argument("--hybrid-script", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--movetime-ms", type=int, default=500)
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--threads", type=int, default=0, help="0 = auto half logical CPUs, capped at 8")
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--max-scouts", type=int, default=4)
    ap.add_argument("--reply-nodes", type=int, default=12000)
    ap.add_argument("--anneal-seconds", type=float, default=0.15)
    ap.add_argument("--min-final-scouts", type=int, default=2)
    ap.add_argument("--opening-plies", type=int, default=10)
    ap.add_argument("--opening-nodes", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    if args.games <= 0 or args.games % 2:
        raise SystemExit("--games must be a positive even number so colors can be paired")
    for p in (args.engine, args.opponent_engine, args.model, args.hybrid_script):
        if not Path(p).exists():
            raise SystemExit(f"missing required file: {p}")

    logical = os.cpu_count() or 8
    threads = args.threads if args.threads > 0 else max(1, min(8, logical // 2))
    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    identity = {
        "engine_sha256": sha256_file(Path(args.engine)),
        "stockfish_sha256": sha256_file(Path(args.opponent_engine)),
        "model_sha256": sha256_file(Path(args.model)),
        "games": args.games,
        "movetime_ms": args.movetime_ms,
        "threads_each": threads,
        "hash": args.hash,
        "max_scouts": args.max_scouts,
        "reply_nodes": args.reply_nodes,
        "anneal_seconds": args.anneal_seconds,
        "min_final_scouts": args.min_final_scouts,
        "opening_plies": args.opening_plies,
        "opening_nodes": args.opening_nodes,
        "seed": args.seed,
        "ponder_both_sides": True,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    out = base_out / run_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != identity:
            raise SystemExit("existing run manifest differs; refusing to mix match configurations")
    else:
        manifest.write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")

    protocol = {
        "ablation_protocol": "decisive-only-no-gpu-v2",
        "trigger": "GPU-enabled hybrid win or loss; no replay for draw",
        "same": ["opening_fen", "leviathan_color", "P18 multi-reply architecture", "Stockfish binary", "threads", "hash", "movetime_ms", "pondering", "reply_nodes", "scout_count"],
        "removed": ["learned P18.4 checkpoint", "GPU/accelerated model inference"],
        "replacement": "same HybridProxyV2 using --gpu-device off and built-in heuristic scorer",
    }
    (out / "gpu-ablation-protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"event": "match_config", "run_id": run_id, "logical_cpus": logical, **identity, **protocol}, indent=2), flush=True)
    openings_path = out / "openings.fen"
    openings = generate_openings(args.opponent_engine, openings_path, args.games // 2, args.opening_plies, args.seed, args.opening_nodes)
    rows_path = out / "games.jsonl"
    ablation_path = out / "decisive-no-gpu-replays.jsonl"
    completed = load_rows(rows_path, "game")
    ablation_completed = load_rows(ablation_path, "original_game")

    if completed:
        print(json.dumps({"event": "resume", "completed_games": len(completed), "remaining": args.games - len(completed)}), flush=True)

    gpu_log = out / "p18-gpu-session.jsonl"
    no_gpu_log = out / "p18-no-gpu-session.jsonl"

    # Backfill decisive GPU games from an already-started run.
    for game_no in sorted(completed):
        row = completed[game_no]
        if float(row.get("score_leviathan", 0.5)) == 0.5 or game_no in ablation_completed:
            continue
        print(json.dumps({"event": "no_gpu_replay_start", "original_game": game_no, "reason": "resume_backfill"}), flush=True)
        ab = run_no_gpu_ablation(args, threads, row, no_gpu_log)
        with ablation_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ab, sort_keys=True) + "\n")
            f.flush()
        ablation_completed[game_no] = ab
        print(json.dumps({"event": "no_gpu_replay_complete", **ab, "ablation_cumulative": ablation_summary(list(ablation_completed.values()))}, sort_keys=True), flush=True)

    for game_no in range(1, args.games + 1):
        if game_no in completed:
            continue
        fen = openings[(game_no - 1) // 2]
        leviathan_white = game_no % 2 == 1
        row = run_gpu_game(args, threads, gpu_log, game_no, fen, leviathan_white)
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        completed[game_no] = row
        current = [completed[g] for g in sorted(completed)]
        print(json.dumps({"event": "game_complete", **row, "cumulative": main_summary(current, args, threads)}, sort_keys=True), flush=True)

        if float(row["score_leviathan"]) != 0.5:
            print(json.dumps({"event": "no_gpu_replay_start", "original_game": game_no, "gpu_result": row["result"]}), flush=True)
            ab = run_no_gpu_ablation(args, threads, row, no_gpu_log)
            with ablation_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ab, sort_keys=True) + "\n")
                f.flush()
            ablation_completed[game_no] = ab
            print(json.dumps({"event": "no_gpu_replay_complete", **ab, "ablation_cumulative": ablation_summary(list(ablation_completed.values()))}, sort_keys=True), flush=True)

    ordered = [completed[g] for g in range(1, args.games + 1) if g in completed]
    ab_ordered = [ablation_completed[g] for g in sorted(ablation_completed)]
    final = main_summary(ordered, args, threads)
    ab_final = ablation_summary(ab_ordered)
    expected_replays = sum(float(r["score_leviathan"]) != 0.5 for r in ordered)
    final_payload = {
        "summary": final,
        "decisive_gpu_ablation": ab_final,
        "expected_decisive_replays": expected_replays,
        "completed_decisive_replays": len(ab_ordered),
        "gpu_ablation_complete": len(ab_ordered) == expected_replays,
        "run_id": run_id,
        "manifest": identity,
        "games_file": str(rows_path),
        "gpu_ablation_file": str(ablation_path),
        "gpu_session_log": str(gpu_log),
        "no_gpu_session_log": str(no_gpu_log),
    }
    (out / "summary.json").write_text(json.dumps(final_payload, indent=2, sort_keys=True), encoding="utf-8")
    print("=== P18.4 CPU+GPU VS STOCKFISH: 100 GAMES + DECISIVE NO-GPU ABLATIONS COMPLETE ===", flush=True)
    print(json.dumps(final_payload, indent=2, sort_keys=True), flush=True)
    return 0 if len(ordered) == args.games and len(ab_ordered) == expected_replays else 6


if __name__ == "__main__":
    raise SystemExit(main())

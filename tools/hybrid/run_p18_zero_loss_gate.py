#!/usr/bin/env python3
"""P18.8 zero-loss promotion gate.

A loss is a counterexample, not an acceptable sample. Before the fresh paired
match, replay every known learned-advisor regression several times. During the
fresh match, the first Leviathan loss is saved, replayed with the learned model
removed, and the candidate is rejected immediately.

This harness never rewrites a loss into a draw and never hides a failed game.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

try:
    import run_p18_vs_stockfish_100_strict_v5 as strict
except ImportError:
    from . import run_p18_vs_stockfish_100_strict_v5 as strict

base = strict.base

# Exact counterexamples observed in the stable P18.7 run. In both cases the
# learned-advisor game lost and the otherwise-identical no-model replay drew.
LOSS_SENTINELS = [
    {
        "name": "p18.7-game24",
        "fen": "rnbqk2r/pp3ppp/2pbpn2/3p4/2PP4/4PN1P/PP3PP1/RNBQKB1R w KQkq - 1 6",
        "leviathan_white": False,
    },
    {
        "name": "p18.7-game27",
        "fen": "rnbqkb1r/pppnp2p/4p1p1/3p4/8/P1N5/1PPP1PPP/R1BQKBNR w KQkq - 0 6",
        "leviathan_white": True,
    },
]

_ORIGINAL_HYBRID_COMMAND = base.hybrid_command


def firewall_hybrid_command(args, threads, log_path, use_gpu_model):
    cmd = _ORIGINAL_HYBRID_COMMAND(args, threads, log_path, use_gpu_model)
    if use_gpu_model:
        # P18.8 default: learned model is shadow-only. It may produce telemetry,
        # but the heuristic counterfactual scheduler owns branch choice,
        # allocation and promotion.
        cmd.extend(["--advisor-authority", "shadow"])
    return cmd


base.hybrid_command = firewall_hybrid_command


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="P18.8 permanent loss-sentinel + zero-loss gate")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--opponent-engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hybrid-script", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--movetime-ms", type=int, default=500)
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--max-scouts", type=int, default=4)
    ap.add_argument("--reply-nodes", type=int, default=12000)
    ap.add_argument("--anneal-seconds", type=float, default=0.0)
    ap.add_argument("--min-final-scouts", type=int, default=2)
    ap.add_argument("--opening-plies", type=int, default=10)
    ap.add_argument("--opening-nodes", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--sentinel-repeats", type=int, default=3)
    return ap


def replay_counterfactual(args, threads, row, no_model_log, out_path):
    print(json.dumps({
        "event": "loss_counterfactual_start",
        "game": row.get("game"),
        "result": row.get("result"),
        "opening_fen": row.get("opening_fen"),
    }), flush=True)
    ab = strict.strict_no_gpu(args, threads, row, no_model_log)
    append_jsonl(out_path, ab)
    print(json.dumps({"event": "loss_counterfactual_complete", **ab}, sort_keys=True), flush=True)
    return ab


def main() -> int:
    args = parser().parse_args()
    if args.games <= 0 or args.games % 2:
        raise SystemExit("--games must be a positive even number")
    if args.sentinel_repeats < 1:
        raise SystemExit("--sentinel-repeats must be >= 1")
    for p in (args.engine, args.opponent_engine, args.model, args.hybrid_script):
        if not Path(p).exists():
            raise SystemExit(f"missing required file: {p}")

    logical = os.cpu_count() or 8
    threads = args.threads if args.threads > 0 else max(1, min(8, logical // 2))
    identity = {
        "gate": "p18.8-zero-loss-v1",
        "engine_sha256": base.sha256_file(Path(args.engine)),
        "stockfish_sha256": base.sha256_file(Path(args.opponent_engine)),
        "model_sha256": base.sha256_file(Path(args.model)),
        "hybrid_sha256": base.sha256_file(Path(args.hybrid_script)),
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
        "sentinel_repeats": args.sentinel_repeats,
        "advisor_authority": "shadow",
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "ZERO_LOSS_GATE_CONFIG", "run_id": run_id, **identity}, indent=2), flush=True)

    sentinel_rows = out / "loss-sentinel-games.jsonl"
    sentinel_ab = out / "loss-sentinel-counterfactuals.jsonl"
    sentinel_gpu_log = out / "loss-sentinel-shadow-session.jsonl"
    sentinel_no_model_log = out / "loss-sentinel-no-model-session.jsonl"

    print("=== PERMANENT LOSS SENTINEL GATE ===", flush=True)
    sentinel_index = 0
    for case in LOSS_SENTINELS:
        for repeat in range(1, args.sentinel_repeats + 1):
            sentinel_index += 1
            row = strict.strict_gpu_game(
                args,
                threads,
                sentinel_gpu_log,
                100000 + sentinel_index,
                case["fen"],
                bool(case["leviathan_white"]),
            )
            row["sentinel"] = case["name"]
            row["repeat"] = repeat
            append_jsonl(sentinel_rows, row)
            print(json.dumps({
                "event": "loss_sentinel_complete",
                "sentinel": case["name"],
                "repeat": repeat,
                "result": row["result"],
                "score_leviathan": row["score_leviathan"],
                "plies": row["plies"],
            }, sort_keys=True), flush=True)

            if float(row["score_leviathan"]) == 0.0:
                ab = replay_counterfactual(args, threads, row, sentinel_no_model_log, sentinel_ab)
                print(json.dumps({
                    "event": "LOSS_SENTINEL_GATE_FAILED",
                    "sentinel": case["name"],
                    "repeat": repeat,
                    "shadow_result": row["result"],
                    "no_model_result": ab["no_gpu_result"],
                    "opening_fen": row["opening_fen"],
                }, sort_keys=True), flush=True)
                return 21

    print(json.dumps({
        "event": "LOSS_SENTINEL_GATE_PASSED",
        "cases": len(LOSS_SENTINELS),
        "repeats_each": args.sentinel_repeats,
        "games": sentinel_index,
        "losses": 0,
    }, sort_keys=True), flush=True)

    print("=== FRESH 100-GAME ZERO-LOSS GATE ===", flush=True)
    openings_path = out / "openings.fen"
    openings = base.generate_openings(
        args.opponent_engine,
        openings_path,
        args.games // 2,
        args.opening_plies,
        args.seed,
        args.opening_nodes,
    )
    games_path = out / "games.jsonl"
    counter_path = out / "decisive-counterfactuals.jsonl"
    shadow_log = out / "shadow-session.jsonl"
    no_model_log = out / "no-model-session.jsonl"
    rows = []
    counters = []

    for game_no in range(1, args.games + 1):
        fen = openings[(game_no - 1) // 2]
        leviathan_white = game_no % 2 == 1
        row = strict.strict_gpu_game(args, threads, shadow_log, game_no, fen, leviathan_white)
        append_jsonl(games_path, row)
        rows.append(row)
        summary = base.main_summary(rows, args, threads)
        print(json.dumps({"event": "game_complete", **row, "cumulative": summary}, sort_keys=True), flush=True)

        score = float(row["score_leviathan"])
        if score != 0.5:
            ab = strict.strict_no_gpu(args, threads, row, no_model_log)
            append_jsonl(counter_path, ab)
            counters.append(ab)
            print(json.dumps({
                "event": "decisive_counterfactual_complete",
                **ab,
                "cumulative": base.ablation_summary(counters),
            }, sort_keys=True), flush=True)

        if score == 0.0:
            payload = {
                "event": "ZERO_LOSS_GATE_FAILED",
                "game": game_no,
                "opening_fen": row["opening_fen"],
                "leviathan_white": row["leviathan_white"],
                "result": row["result"],
                "termination": row["termination"],
                "summary": summary,
                "counterfactual": counters[-1] if counters else None,
            }
            (out / "failure.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
            return 31

    final = {
        "event": "ZERO_LOSS_GATE_PASSED",
        "run_id": run_id,
        "summary": base.main_summary(rows, args, threads),
        "decisive_counterfactual": base.ablation_summary(counters),
        "known_loss_sentinels_passed": True,
        "advisor_authority": "shadow",
    }
    (out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

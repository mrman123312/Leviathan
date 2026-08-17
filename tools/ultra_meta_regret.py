#!/usr/bin/env python3
"""Equal-budget root-allocation experiments for Fundamentals Ultra.

This is intentionally an experiment harness, not a production UCI wrapper. It asks whether
one monolithic search is the best use of a fixed node budget, and grades the selected move
against a much deeper pinned Stockfish oracle.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import chess
import chess.engine

MATE = 100000
FUNDAMENTALS = {
    "Threads": 1,
    "Hash": 64,
    "Leviathan Fundamentals": True,
    "Leviathan Fundamentals Authority": 1,
    "Leviathan Quiet Overdrive": 0,
}
PLAIN = {"Threads": 1, "Hash": 64}


def cp(info: dict, color: chess.Color) -> int:
    return int(info["score"].pov(color).score(mate_score=MATE) or 0)


def best(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int,
         multipv: int = 1, root_moves=None):
    result = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=multipv,
        root_moves=root_moves,
    )
    infos = result if isinstance(result, list) else [result]
    rows = []
    for info in infos:
        pv = info.get("pv", [])
        if pv:
            rows.append((pv[0], cp(info, board.turn), int(info.get("nodes", nodes))))
    if not rows:
        raise RuntimeError("engine returned no PV")
    return rows


def choose_monolith(kernel, base, sf, board, total):
    row = best(kernel, board, total)[0]
    return row[0], {"requested_nodes": total, "views": ["kernel"]}


def choose_portfolio(kernel, base, sf, board, total, width):
    first = 44000 if width == 2 else 35000
    remain = total - first
    initial = best(kernel, board, first, multipv=width)
    candidates = []
    seen = set()
    for mv, score, _ in initial:
        if mv not in seen:
            seen.add(mv); candidates.append(mv)
    each = max(1000, remain // max(1, len(candidates)))
    verified = []
    for mv in candidates:
        row = best(kernel, board, each, root_moves=[mv])[0]
        verified.append((mv, row[1]))
    verified.sort(key=lambda x: x[1], reverse=True)
    return verified[0][0], {
        "requested_nodes": first + each * len(candidates),
        "views": [f"kernel-multipv{width}", "kernel-restricted-verify"],
        "candidates": [m.uci() for m in candidates],
        "verified": [(m.uci(), s) for m, s in verified],
    }


def choose_duel(kernel, other, board, total, other_name):
    first = 30000
    a = best(kernel, board, first)[0][0]
    b = best(other, board, first)[0][0]
    if a == b:
        return a, {"requested_nodes": first * 2, "views": ["kernel", other_name], "agreed": True}
    each = (total - 2 * first) // 2
    candidates = [a, b]
    verified = []
    for mv in candidates:
        row = best(kernel, board, each, root_moves=[mv])[0]
        verified.append((mv, row[1]))
    verified.sort(key=lambda x: x[1], reverse=True)
    return verified[0][0], {
        "requested_nodes": first * 2 + each * 2,
        "views": ["kernel", other_name, "kernel-restricted-verify"],
        "agreed": False,
        "candidates": [m.uci() for m in candidates],
        "verified": [(m.uci(), s) for m, s in verified],
    }


def choose_committee3(kernel, base, sf, board, total):
    per = 20000
    proposals = [
        ("kernel", best(kernel, board, per)[0][0]),
        ("base", best(base, board, per)[0][0]),
        ("stockfish", best(sf, board, per)[0][0]),
    ]
    candidates = []
    for _, mv in proposals:
        if mv not in candidates:
            candidates.append(mv)
    remain = total - 3 * per
    each = max(1000, remain // len(candidates))
    verified = []
    for mv in candidates:
        row = best(kernel, board, each, root_moves=[mv])[0]
        verified.append((mv, row[1]))
    verified.sort(key=lambda x: x[1], reverse=True)
    return verified[0][0], {
        "requested_nodes": 3 * per + each * len(candidates),
        "views": ["kernel", "base", "stockfish", "kernel-restricted-verify"],
        "proposals": [(n, m.uci()) for n, m in proposals],
        "candidates": [m.uci() for m in candidates],
        "verified": [(m.uci(), s) for m, s in verified],
    }


def choose(strategy, kernel, base, sf, board, total):
    if strategy == "monolith":
        return choose_monolith(kernel, base, sf, board, total)
    if strategy == "portfolio2":
        return choose_portfolio(kernel, base, sf, board, total, 2)
    if strategy == "portfolio3":
        return choose_portfolio(kernel, base, sf, board, total, 3)
    if strategy == "duel-base":
        return choose_duel(kernel, base, board, total, "base")
    if strategy == "duel-stockfish":
        return choose_duel(kernel, sf, board, total, "stockfish")
    if strategy == "committee3":
        return choose_committee3(kernel, base, sf, board, total)
    raise ValueError(strategy)


def generate_positions(sf, count, seed):
    rng = random.Random(seed)
    out, seen = [], set()
    attempts = 0
    while len(out) < count and attempts < count * 100:
        attempts += 1
        board = chess.Board()
        plies = rng.randrange(8, 19, 2)
        for _ in range(plies):
            if board.is_game_over():
                break
            infos = best(sf, board, 1800, multipv=min(6, board.legal_moves.count()))
            best_cp = infos[0][1]
            viable = [(m, s) for m, s, _ in infos if best_cp - s <= 110]
            weights = list(range(len(viable), 0, -1))
            board.push(rng.choices([m for m, _ in viable], weights=weights, k=1)[0])
        if board.is_game_over():
            continue
        judge = best(sf, board, 12000)[0]
        if abs(judge[1]) > 110:
            continue
        key = " ".join(board.fen().split()[:4])
        if key in seen:
            continue
        seen.add(key)
        out.append(board.fen())
    if len(out) != count:
        raise RuntimeError(f"generated {len(out)} of {count} requested positions")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--strategy", required=True,
                    choices=["monolith", "portfolio2", "portfolio3", "duel-base", "duel-stockfish", "committee3"])
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--total-nodes", type=int, default=80000)
    ap.add_argument("--oracle-nodes", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=26081701)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    kernel = chess.engine.SimpleEngine.popen_uci(args.kernel)
    base = chess.engine.SimpleEngine.popen_uci(args.base)
    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    kernel.configure(FUNDAMENTALS)
    base.configure(FUNDAMENTALS)
    sf.configure(PLAIN)

    rows = []
    try:
        fens = generate_positions(sf, args.positions, args.seed)
        for i, fen in enumerate(fens, 1):
            board = chess.Board(fen)
            baseline_move, baseline_meta = choose_monolith(kernel, base, sf, board, args.total_nodes)
            cand_move, cand_meta = choose(args.strategy, kernel, base, sf, board, args.total_nodes)

            oracle_best = best(sf, board, args.oracle_nodes)[0]
            oracle_move, oracle_score = oracle_best[0], oracle_best[1]
            bscore = best(sf, board, args.oracle_nodes, root_moves=[baseline_move])[0][1]
            cscore = best(sf, board, args.oracle_nodes, root_moves=[cand_move])[0][1]
            breg = max(0, oracle_score - bscore)
            creg = max(0, oracle_score - cscore)
            rows.append({
                "index": i,
                "fen": fen,
                "oracle_move": oracle_move.uci(),
                "oracle_score": oracle_score,
                "baseline_move": baseline_move.uci(),
                "baseline_score": bscore,
                "baseline_regret_cp": breg,
                "candidate_move": cand_move.uci(),
                "candidate_score": cscore,
                "candidate_regret_cp": creg,
                "candidate_meta": cand_meta,
            })
            print(f"{i:02d}/{len(fens)} base={baseline_move.uci()} r={breg} cand={cand_move.uci()} r={creg} oracle={oracle_move.uci()}", flush=True)
    finally:
        kernel.quit(); base.quit(); sf.quit()

    br = [r["baseline_regret_cp"] for r in rows]
    cr = [r["candidate_regret_cp"] for r in rows]
    out = {
        "strategy": args.strategy,
        "positions": len(rows),
        "total_nodes_per_choice": args.total_nodes,
        "oracle_nodes_per_grade": args.oracle_nodes,
        "baseline": {
            "mean_regret": statistics.fmean(br),
            "median_regret": statistics.median(br),
            "max_regret": max(br),
            "zero_count": sum(x == 0 for x in br),
            "oracle_move_agreement": sum(r["baseline_move"] == r["oracle_move"] for r in rows) / len(rows),
        },
        "candidate": {
            "mean_regret": statistics.fmean(cr),
            "median_regret": statistics.median(cr),
            "max_regret": max(cr),
            "zero_count": sum(x == 0 for x in cr),
            "oracle_move_agreement": sum(r["candidate_move"] == r["oracle_move"] for r in rows) / len(rows),
        },
        "mean_regret_delta_candidate_minus_baseline": statistics.fmean(cr) - statistics.fmean(br),
        "candidate_better_positions": sum(c < b for b, c in zip(br, cr)),
        "candidate_worse_positions": sum(c > b for b, c in zip(br, cr)),
        "same_positions": sum(c == b for b, c in zip(br, cr)),
        "rows": rows,
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k:v for k,v in out.items() if k != "rows"}, indent=2))

if __name__ == "__main__":
    main()

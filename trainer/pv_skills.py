"""Generate, evaluate, and preserve explicit short-PV skills for Leviathan P006.

A PV skill is a 2-6 ply proposal from a bounded teacher search. The full line is
stored losslessly for analysis and future sequence-aware routing. The current
live engine is deliberately conservative: only validated first-move evidence is
allowed to become an Atlas ordering hint; alpha-beta remains authoritative.

This tool therefore closes the data/artifact side of P006 without pretending an
unvalidated sequence model has earned live search authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine


def cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    return score.pov(turn).score(mate_score=100000) or 0


def analyse(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int, multipv: int) -> list[dict[str, Any]]:
    infos = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=multipv)
    if isinstance(infos, dict):
        infos = [infos]
    rows = []
    for rank, info in enumerate(infos, 1):
        pv = list(info.get("pv", []))
        rows.append({
            "rank": rank,
            "score_cp": cp(info["score"], board.turn),
            "depth": int(info.get("depth", 0)),
            "seldepth": int(info.get("seldepth", 0)),
            "nodes": int(info.get("nodes", 0)),
            "pv": [m.uci() for m in pv],
        })
    return rows


def generate(args: argparse.Namespace) -> None:
    fens = [x.strip() for x in args.fens.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.max_positions:
        fens = fens[: args.max_positions]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    count = 0
    try:
        with args.out.open("w", encoding="utf-8") as out:
            for position_id, fen in enumerate(fens):
                board = chess.Board(fen)
                for item in analyse(engine, board, args.nodes, args.multipv):
                    line = item["pv"][: args.plies]
                    if len(line) < 2:
                        continue
                    row = {
                        "position_id": str(position_id),
                        "fen": fen,
                        "teacher_nodes": args.nodes,
                        "skill_plies": len(line),
                        **{k: v for k, v in item.items() if k != "pv"},
                        "pv": line,
                    }
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                    count += 1
    finally:
        engine.quit()
    print(json.dumps({"positions": len(fens), "skills": count, "out": str(args.out)}))


def evaluate(args: argparse.Namespace) -> None:
    rows = [json.loads(x) for x in args.skills.read_text(encoding="utf-8").splitlines() if x.strip()]
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    total = first_match = prefix_sum = 0
    score_regret = []
    try:
        for row in rows:
            if int(row.get("rank", 1)) != 1:
                continue
            board = chess.Board(row["fen"])
            deep = analyse(engine, board, args.nodes, 1)[0]
            deep_pv = deep["pv"][: int(row["skill_plies"])]
            skill = row["pv"]
            total += 1
            first_match += bool(skill and deep_pv and skill[0] == deep_pv[0])
            prefix = 0
            for a, b in zip(skill, deep_pv):
                if a != b:
                    break
                prefix += 1
            prefix_sum += prefix
            score_regret.append(abs(int(row["score_cp"]) - int(deep["score_cp"])))
    finally:
        engine.quit()
    report = {
        "skills": total,
        "deep_nodes": args.nodes,
        "first_move_stability": first_match / total if total else 0.0,
        "mean_matching_prefix_plies": prefix_sum / total if total else 0.0,
        "mean_abs_score_regret_cp": sum(score_regret) / len(score_regret) if score_regret else 0.0,
    }
    print(json.dumps(report, sort_keys=True))


def pack(args: argparse.Namespace) -> None:
    """Pack full short lines into a content-addressed LVPS1 artifact.

    LVPS1 is deliberately analysis/proposal storage, not a hot-loop format.
    A future sequence-aware runtime must prove that consuming the extra plies is
    worth its cost before authority expands beyond first-move ordering.
    """
    rows = [json.loads(x) for x in args.skills.read_text(encoding="utf-8").splitlines() if x.strip()]
    canonical = []
    for row in rows:
        pv = row.get("pv", [])
        if not 2 <= len(pv) <= 6:
            continue
        canonical.append({
            "fen": row["fen"],
            "rank": int(row.get("rank", 1)),
            "score_cp": int(row.get("score_cp", 0)),
            "pv": list(pv),
            "teacher_nodes": int(row.get("teacher_nodes", 0)),
        })
    canonical.sort(key=lambda r: (r["fen"], r["rank"], r["pv"]))
    payload = "LVPS1\n" + "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in canonical) + "\n"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps({"format": "LVPS1", "sha256": digest, "skills": len(canonical)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"format": "LVPS1", "sha256": digest, "skills": len(canonical)}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--engine", required=True)
    g.add_argument("--fens", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--nodes", type=int, default=50000)
    g.add_argument("--multipv", type=int, default=3)
    g.add_argument("--plies", type=int, choices=range(2, 7), default=6)
    g.add_argument("--max-positions", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--engine", required=True)
    e.add_argument("--skills", type=Path, required=True)
    e.add_argument("--nodes", type=int, default=500000)
    p = sub.add_parser("pack")
    p.add_argument("--skills", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args)
    elif args.cmd == "evaluate":
        evaluate(args)
    else:
        pack(args)


if __name__ == "__main__":
    main()

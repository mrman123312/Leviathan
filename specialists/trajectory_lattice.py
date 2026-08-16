#!/usr/bin/env python3
"""
Adversarial Trajectory Lattice (ATL) research prototype.

This is intentionally outside the trusted search core. It tests whether a bounded
set of repeatedly re-planned future trajectories can improve root decisions at a
fixed advisory node budget. It does not claim exhaustive search or proof.

The local engine remains the tactical oracle. ATL changes the representation of
lookahead: short PV segments are stitched into longer chronological futures and
multiple paths are retained when the side to move has close alternatives.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import chess
import chess.engine


MATE_CP = 100000


def cp_for(info: dict, pov: chess.Color) -> int:
    score = info["score"].pov(pov)
    v = score.score(mate_score=MATE_CP)
    return int(v if v is not None else 0)


@dataclass
class PathState:
    board: chess.Board
    moves: list[str] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    nodes: int = 0

    def clone(self) -> "PathState":
        return PathState(self.board.copy(stack=True), list(self.moves), list(self.scores), self.nodes)


def analyse_multi(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int, multipv: int):
    infos = engine.analyse(board, chess.engine.Limit(nodes=max(1, nodes)), multipv=max(1, multipv))
    return infos if isinstance(infos, list) else [infos]


def apply_segment(path: PathState, info: dict, segment_plies: int, root_color: chess.Color) -> PathState:
    q = path.clone()
    q.scores.append(cp_for(info, root_color))
    pv = info.get("pv", [])[: max(1, segment_plies)]
    for mv in pv:
        if q.board.is_game_over(claim_draw=True) or mv not in q.board.legal_moves:
            break
        q.moves.append(mv.uci())
        q.board.push(mv)
    return q


def path_metric(path: PathState, mode: str) -> float:
    if not path.scores:
        return -1e9
    if mode == "terminal":
        return float(path.scores[-1])
    if mode == "worst":
        return float(min(path.scores))
    if mode == "consensus":
        tail = path.scores[-min(3, len(path.scores)) :]
        return float(sorted(tail)[len(tail) // 2])
    if mode == "stable":
        tail = path.scores[-min(4, len(path.scores)) :]
        mean = sum(tail) / len(tail)
        var = sum((x - mean) ** 2 for x in tail) / len(tail)
        # Penalize forecasts that are still violently changing at the horizon.
        return float(path.scores[-1] - 0.20 * math.sqrt(var))
    raise ValueError(mode)


def choose_beam(paths: list[PathState], root_color: chess.Color, side_to_move: chess.Color,
                beam_width: int, metric: str) -> list[PathState]:
    # Root side preserves its most promising continuations. Opponent side preserves
    # the continuations that are worst for root. This is a bounded minimax envelope.
    reverse = side_to_move == root_color
    return sorted(paths, key=lambda p: path_metric(p, metric), reverse=reverse)[:beam_width]


def extend_candidate(engine: chess.engine.SimpleEngine,
                     root_after_move: chess.Board,
                     root_color: chess.Color,
                     budget_nodes: int,
                     horizon_plies: int,
                     beam_width: int,
                     branch_width: int,
                     segment_plies: int,
                     metric: str) -> tuple[float, dict]:
    paths = [PathState(root_after_move.copy(stack=True))]
    spent = 0
    chronological = 0

    # A segment is a macro-edge: several PV plies are traversed for one tactical
    # re-planning probe. This is how chronological horizon can exceed decision depth.
    rounds = max(1, math.ceil(max(1, horizon_plies - 1) / max(1, segment_plies)))
    probes_left = max(1, rounds * beam_width)

    for _ in range(rounds):
        if spent >= budget_nodes or not paths:
            break
        expanded: list[PathState] = []
        side = paths[0].board.turn
        live = [p for p in paths if not p.board.is_game_over(claim_draw=True)]
        if not live:
            break

        per_probe = max(32, (budget_nodes - spent) // max(1, probes_left))
        for p in paths:
            if p.board.is_game_over(claim_draw=True):
                outcome = p.board.outcome(claim_draw=True)
                if outcome and outcome.winner is None:
                    p.scores.append(0)
                elif outcome:
                    p.scores.append(MATE_CP if outcome.winner == root_color else -MATE_CP)
                expanded.append(p)
                continue

            infos = analyse_multi(engine, p.board, per_probe, min(branch_width, p.board.legal_moves.count()))
            # python-chess reports cumulative nodes for the analyse call. Treat the
            # requested limit as the accounting unit so profiles remain comparable.
            spent += per_probe
            probes_left = max(0, probes_left - 1)
            for info in infos:
                q = apply_segment(p, info, segment_plies, root_color)
                q.nodes += per_probe
                expanded.append(q)

            if spent >= budget_nodes:
                break

        if not expanded:
            break
        side = paths[0].board.turn
        paths = choose_beam(expanded, root_color, side, beam_width, metric)
        chronological = max((len(p.moves) for p in paths), default=chronological)

    if not paths:
        return -1e9, {"spent_nodes": spent, "paths": []}

    # For a candidate root move, evaluate the retained future envelope
    # pessimistically: the lowest surviving trajectory is the adversarial score.
    vals = [path_metric(p, metric) for p in paths]
    score = min(vals)
    return score, {
        "spent_nodes": spent,
        "chronological_plies": chronological + 1,
        "paths": [
            {"score": path_metric(p, metric), "scores": p.scores, "moves": p.moves}
            for p in paths
        ],
    }


def recommend(engine: chess.engine.SimpleEngine,
              board: chess.Board,
              total_nodes: int,
              root_candidates: int,
              horizon_plies: int,
              beam_width: int,
              branch_width: int,
              segment_plies: int,
              metric: str) -> dict:
    root_color = board.turn
    seed_nodes = max(1000, total_nodes // 3)
    infos = analyse_multi(engine, board, seed_nodes, min(root_candidates, board.legal_moves.count()))
    candidates = []
    remaining = max(0, total_nodes - seed_nodes)
    per_candidate = max(64, remaining // max(1, len(infos)))

    for info in infos:
        pv = info.get("pv", [])
        if not pv:
            continue
        move = pv[0]
        b = board.copy(stack=True)
        b.push(move)
        traj_score, detail = extend_candidate(
            engine, b, root_color, per_candidate, horizon_plies, beam_width,
            branch_width, segment_plies, metric
        )
        candidates.append({
            "move": move.uci(),
            "seed_score": cp_for(info, root_color),
            "trajectory_score": traj_score,
            "detail": detail,
        })

    candidates.sort(key=lambda x: (x["trajectory_score"], x["seed_score"]), reverse=True)
    return {
        "move": candidates[0]["move"] if candidates else None,
        "candidates": candidates,
        "seed_nodes": seed_nodes,
        "budget_nodes": total_nodes,
        "horizon_plies": horizon_plies,
        "beam_width": beam_width,
        "branch_width": branch_width,
        "segment_plies": segment_plies,
        "metric": metric,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--fen", required=True)
    ap.add_argument("--nodes", type=int, default=24000)
    ap.add_argument("--root-candidates", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--branch", type=int, default=2)
    ap.add_argument("--segment", type=int, default=4)
    ap.add_argument("--metric", choices=["terminal", "worst", "consensus", "stable"], default="stable")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    try:
        eng.configure({"Threads": 1, "Hash": 64})
        result = recommend(eng, chess.Board(args.fen), args.nodes, args.root_candidates,
                           args.horizon, args.beam, args.branch, args.segment, args.metric)
    finally:
        eng.quit()

    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P19.1 certified-survival experiment.

Goal: a Leviathan move is not authoritative merely because one search likes it.
Every move must survive a deterministic, fixed-node, heterogeneous challenge:
P09 and the frozen Stockfish baseline both get veto power. If no move clears the
safety certificate, the experiment fails as UNCERTIFIED instead of gambling.

This is still an engineering certificate, not a mathematical solution of chess.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.engine

try:
    import run_p19_survival_match as v1
except ImportError:
    from . import run_p19_survival_match as v1

MATE_SCORE = 100000

LOSS_SENTINELS = list(v1.LOSS_SENTINELS) + [
    {
        "name": "p19-v1-ne4-nc6-micro",
        "fen": "r1bqkb1r/ppp2ppp/2n2n2/8/3pN3/3Q2P1/PPP1PP1P/R1B1KBNR w KQkq - 2 7",
        "leviathan_white": True,
    }
]


def score_cp(info: dict[str, Any], color: chess.Color) -> int:
    s = info.get("score")
    if s is None:
        return -MATE_SCORE
    v = s.pov(color).score(mate_score=MATE_SCORE)
    return int(v if v is not None else -MATE_SCORE)


def infos_list(x):
    return x if isinstance(x, list) else [x]


def root_map(engine, board, nodes: int, multipv: int, game_token: str) -> dict[chess.Move, int]:
    mpv = max(1, min(multipv, board.legal_moves.count()))
    infos = infos_list(
        engine.analyse(
            board,
            chess.engine.Limit(nodes=max(1, nodes)),
            multipv=mpv,
            game=game_token,
        )
    )
    out: dict[chess.Move, int] = {}
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        m = pv[0]
        val = score_cp(info, board.turn)
        if m not in out or val > out[m]:
            out[m] = val
    return out


def hostile_eval(engine, board_after, leviathan_color: chess.Color, nodes: int, game_token: str) -> int:
    info = engine.analyse(
        board_after,
        chess.engine.Limit(nodes=max(1, nodes)),
        multipv=1,
        game=game_token,
    )
    if isinstance(info, list):
        info = info[0] if info else {}
    return score_cp(info, leviathan_color)


def immediate_claim_draw(board: chess.Board, move: chess.Move) -> bool:
    if move not in board.legal_moves:
        return False
    b = board.copy(stack=True)
    b.push(move)
    return b.can_claim_threefold_repetition() or b.can_claim_fifty_moves() or b.is_stalemate()


class UncertifiedPosition(RuntimeError):
    def __init__(self, telemetry: dict[str, Any]):
        super().__init__("no move cleared the P19.1 survival certificate")
        self.telemetry = telemetry


@dataclass
class Candidate:
    move: chess.Move
    root_p09: int | None = None
    root_sf: int | None = None
    post_p09: int | None = None
    post_sf: int | None = None
    deep_p09: int | None = None
    deep_sf: int | None = None
    immediate_draw: bool = False
    certificate: float = -1e9
    spread: int = 0
    hard_refuted: bool = False
    stages: list[str] = field(default_factory=list)

    def values(self) -> list[int]:
        return [
            int(x)
            for x in (
                self.root_p09,
                self.root_sf,
                self.post_p09,
                self.post_sf,
                self.deep_p09,
                self.deep_sf,
            )
            if x is not None
        ]


class CertifiedSurvivalFunnel:
    def __init__(
        self,
        engine_path: str,
        stockfish_path: str,
        hash_mb: int,
        root_nodes: int,
        verify_nodes: int,
        panic_root_nodes: int,
        panic_verify_nodes: int,
        finalist_nodes: int,
        safety_floor_cp: int,
        disagreement_cp: int,
        disagreement_weight: float,
        candidate_mpv: int,
        panic_finalists: int,
        draw_lock_cp: int,
    ):
        self.p09 = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=60.0)
        self.sf_guard = chess.engine.SimpleEngine.popen_uci(stockfish_path, timeout=60.0)
        # Deterministic critics: one thread, fixed nodes, independent process families.
        v1.configure(self.p09, 1, max(16, hash_mb // 2))
        v1.configure(self.sf_guard, 1, max(16, hash_mb // 2))
        self.root_nodes = root_nodes
        self.verify_nodes = verify_nodes
        self.panic_root_nodes = panic_root_nodes
        self.panic_verify_nodes = panic_verify_nodes
        self.finalist_nodes = finalist_nodes
        self.safety_floor_cp = safety_floor_cp
        self.disagreement_cp = disagreement_cp
        self.disagreement_weight = disagreement_weight
        self.candidate_mpv = candidate_mpv
        self.panic_finalists = panic_finalists
        self.draw_lock_cp = draw_lock_cp
        self.serial = 0

    def close(self):
        v1.safe_quit(self.p09)
        v1.safe_quit(self.sf_guard)

    def _tok(self, base: str) -> str:
        self.serial += 1
        return f"{base}-{self.serial}"

    def _recompute_certificate(self, c: Candidate) -> None:
        vals = c.values()
        if not vals:
            c.certificate = -1e9
            c.spread = MATE_SCORE * 2
            c.hard_refuted = True
            return
        lo, hi = min(vals), max(vals)
        c.spread = hi - lo
        c.hard_refuted = lo <= -MATE_SCORE // 2
        c.certificate = float(lo) - self.disagreement_weight * float(c.spread)

    def _root_candidates(self, board: chess.Board, nodes: int, multipv: int, token: str):
        p = root_map(self.p09, board, nodes, multipv, self._tok(token + "-p09-root"))
        s = root_map(self.sf_guard, board, nodes, multipv, self._tok(token + "-sf-root"))
        union = list(dict.fromkeys(list(p.keys()) + list(s.keys())))
        return p, s, union

    def _build_candidates(self, board, p, s, union) -> list[Candidate]:
        p_best = max(p.values()) if p else -MATE_SCORE
        s_best = max(s.values()) if s else -MATE_SCORE
        out = []
        for m in union:
            c = Candidate(
                move=m,
                root_p09=p.get(m, p_best - 140),
                root_sf=s.get(m, s_best - 140),
                immediate_draw=immediate_claim_draw(board, m),
                stages=["root"],
            )
            self._recompute_certificate(c)
            out.append(c)
        out.sort(key=lambda x: (x.certificate, -x.spread), reverse=True)
        return out

    def _verify(self, board: chess.Board, candidates: list[Candidate], nodes: int, token: str, deep: bool):
        lev_color = board.turn
        for i, c in enumerate(candidates):
            b = board.copy(stack=True)
            b.push(c.move)
            p = hostile_eval(self.p09, b, lev_color, nodes, self._tok(f"{token}-p09-{i}"))
            s = hostile_eval(self.sf_guard, b, lev_color, nodes, self._tok(f"{token}-sf-{i}"))
            if deep:
                c.deep_p09, c.deep_sf = p, s
                c.stages.append("deep")
            else:
                c.post_p09, c.post_sf = p, s
                c.stages.append("post")
            self._recompute_certificate(c)
        candidates.sort(key=lambda x: (x.certificate, -x.spread), reverse=True)

    def _safe(self, c: Candidate) -> bool:
        return (
            not c.hard_refuted
            and c.certificate >= self.safety_floor_cp
            and c.spread <= max(self.disagreement_cp * 3, 120)
        )

    def _row(self, c: Candidate) -> dict[str, Any]:
        return {
            "move": c.move.uci(),
            "root_p09": c.root_p09,
            "root_sf": c.root_sf,
            "post_p09": c.post_p09,
            "post_sf": c.post_sf,
            "deep_p09": c.deep_p09,
            "deep_sf": c.deep_sf,
            "certificate_cp": round(c.certificate, 2),
            "spread_cp": c.spread,
            "hard_refuted": c.hard_refuted,
            "safe": self._safe(c),
            "immediate_draw": c.immediate_draw,
            "stages": c.stages,
        }

    def choose(self, board: chess.Board, token: str):
        # Stage A: deterministic top-move census from two different engine families.
        p, s, union = self._root_candidates(board, self.root_nodes, self.candidate_mpv, token + "-normal")
        if not union:
            raise UncertifiedPosition({"reason": "no_root_candidates", "fen": board.fen()})
        candidates = self._build_candidates(board, p, s, union)

        # Verify the best conservative candidates by searching after our move, i.e.
        # with the opponent to move and every incentive to refute it.
        normal = candidates[: min(4, len(candidates))]
        self._verify(board, normal, self.verify_nodes, token + "-normal-post", deep=False)

        immediate = [c for c in normal if c.immediate_draw]
        safe = [c for c in normal if self._safe(c)]
        top = normal[0] if normal else candidates[0]
        needs_panic = (
            not safe
            or top.spread > self.disagreement_cp
            or top.certificate < self.safety_floor_cp + 20
        )

        # If a legal draw is already available and our best certified state is not
        # clearly winning, take the proof we already possess instead of speculating.
        if immediate and top.certificate <= self.draw_lock_cp:
            chosen = max(immediate, key=lambda c: c.certificate)
            return chosen.move, {
                "version": "p19-certified-v2",
                "reason": "draw_lock",
                "fen": board.fen(),
                "chosen": chosen.move.uci(),
                "candidates": [self._row(c) for c in normal],
            }

        if not needs_panic and safe:
            chosen = max(safe, key=lambda c: (c.certificate, -c.spread))
            return chosen.move, {
                "version": "p19-certified-v2",
                "reason": "normal_certificate",
                "fen": board.fen(),
                "chosen": chosen.move.uci(),
                "candidates": [self._row(c) for c in normal],
            }

        # Stage B: uncertainty is not permission to guess. Expand the root census
        # to every legal move, then deeply challenge the best worst-case options.
        legal_n = board.legal_moves.count()
        pp, ss, all_union = self._root_candidates(
            board, self.panic_root_nodes, legal_n, token + "-panic"
        )
        # Ensure literally every legal move is represented even if MultiPV output is incomplete.
        seen = set(all_union)
        for m in board.legal_moves:
            if m not in seen:
                all_union.append(m)
                seen.add(m)
        panic = self._build_candidates(board, pp, ss, all_union)
        finalists = panic[: min(self.panic_finalists, len(panic))]
        self._verify(board, finalists, self.panic_verify_nodes, token + "-panic-post", deep=False)
        finalists = finalists[: min(3, len(finalists))]
        self._verify(board, finalists, self.finalist_nodes, token + "-final", deep=True)

        draw_finalists = [c for c in finalists if c.immediate_draw]
        safe_finalists = [c for c in finalists if self._safe(c)]
        best = finalists[0] if finalists else None
        if draw_finalists and best is not None and best.certificate <= self.draw_lock_cp:
            chosen = max(draw_finalists, key=lambda c: c.certificate)
            return chosen.move, {
                "version": "p19-certified-v2",
                "reason": "panic_draw_lock",
                "fen": board.fen(),
                "chosen": chosen.move.uci(),
                "candidates": [self._row(c) for c in finalists],
            }

        if safe_finalists:
            chosen = max(safe_finalists, key=lambda c: (c.certificate, -c.spread))
            return chosen.move, {
                "version": "p19-certified-v2",
                "reason": "panic_certificate",
                "fen": board.fen(),
                "chosen": chosen.move.uci(),
                "candidates": [self._row(c) for c in finalists],
            }

        telemetry = {
            "version": "p19-certified-v2",
            "reason": "UNCERTIFIED_POSITION",
            "fen": board.fen(),
            "safety_floor_cp": self.safety_floor_cp,
            "disagreement_cp": self.disagreement_cp,
            "normal": [self._row(c) for c in normal],
            "panic_finalists": [self._row(c) for c in finalists],
        }
        raise UncertifiedPosition(telemetry)


def append_jsonl(path: Path, row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def summary(rows):
    return v1.summary(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="P19.1 deterministic heterogeneous survival certificate")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--opponent-engine", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--opponent-threads", type=int, default=6)
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--opponent-movetime-ms", type=int, default=500)
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--opening-plies", type=int, default=10)
    ap.add_argument("--opening-nodes", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--sentinel-repeats", type=int, default=5)
    ap.add_argument("--root-nodes", type=int, default=24000)
    ap.add_argument("--verify-nodes", type=int, default=16000)
    ap.add_argument("--panic-root-nodes", type=int, default=120000)
    ap.add_argument("--panic-verify-nodes", type=int, default=60000)
    ap.add_argument("--finalist-nodes", type=int, default=240000)
    ap.add_argument("--candidate-mpv", type=int, default=8)
    ap.add_argument("--panic-finalists", type=int, default=6)
    ap.add_argument("--safety-floor-cp", type=int, default=-20)
    ap.add_argument("--disagreement-cp", type=int, default=55)
    ap.add_argument("--disagreement-weight", type=float, default=0.35)
    ap.add_argument("--draw-lock-cp", type=int, default=120)
    a = ap.parse_args()
    if a.games <= 0 or a.games % 2:
        raise SystemExit("--games must be positive and even")

    identity = {
        "version": "p19-certified-v2",
        "engine_sha256": v1.sha256_file(Path(a.engine)),
        "stockfish_sha256": v1.sha256_file(Path(a.opponent_engine)),
        "games": a.games,
        "opponent_threads": a.opponent_threads,
        "opponent_movetime_ms": a.opponent_movetime_ms,
        "root_nodes": a.root_nodes,
        "verify_nodes": a.verify_nodes,
        "panic_root_nodes": a.panic_root_nodes,
        "panic_verify_nodes": a.panic_verify_nodes,
        "finalist_nodes": a.finalist_nodes,
        "safety_floor_cp": a.safety_floor_cp,
        "disagreement_cp": a.disagreement_cp,
        "seed": a.seed,
        "sentinel_repeats": a.sentinel_repeats,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    out = Path(a.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    v1.jprint({"event": "P19_1_CONFIG", "run_id": run_id, **identity})

    funnel = CertifiedSurvivalFunnel(
        a.engine, a.opponent_engine, a.hash,
        a.root_nodes, a.verify_nodes, a.panic_root_nodes, a.panic_verify_nodes,
        a.finalist_nodes, a.safety_floor_cp, a.disagreement_cp,
        a.disagreement_weight, a.candidate_mpv, a.panic_finalists, a.draw_lock_cp,
    )
    sf = chess.engine.SimpleEngine.popen_uci(a.opponent_engine, timeout=60.0)
    v1.configure(sf, a.opponent_threads, a.hash)
    decisions = out / "certificates.jsonl"
    rows = []
    try:
        v1.jprint({"event": "P19_1_SENTINEL_GATE_START", "cases": len(LOSS_SENTINELS), "repeats": a.sentinel_repeats})
        idx = 0
        for case in LOSS_SENTINELS:
            for rep in range(1, a.sentinel_repeats + 1):
                idx += 1
                try:
                    r = v1.play_game(
                        funnel, sf, case["fen"], bool(case["leviathan_white"]),
                        f"p19-1-sentinel-{idx}", a.opponent_movetime_ms, a.max_plies, decisions,
                    )
                except UncertifiedPosition as exc:
                    fail = {
                        "event": "P19_1_UNCERTIFIED_SENTINEL",
                        "sentinel": case["name"],
                        "repeat": rep,
                        "certificate": exc.telemetry,
                    }
                    (out / "uncertified.json").write_text(json.dumps(fail, indent=2, sort_keys=True), encoding="utf-8")
                    v1.jprint(fail)
                    return 41
                r.update({"sentinel": case["name"], "repeat": rep})
                append_jsonl(out / "sentinels.jsonl", r)
                v1.jprint({"event": "P19_1_SENTINEL_COMPLETE", **r})
                if r["score_leviathan"] == 0.0:
                    v1.jprint({"event": "P19_1_SENTINEL_LOSS", **r})
                    return 21
        v1.jprint({"event": "P19_1_SENTINEL_GATE_PASSED", "games": idx, "losses": 0})

        openings = v1.generate_openings(
            a.opponent_engine, out / "openings.fen", a.games // 2,
            a.opening_plies, a.seed, a.opening_nodes,
        )
        for game in range(1, a.games + 1):
            fen = openings[(game - 1) // 2]
            lev_white = game % 2 == 1
            try:
                r = v1.play_game(
                    funnel, sf, fen, lev_white, f"p19-1-game-{game}",
                    a.opponent_movetime_ms, a.max_plies, decisions,
                )
            except UncertifiedPosition as exc:
                fail = {
                    "event": "P19_1_UNCERTIFIED_FRESH",
                    "game": game,
                    "fen": exc.telemetry.get("fen"),
                    "certificate": exc.telemetry,
                    "summary": summary(rows),
                }
                (out / "uncertified.json").write_text(json.dumps(fail, indent=2, sort_keys=True), encoding="utf-8")
                v1.jprint(fail)
                return 42
            r["game"] = game
            append_jsonl(out / "games.jsonl", r)
            rows.append(r)
            v1.jprint({"event": "P19_1_GAME_COMPLETE", **r, "cumulative": summary(rows)})
            if r["score_leviathan"] == 0.0:
                fail = {"event": "P19_1_ZERO_LOSS_GATE_FAILED", **r, "summary": summary(rows)}
                (out / "failure.json").write_text(json.dumps(fail, indent=2, sort_keys=True), encoding="utf-8")
                v1.jprint(fail)
                return 31

        final = {
            "event": "P19_1_ZERO_LOSS_GATE_PASSED",
            "run_id": run_id,
            "known_loss_sentinels": len(LOSS_SENTINELS) * a.sentinel_repeats,
            "summary": summary(rows),
        }
        (out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
        v1.jprint(final)
        return 0
    finally:
        funnel.close()
        v1.safe_quit(sf)


if __name__ == "__main__":
    raise SystemExit(main())

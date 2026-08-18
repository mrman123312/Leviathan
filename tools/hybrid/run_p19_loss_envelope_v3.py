#!/usr/bin/env python3
"""P19.2 loss-envelope survival experiment.

P19.1 correctly refused to gamble, but its absolute centipawn safety floor was
the wrong invariant: a drawable position may legitimately evaluate below zero.
P19.2 instead asks whether a move *increases the estimated loss envelope*.

For every candidate:
  1. P09 and frozen Stockfish establish a fixed-node WDL baseline for the
     current position from Leviathan's point of view.
  2. After the candidate, both critics census all legal opponent replies.
  3. The most dangerous replies are pushed and independently re-evaluated by
     both critics at deeper fixed-node budgets.
  4. Mate/tablebase-style decisive losses are hard vetoes.
  5. Otherwise the candidate passes only if its worst observed loss probability
     stays within a small per-mille allowance of the current-position envelope.

This remains an empirical engineering certificate, not a proof that chess is
solved. Uncertified positions fail closed and become new counterexamples.
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


def jprint(obj: Any) -> None:
    print(json.dumps(obj, sort_keys=True), flush=True)


def infos_list(x):
    return x if isinstance(x, list) else [x]


def score_cp(info: dict[str, Any], color: chess.Color) -> int:
    s = info.get("score")
    if s is None:
        return -MATE_SCORE
    v = s.pov(color).score(mate_score=MATE_SCORE)
    return int(v if v is not None else -MATE_SCORE)


def mate_against(info: dict[str, Any], color: chess.Color) -> bool:
    s = info.get("score")
    if s is None:
        return False
    try:
        p = s.pov(color)
        return p.is_mate() and (p.mate() is not None and p.mate() <= 0)
    except Exception:
        return score_cp(info, color) <= -MATE_SCORE // 2


def loss_permille(info: dict[str, Any], color: chess.Color, ply: int) -> int:
    """Loss probability in per-mille, preferring engine-reported UCI WDL."""
    w = info.get("wdl")
    if w is not None:
        try:
            x = w.pov(color)
            total = max(1, int(x.total()))
            return int(round(1000.0 * int(x.losses) / total))
        except Exception:
            pass
    s = info.get("score")
    if s is not None:
        try:
            x = s.pov(color).wdl(model="sf", ply=max(0, ply))
            total = max(1, int(x.total()))
            return int(round(1000.0 * int(x.losses) / total))
        except Exception:
            pass
    cp = score_cp(info, color)
    if cp <= -MATE_SCORE // 2:
        return 1000
    if cp >= MATE_SCORE // 2:
        return 0
    # Last-resort smooth fallback only if WDL is unavailable.
    return int(round(1000.0 / (1.0 + math.exp((cp + 120.0) / 85.0))))


@dataclass
class View:
    cp: int
    loss_pm: int
    mate_loss: bool
    depth: int = 0
    nodes: int = 0


def view(info: dict[str, Any], color: chess.Color, ply: int) -> View:
    return View(
        cp=score_cp(info, color),
        loss_pm=loss_permille(info, color, ply),
        mate_loss=mate_against(info, color),
        depth=int(info.get("depth") or 0),
        nodes=int(info.get("nodes") or 0),
    )


def analyse_one(engine, board: chess.Board, nodes: int, color: chess.Color, token: str) -> View:
    info = engine.analyse(
        board,
        chess.engine.Limit(nodes=max(1, nodes)),
        multipv=1,
        game=token,
    )
    if isinstance(info, list):
        info = info[0] if info else {}
    return view(info, color, board.ply())


def root_views(engine, board: chess.Board, nodes: int, multipv: int, color: chess.Color, token: str) -> dict[chess.Move, View]:
    mpv = max(1, min(multipv, board.legal_moves.count()))
    infos = infos_list(engine.analyse(
        board,
        chess.engine.Limit(nodes=max(1, nodes)),
        multipv=mpv,
        game=token,
    ))
    out: dict[chess.Move, View] = {}
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        m = pv[0]
        vw = view(info, color, board.ply())
        old = out.get(m)
        if old is None or vw.loss_pm < old.loss_pm or (vw.loss_pm == old.loss_pm and vw.cp > old.cp):
            out[m] = vw
    return out


def immediate_claim_draw(board: chess.Board, move: chess.Move) -> bool:
    if move not in board.legal_moves:
        return False
    b = board.copy(stack=True)
    b.push(move)
    return b.can_claim_threefold_repetition() or b.can_claim_fifty_moves() or b.is_stalemate()


class UncertifiedPosition(RuntimeError):
    def __init__(self, telemetry: dict[str, Any]):
        super().__init__("no move cleared the P19.2 loss-envelope certificate")
        self.telemetry = telemetry


@dataclass
class ReplyRisk:
    move: chess.Move
    broad_p09: View | None = None
    broad_sf: View | None = None
    deep_p09: View | None = None
    deep_sf: View | None = None

    def all_views(self) -> list[View]:
        return [x for x in (self.broad_p09, self.broad_sf, self.deep_p09, self.deep_sf) if x is not None]

    def worst_loss(self) -> int:
        xs = self.all_views()
        return max((x.loss_pm for x in xs), default=1000)

    def worst_cp(self) -> int:
        xs = self.all_views()
        return min((x.cp for x in xs), default=-MATE_SCORE)

    def hard_refuted(self) -> bool:
        return any(x.mate_loss or x.cp <= -MATE_SCORE // 2 for x in self.all_views())


@dataclass
class Candidate:
    move: chess.Move
    root_p09: View | None = None
    root_sf: View | None = None
    immediate_draw: bool = False
    replies: dict[chess.Move, ReplyRisk] = field(default_factory=dict)
    worst_loss_pm: int = 1000
    worst_cp: int = -MATE_SCORE
    risk_delta_pm: int = 1000
    hard_refuted: bool = False
    certified: bool = False


class LossEnvelopeFunnel:
    def __init__(self, engine_path: str, stockfish_path: str, hash_mb: int,
                 baseline_nodes: int, root_nodes: int, reply_nodes: int,
                 deep_reply_nodes: int, panic_root_nodes: int,
                 panic_reply_nodes: int, panic_deep_nodes: int,
                 candidate_mpv: int, dangerous_replies: int,
                 loss_delta_pm: int, max_loss_pm: int, disagreement_pm: int):
        self.p09 = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=90.0)
        self.sf = chess.engine.SimpleEngine.popen_uci(stockfish_path, timeout=90.0)
        v1.configure(self.p09, 1, max(16, hash_mb // 2))
        v1.configure(self.sf, 1, max(16, hash_mb // 2))
        self.baseline_nodes = baseline_nodes
        self.root_nodes = root_nodes
        self.reply_nodes = reply_nodes
        self.deep_reply_nodes = deep_reply_nodes
        self.panic_root_nodes = panic_root_nodes
        self.panic_reply_nodes = panic_reply_nodes
        self.panic_deep_nodes = panic_deep_nodes
        self.candidate_mpv = candidate_mpv
        self.dangerous_replies = dangerous_replies
        self.loss_delta_pm = loss_delta_pm
        self.max_loss_pm = max_loss_pm
        self.disagreement_pm = disagreement_pm
        self.serial = 0

    def close(self):
        v1.safe_quit(self.p09)
        v1.safe_quit(self.sf)

    def tok(self, base: str) -> str:
        self.serial += 1
        return f"{base}-{self.serial}"

    def baseline(self, board: chess.Board, lev_color: chess.Color, token: str) -> dict[str, Any]:
        p = analyse_one(self.p09, board, self.baseline_nodes, lev_color, self.tok(token + "-base-p09"))
        s = analyse_one(self.sf, board, self.baseline_nodes, lev_color, self.tok(token + "-base-sf"))
        return {
            "p09": p,
            "sf": s,
            "loss_pm": max(p.loss_pm, s.loss_pm),
            "loss_spread_pm": abs(p.loss_pm - s.loss_pm),
            "worst_cp": min(p.cp, s.cp),
            "hard_lost": p.mate_loss or s.mate_loss,
        }

    def root_candidates(self, board: chess.Board, lev_color: chess.Color, nodes: int, multipv: int, token: str) -> list[Candidate]:
        p = root_views(self.p09, board, nodes, multipv, lev_color, self.tok(token + "-p09-root"))
        s = root_views(self.sf, board, nodes, multipv, lev_color, self.tok(token + "-sf-root"))
        union = list(dict.fromkeys(list(p.keys()) + list(s.keys())))
        out = [Candidate(m, p.get(m), s.get(m), immediate_claim_draw(board, m)) for m in union]
        def seed(c: Candidate):
            vals = [x for x in (c.root_p09, c.root_sf) if x is not None]
            return (-(max((x.loss_pm for x in vals), default=1000)), max((x.cp for x in vals), default=-MATE_SCORE))
        out.sort(key=seed, reverse=True)
        return out

    def census_replies(self, board: chess.Board, c: Candidate, lev_color: chess.Color,
                       reply_nodes: int, deep_nodes: int, token: str) -> None:
        after = board.copy(stack=True)
        after.push(c.move)
        if after.is_game_over(claim_draw=True):
            outcome = after.outcome(claim_draw=True)
            if outcome is not None and outcome.winner is None:
                c.worst_loss_pm = 0
                c.worst_cp = 0
                c.risk_delta_pm = 0
                c.certified = True
                return

        legal_n = after.legal_moves.count()
        p = root_views(self.p09, after, reply_nodes, legal_n, lev_color, self.tok(token + "-p09-replies"))
        s = root_views(self.sf, after, reply_nodes, legal_n, lev_color, self.tok(token + "-sf-replies"))
        union = list(dict.fromkeys(list(p.keys()) + list(s.keys())))
        seen = set(union)
        for m in after.legal_moves:
            if m not in seen:
                union.append(m); seen.add(m)
        replies: dict[chess.Move, ReplyRisk] = {}
        p_fallback = max((x.loss_pm for x in p.values()), default=1000)
        s_fallback = max((x.loss_pm for x in s.values()), default=1000)
        for m in union:
            rr = ReplyRisk(m, p.get(m), s.get(m))
            # Missing a reply from a full MultiPV census is uncertainty, never safety.
            if rr.broad_p09 is None:
                rr.broad_p09 = View(-MATE_SCORE // 4, p_fallback, False)
            if rr.broad_sf is None:
                rr.broad_sf = View(-MATE_SCORE // 4, s_fallback, False)
            replies[m] = rr

        dangerous = sorted(replies.values(), key=lambda r: (r.worst_loss(), -r.worst_cp()), reverse=True)
        for i, rr in enumerate(dangerous[: min(self.dangerous_replies, len(dangerous))]):
            leaf = after.copy(stack=True)
            leaf.push(rr.move)
            rr.deep_p09 = analyse_one(self.p09, leaf, deep_nodes, lev_color, self.tok(f"{token}-deep-p09-{i}"))
            rr.deep_sf = analyse_one(self.sf, leaf, deep_nodes, lev_color, self.tok(f"{token}-deep-sf-{i}"))

        c.replies = replies
        c.worst_loss_pm = max((r.worst_loss() for r in replies.values()), default=1000)
        c.worst_cp = min((r.worst_cp() for r in replies.values()), default=-MATE_SCORE)
        c.hard_refuted = any(r.hard_refuted() for r in replies.values())

    def certify(self, c: Candidate, base: dict[str, Any]) -> None:
        if c.immediate_draw:
            c.worst_loss_pm = 0
            c.worst_cp = 0
            c.risk_delta_pm = -int(base["loss_pm"])
            c.hard_refuted = False
            c.certified = True
            return
        c.risk_delta_pm = int(c.worst_loss_pm) - int(base["loss_pm"])
        # Relative invariant first. max_loss_pm is only a catastrophic absolute cap.
        c.certified = (
            not c.hard_refuted
            and c.risk_delta_pm <= self.loss_delta_pm
            and c.worst_loss_pm <= self.max_loss_pm
        )

    def candidate_row(self, c: Candidate) -> dict[str, Any]:
        def vr(x: View | None):
            return None if x is None else {"cp": x.cp, "loss_pm": x.loss_pm, "mate_loss": x.mate_loss, "depth": x.depth, "nodes": x.nodes}
        replies = sorted(c.replies.values(), key=lambda r: (r.worst_loss(), -r.worst_cp()), reverse=True)
        return {
            "move": c.move.uci(),
            "immediate_draw": c.immediate_draw,
            "root_p09": vr(c.root_p09),
            "root_sf": vr(c.root_sf),
            "worst_loss_pm": c.worst_loss_pm,
            "worst_cp": c.worst_cp,
            "risk_delta_pm": c.risk_delta_pm,
            "hard_refuted": c.hard_refuted,
            "certified": c.certified,
            "dangerous_replies": [
                {
                    "move": r.move.uci(),
                    "worst_loss_pm": r.worst_loss(),
                    "worst_cp": r.worst_cp(),
                    "hard_refuted": r.hard_refuted(),
                    "broad_p09": vr(r.broad_p09),
                    "broad_sf": vr(r.broad_sf),
                    "deep_p09": vr(r.deep_p09),
                    "deep_sf": vr(r.deep_sf),
                }
                for r in replies[:6]
            ],
        }

    def choose(self, board: chess.Board, token: str):
        lev_color = board.turn
        base = self.baseline(board, lev_color, token)
        if base["hard_lost"]:
            raise UncertifiedPosition({"version": "p19-loss-envelope-v3", "reason": "BASELINE_FORCED_LOSS", "fen": board.fen()})

        # Guaranteed legal draw dominates every probabilistic certificate.
        draws = [m for m in board.legal_moves if immediate_claim_draw(board, m)]
        if draws:
            chosen = draws[0]
            return chosen, {
                "version": "p19-loss-envelope-v3",
                "reason": "guaranteed_draw_lock",
                "fen": board.fen(),
                "chosen": chosen.uci(),
                "baseline_loss_pm": base["loss_pm"],
            }

        candidates = self.root_candidates(board, lev_color, self.root_nodes, self.candidate_mpv, token + "-normal")
        if not candidates:
            raise UncertifiedPosition({"version": "p19-loss-envelope-v3", "reason": "NO_ROOT_CANDIDATES", "fen": board.fen()})

        normal = candidates[: min(4, len(candidates))]
        for i, c in enumerate(normal):
            self.census_replies(board, c, lev_color, self.reply_nodes, self.deep_reply_nodes, f"{token}-normal-{i}")
            self.certify(c, base)
        safe = [c for c in normal if c.certified]
        if safe:
            chosen = min(safe, key=lambda c: (c.worst_loss_pm, c.risk_delta_pm, -max((x.cp for x in (c.root_p09, c.root_sf) if x is not None), default=-MATE_SCORE)))
            return chosen.move, {
                "version": "p19-loss-envelope-v3",
                "reason": "normal_loss_envelope",
                "fen": board.fen(),
                "chosen": chosen.move.uci(),
                "baseline": {"loss_pm": base["loss_pm"], "loss_spread_pm": base["loss_spread_pm"], "worst_cp": base["worst_cp"]},
                "candidates": [self.candidate_row(c) for c in normal],
            }

        # Panic: census every Leviathan move, then deeply challenge the best envelopes.
        panic = self.root_candidates(board, lev_color, self.panic_root_nodes, board.legal_moves.count(), token + "-panic")
        seen = {c.move for c in panic}
        for m in board.legal_moves:
            if m not in seen:
                panic.append(Candidate(m, immediate_draw=immediate_claim_draw(board, m)))
        for i, c in enumerate(panic):
            # Root ranking may be missing for rare moves; full reply census is authoritative.
            self.census_replies(board, c, lev_color, self.panic_reply_nodes, self.panic_deep_nodes, f"{token}-panic-{i}")
            self.certify(c, base)
        panic.sort(key=lambda c: (c.worst_loss_pm, c.risk_delta_pm, -c.worst_cp))
        safe = [c for c in panic if c.certified]
        if safe:
            chosen = safe[0]
            return chosen.move, {
                "version": "p19-loss-envelope-v3",
                "reason": "panic_loss_envelope",
                "fen": board.fen(),
                "chosen": chosen.move.uci(),
                "baseline": {"loss_pm": base["loss_pm"], "loss_spread_pm": base["loss_spread_pm"], "worst_cp": base["worst_cp"]},
                "candidates": [self.candidate_row(c) for c in panic[:8]],
            }

        raise UncertifiedPosition({
            "version": "p19-loss-envelope-v3",
            "reason": "UNCERTIFIED_LOSS_ENVELOPE",
            "fen": board.fen(),
            "baseline": {"loss_pm": base["loss_pm"], "loss_spread_pm": base["loss_spread_pm"], "worst_cp": base["worst_cp"]},
            "loss_delta_pm": self.loss_delta_pm,
            "max_loss_pm": self.max_loss_pm,
            "candidates": [self.candidate_row(c) for c in panic[:8]],
        })


def append_jsonl(path: Path, row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n"); f.flush()


def play_game(funnel: LossEnvelopeFunnel, sf, fen: str, lev_white: bool, token: str,
              movetime_ms: int, max_plies: int, decision_log: Path):
    board = chess.Board(fen); moves = []
    limit = chess.engine.Limit(time=movetime_ms / 1000.0)
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        lev_turn = board.turn == (chess.WHITE if lev_white else chess.BLACK)
        if lev_turn:
            move, telem = funnel.choose(board, token)
            telem.update({"game": token, "ply": ply + 1})
            append_jsonl(decision_log, telem)
        else:
            result = sf.play(board, limit, game=token, ponder=False)
            move = result.move
        if move is None or move not in board.legal_moves:
            raise RuntimeError(f"illegal/no move at ply {ply+1}: {move} fen={board.fen()}")
        moves.append(move.uci()); board.push(move)
    out = board.outcome(claim_draw=True)
    result = "1/2-1/2" if out is None else out.result()
    term = "MAX_PLIES" if out is None else out.termination.name
    return {
        "opening_fen": fen,
        "leviathan_white": lev_white,
        "result": result,
        "score_leviathan": v1.score_from_result(result, lev_white),
        "termination": term,
        "plies": len(moves),
        "moves": moves,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P19.2 loss-envelope survival gate")
    ap.add_argument("--engine", required=True); ap.add_argument("--opponent-engine", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--games", type=int, default=100); ap.add_argument("--opponent-threads", type=int, default=6); ap.add_argument("--hash", type=int, default=128); ap.add_argument("--opponent-movetime-ms", type=int, default=500)
    ap.add_argument("--max-plies", type=int, default=240); ap.add_argument("--opening-plies", type=int, default=10); ap.add_argument("--opening-nodes", type=int, default=1500); ap.add_argument("--seed", type=int, default=20260818); ap.add_argument("--sentinel-repeats", type=int, default=5)
    ap.add_argument("--baseline-nodes", type=int, default=120000); ap.add_argument("--root-nodes", type=int, default=40000); ap.add_argument("--reply-nodes", type=int, default=30000); ap.add_argument("--deep-reply-nodes", type=int, default=120000)
    ap.add_argument("--panic-root-nodes", type=int, default=180000); ap.add_argument("--panic-reply-nodes", type=int, default=80000); ap.add_argument("--panic-deep-nodes", type=int, default=300000)
    ap.add_argument("--candidate-mpv", type=int, default=8); ap.add_argument("--dangerous-replies", type=int, default=6)
    ap.add_argument("--loss-delta-pm", type=int, default=10, help="max allowed increase in worst loss probability, per mille")
    ap.add_argument("--max-loss-pm", type=int, default=350, help="catastrophic absolute loss-probability cap, per mille")
    ap.add_argument("--disagreement-pm", type=int, default=80)
    a = ap.parse_args()
    if a.games <= 0 or a.games % 2: raise SystemExit("games must be positive and even")
    identity = {
        "version": "p19-loss-envelope-v3",
        "engine_sha256": v1.sha256_file(Path(a.engine)),
        "stockfish_sha256": v1.sha256_file(Path(a.opponent_engine)),
        "games": a.games, "opponent_threads": a.opponent_threads, "hash": a.hash,
        "opponent_movetime_ms": a.opponent_movetime_ms, "seed": a.seed,
        "sentinel_repeats": a.sentinel_repeats,
        "baseline_nodes": a.baseline_nodes, "root_nodes": a.root_nodes,
        "reply_nodes": a.reply_nodes, "deep_reply_nodes": a.deep_reply_nodes,
        "panic_root_nodes": a.panic_root_nodes, "panic_reply_nodes": a.panic_reply_nodes,
        "panic_deep_nodes": a.panic_deep_nodes, "candidate_mpv": a.candidate_mpv,
        "dangerous_replies": a.dangerous_replies, "loss_delta_pm": a.loss_delta_pm,
        "max_loss_pm": a.max_loss_pm,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    out = Path(a.out_dir) / run_id; out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    jprint({"event": "P19_2_CONFIG", "run_id": run_id, **identity})

    funnel = LossEnvelopeFunnel(a.engine, a.opponent_engine, a.hash, a.baseline_nodes, a.root_nodes,
                                a.reply_nodes, a.deep_reply_nodes, a.panic_root_nodes,
                                a.panic_reply_nodes, a.panic_deep_nodes, a.candidate_mpv,
                                a.dangerous_replies, a.loss_delta_pm, a.max_loss_pm, a.disagreement_pm)
    sf = chess.engine.SimpleEngine.popen_uci(a.opponent_engine, timeout=60.0)
    v1.configure(sf, a.opponent_threads, a.hash)
    decisions = out / "loss-envelope-decisions.jsonl"
    rows = []
    try:
        jprint({"event": "P19_2_SENTINEL_GATE_START", "cases": len(LOSS_SENTINELS), "repeats": a.sentinel_repeats})
        sidx = 0
        for case in LOSS_SENTINELS:
            for rep in range(1, a.sentinel_repeats + 1):
                sidx += 1
                try:
                    r = play_game(funnel, sf, case["fen"], bool(case["leviathan_white"]), f"p19-2-sentinel-{sidx}", a.opponent_movetime_ms, a.max_plies, decisions)
                except UncertifiedPosition as exc:
                    payload = {"event": "P19_2_UNCERTIFIED_SENTINEL", "sentinel": case["name"], "repeat": rep, "certificate": exc.telemetry}
                    (out / "uncertified-sentinel.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                    jprint(payload); return 41
                r.update({"sentinel": case["name"], "repeat": rep}); append_jsonl(out / "sentinels.jsonl", r)
                jprint({"event": "P19_2_SENTINEL_COMPLETE", **r})
                if r["score_leviathan"] == 0.0:
                    jprint({"event": "P19_2_SENTINEL_LOSS", **r}); return 21
        jprint({"event": "P19_2_SENTINEL_GATE_PASSED", "games": sidx, "losses": 0})

        openings = v1.generate_openings(a.opponent_engine, out / "openings.fen", a.games // 2, a.opening_plies, a.seed, a.opening_nodes)
        for game in range(1, a.games + 1):
            fen = openings[(game - 1) // 2]; lev_white = game % 2 == 1
            try:
                r = play_game(funnel, sf, fen, lev_white, f"p19-2-game-{game}", a.opponent_movetime_ms, a.max_plies, decisions)
            except UncertifiedPosition as exc:
                payload = {"event": "P19_2_UNCERTIFIED_FRESH", "game": game, "certificate": exc.telemetry}
                (out / "uncertified-fresh.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                jprint(payload); return 42
            r["game"] = game; append_jsonl(out / "games.jsonl", r); rows.append(r)
            jprint({"event": "P19_2_GAME_COMPLETE", **r, "cumulative": v1.summary(rows)})
            if r["score_leviathan"] == 0.0:
                payload = {"event": "P19_2_ZERO_LOSS_GATE_FAILED", **r, "cumulative": v1.summary(rows)}
                (out / "failure.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                jprint(payload); return 31
        final = {"event": "P19_2_ZERO_LOSS_GATE_PASSED", "summary": v1.summary(rows), "sentinels": len(LOSS_SENTINELS) * a.sentinel_repeats, "losses": 0}
        (out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
        jprint(final); return 0
    finally:
        funnel.close(); v1.safe_quit(sf)


if __name__ == "__main__":
    raise SystemExit(main())

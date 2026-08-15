"""Goal-attractor / reverse tactical proposal specialist for Leviathan.

This does not attempt unsafe general retrograde move generation. Instead it
builds a reverse index from already-solved/proposed tactical lines: each
intermediate board stores a suffix toward the known goal. A bounded forward
search from a new position then tries to meet that index. The result is a
candidate line only; Stockfish or the AND/OR mate prover must verify it.

This realizes the reverse-goal-index idea while preserving chess correctness.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import chess


def canonical_fen(board: chess.Board) -> str:
    """Position identity without halfmove/fullmove counters."""
    parts = board.fen().split()
    return " ".join(parts[:4])


def key_of(board: chess.Board) -> str:
    return hashlib.blake2b(canonical_fen(board).encode(), digest_size=16).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def infer_goal(board: chess.Board) -> str:
    if board.is_checkmate():
        return "mate"
    if board.is_game_over(claim_draw=True):
        return "terminal"
    return "tactical-line-end"


def build(args: argparse.Namespace) -> None:
    records: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(args.lines):
        fen = row.get("fen")
        pv = row.get("pv", [])
        if not fen or not isinstance(pv, list) or not pv:
            continue
        board = chess.Board(fen)
        states: list[tuple[str, str, list[str]]] = []
        for i, uci in enumerate(pv):
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                states = []
                break
            states.append((key_of(board), canonical_fen(board), pv[i:]))
            board.push(move)
        if not states:
            continue
        # Promotion/material targets should be provided explicitly by the source
        # artifact; only terminal board states can be inferred without ambiguity.
        goal = str(row.get("goal", infer_goal(board)))
        source = str(row.get("source", row.get("position_id", "unknown")))
        confidence = int(row.get("confidence", 1000 if row.get("proven") else 500))
        confidence = max(0, min(1000, confidence))
        for key, cfen, suffix in states:
            candidate = {
                "fen": cfen,
                "suffix": suffix,
                "distance": len(suffix),
                "goal": goal,
                "confidence": confidence,
                "source": source,
            }
            old = records.get(key)
            if old is None or (candidate["confidence"], -candidate["distance"]) > (
                old["confidence"], -old["distance"]
            ):
                records[key] = candidate

    payload_obj = {"format": "LVGA1", "entries": records}
    canonical = json.dumps(payload_obj, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    payload_obj["sha256"] = digest
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"format": "LVGA1", "entries": len(records), "sha256": digest}, sort_keys=True))


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    moves = list(board.legal_moves)
    moves.sort(
        key=lambda m: (
            board.gives_check(m),
            bool(m.promotion),
            board.is_capture(m),
            chess.square_rank(m.to_square) if board.turn == chess.WHITE else 7 - chess.square_rank(m.to_square),
        ),
        reverse=True,
    )
    return moves


def query(args: argparse.Namespace) -> None:
    data = json.loads(args.index.read_text(encoding="utf-8"))
    if data.get("format") != "LVGA1" or not isinstance(data.get("entries"), dict):
        raise SystemExit("invalid LVGA1 index")
    entries: dict[str, dict[str, Any]] = data["entries"]
    root = chess.Board(args.fen)

    # BFS gives the shortest forward bridge under the bounded depth. A transposition
    # set turns repeated positions into a DAG instead of re-expanding them.
    queue = collections.deque([(root.copy(stack=False), [])])
    seen = {canonical_fen(root)}
    nodes = 0
    hit: dict[str, Any] | None = None
    bridge: list[str] = []

    while queue and nodes < args.max_nodes:
        board, path = queue.popleft()
        nodes += 1
        item = entries.get(key_of(board))
        if item is not None and canonical_fen(board) == item.get("fen"):
            hit = item
            bridge = path
            break
        if len(path) >= args.max_plies:
            continue
        for move in ordered_moves(board):
            child = board.copy(stack=False)
            child.push(move)
            ident = canonical_fen(child)
            if ident in seen:
                continue
            seen.add(ident)
            queue.append((child, [*path, move.uci()]))

    result = {
        "found": hit is not None,
        "nodes": nodes,
        "bridge": bridge,
        "goal": hit.get("goal") if hit else None,
        "indexed_suffix": hit.get("suffix", []) if hit else [],
        "candidate_line": bridge + (hit.get("suffix", []) if hit else []),
        "confidence": hit.get("confidence", 0) if hit else 0,
        "source": hit.get("source") if hit else None,
        "verification_required": True,
    }
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--lines", type=Path, required=True, help="JSONL with fen + pv; mate_prover/PV-skill output works")
    b.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("query")
    q.add_argument("--index", type=Path, required=True)
    q.add_argument("--fen", required=True)
    q.add_argument("--max-plies", type=int, default=4)
    q.add_argument("--max-nodes", type=int, default=50000)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    else:
        query(args)


if __name__ == "__main__":
    main()

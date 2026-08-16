"""Reproducible paired UCI match harness for Leviathan experiments.

Each opening is played with reversed colors. A and B may point to the same
binary with different UCI option JSON files, eliminating compiler/build noise in
ablation tests. Every nominal match game gets a distinct python-chess ``game``
token so UCI engines receive an explicit ``ucinewgame`` boundary; state intended
to persist within a real game may survive moves, but must never leak into the next
independent test game. For final claims prefer fixed wall-clock move time; fixed
nodes is useful for search-efficiency diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine

HARNESS_VERSION = 2
GAME_BOUNDARY_MODE = "python-chess-game-token/ucinewgame"


def load_options(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("options JSON must be an object")
    return data


def safe_configure(engine: chess.engine.SimpleEngine, options: dict[str, Any]) -> None:
    available = engine.options
    unknown = sorted(set(options) - set(available))
    if unknown:
        raise ValueError(f"engine does not expose options: {unknown}")
    if options:
        engine.configure(options)


def clear_hash(engine: chess.engine.SimpleEngine) -> None:
    if "Clear Hash" in engine.options:
        try:
            engine.configure({"Clear Hash": None})
        except Exception:
            pass


def game_limit(args: argparse.Namespace) -> chess.engine.Limit:
    if args.movetime_ms:
        return chess.engine.Limit(time=args.movetime_ms / 1000.0)
    return chess.engine.Limit(nodes=args.nodes_per_move)


def play_one(
    eng_a: chess.engine.SimpleEngine,
    eng_b: chess.engine.SimpleEngine,
    fen: str,
    a_white: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    board = chess.Board(fen)
    clear_hash(eng_a)
    clear_hash(eng_b)
    moves: list[str] = []
    limit = game_limit(args)

    # A fresh object is unequal by identity to the previous game's token. Passing
    # this same token on every move of this game lets python-chess preserve normal
    # within-game state while sending ucinewgame exactly when a new test game starts.
    game_token = object()

    for _ in range(args.max_plies):
        if board.is_game_over(claim_draw=True):
            break
        a_to_move = board.turn == chess.WHITE if a_white else board.turn == chess.BLACK
        engine = eng_a if a_to_move else eng_b
        result = engine.play(board, limit, game=game_token, ponder=False)
        move = result.move
        if move is None or move not in board.legal_moves:
            raise RuntimeError(f"engine returned illegal/no move at {board.fen()}: {move}")
        moves.append(move.uci())
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result_text = "1/2-1/2"
        termination = "max_plies"
    else:
        result_text = outcome.result()
        termination = outcome.termination.name

    if result_text == "1/2-1/2":
        score_a = 0.5
    elif (result_text == "1-0" and a_white) or (result_text == "0-1" and not a_white):
        score_a = 1.0
    else:
        score_a = 0.0

    return {
        "fen": fen,
        "a_white": a_white,
        "result": result_text,
        "score_a": score_a,
        "termination": termination,
        "plies": len(moves),
        "moves": moves,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--options-a", type=Path)
    ap.add_argument("--options-b", type=Path)
    ap.add_argument("--openings", type=Path, help="one FEN per line; default start position")
    ap.add_argument("--games", type=int, default=20)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--movetime-ms", type=int)
    group.add_argument("--nodes-per-move", type=int)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive")

    fens = [chess.STARTING_FEN]
    if args.openings:
        fens = [x.strip() for x in args.openings.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not fens:
            raise SystemExit("opening file is empty")

    opts_a, opts_b = load_options(args.options_a), load_options(args.options_b)
    a = chess.engine.SimpleEngine.popen_uci(args.engine_a)
    b = chess.engine.SimpleEngine.popen_uci(args.engine_b)
    rows: list[dict[str, Any]] = []
    try:
        safe_configure(a, opts_a)
        safe_configure(b, opts_b)
        for g in range(args.games):
            fen = fens[(g // 2) % len(fens)]
            row = play_one(a, b, fen, a_white=(g % 2 == 0), args=args)
            row["game"] = g + 1
            rows.append(row)
            print(f"game={g + 1}/{args.games} score_a={row['score_a']} result={row['result']}")
    finally:
        a.quit()
        b.quit()

    wins = sum(r["score_a"] == 1.0 for r in rows)
    draws = sum(r["score_a"] == 0.5 for r in rows)
    losses = sum(r["score_a"] == 0.0 for r in rows)
    summary = {
        "games": len(rows),
        "wins_a": wins,
        "draws": draws,
        "losses_a": losses,
        "score_a": sum(r["score_a"] for r in rows) / len(rows),
        "resource": {"movetime_ms": args.movetime_ms, "nodes_per_move": args.nodes_per_move},
        "options_a": opts_a,
        "options_b": opts_b,
        "harness_version": HARNESS_VERSION,
        "game_boundary_mode": GAME_BOUNDARY_MODE,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "games": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

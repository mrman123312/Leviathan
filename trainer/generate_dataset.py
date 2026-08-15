"""Generate policy training examples using a UCI teacher engine.

Input: text file containing one FEN per line.
For every legal quiet move, run a bounded root-move teacher search and write the
same 12 move-local features used by src/leviathan_policy.h plus a utility target.
This is intentionally expensive but clean for the first proof-of-signal dataset.
Later Leviathan phases can replace it with direct search-node instrumentation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import chess
import chess.engine


def centered_file(square: chess.Square) -> int:
    return 2 * chess.square_file(square) - 7


def rank_for(color: chess.Color, square: chess.Square) -> int:
    r = chess.square_rank(square)
    return r if color == chess.WHITE else 7 - r


def move_features(board: chess.Board, move: chess.Move) -> list[int]:
    us = board.turn
    ff, tf = centered_file(move.from_square), centered_file(move.to_square)
    fr, tr = rank_for(us, move.from_square), rank_for(us, move.to_square)
    piece = board.piece_at(move.from_square)
    assert piece is not None
    from_center = 14 - abs(ff) - abs(2 * fr - 7)
    to_center = 14 - abs(tf) - abs(2 * tr - 7)

    board.push(move)
    gives_check = int(board.is_check())
    board.pop()

    enemy = not us
    pawn_attacks = 0
    for sq in board.pieces(chess.PAWN, enemy):
        pawn_attacks |= int(board.attacks(sq).mask)

    from_mask = 1 << move.from_square
    to_mask = 1 << move.to_square
    return [
        ff,
        2 * fr - 7,
        tf,
        2 * tr - 7,
        tf - ff,
        tr - fr,
        piece.piece_type,
        to_center - from_center,
        gives_check * 8,
        int(bool(pawn_attacks & from_mask)) * 8,
        int(bool(pawn_attacks & to_mask)) * 8,
        2 * (tr - fr),
    ]


def score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    pov = score.pov(turn)
    # Treat mate as a very large bounded centipawn score.
    return pov.score(mate_score=100000) or 0


def utility(cp: int) -> float:
    # Bounded target: preserves ordering while reducing domination by mates.
    return 8.0 * math.tanh(cp / 600.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--fens", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--max-positions", type=int, default=0)
    args = ap.parse_args()

    fens = [x.strip() for x in args.fens.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.max_positions:
        fens = fens[: args.max_positions]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    written = 0
    try:
        with args.out.open("w", encoding="utf-8") as out:
            for game_id, fen in enumerate(fens):
                board = chess.Board(fen)
                turn = board.turn
                for move in board.legal_moves:
                    if board.is_capture(move) or move.promotion:
                        continue
                    info = engine.analyse(
                        board,
                        chess.engine.Limit(nodes=args.nodes),
                        root_moves=[move],
                    )
                    cp = score_to_cp(info["score"], turn)
                    row = {
                        "game_id": str(game_id),
                        "fen": fen,
                        "move": move.uci(),
                        "features": move_features(board, move),
                        "teacher_cp": cp,
                        "target": utility(cp),
                    }
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                    written += 1
                print(f"position={game_id + 1}/{len(fens)} rows={written}")
    finally:
        engine.quit()


if __name__ == "__main__":
    main()

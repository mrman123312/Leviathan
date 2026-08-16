#!/usr/bin/env python3
"""CPU-scale Stockfish teacher distillation into a compact Leviathan linear evaluator.

This is deliberately a cheap bootstrap experiment, not an attempt to reproduce NNUE.
It learns symmetric piece-square and a few structural weights from a deterministic
corpus, then evaluates on a held-out split before emitting a C++ header.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import chess
import chess.engine
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PT_INDEX = {pt: i for i, pt in enumerate(PIECE_TYPES)}
PSQT_FEATURES = 6 * 64
# Extra features: material count P/N/B/R/Q, bishop pair, doubled pawns, isolated pawns,
# passed pawns, castling rights, king shield, tempo.
EXTRA_NAMES = [
    "mat_p", "mat_n", "mat_b", "mat_r", "mat_q",
    "bishop_pair", "doubled_pawns", "isolated_pawns", "passed_pawns",
    "castling_rights", "king_shield", "tempo"
]
FEATURE_COUNT = PSQT_FEATURES + len(EXTRA_NAMES)


def canonical_sq(square: int, color: chess.Color) -> int:
    return square if color == chess.WHITE else chess.square_mirror(square)


def pawn_files(board: chess.Board, color: chess.Color) -> list[int]:
    counts = [0] * 8
    for sq in board.pieces(chess.PAWN, color):
        counts[chess.square_file(sq)] += 1
    return counts


def isolated_count(board: chess.Board, color: chess.Color) -> int:
    files = pawn_files(board, color)
    total = 0
    for f, n in enumerate(files):
        if not n:
            continue
        left = files[f - 1] if f > 0 else 0
        right = files[f + 1] if f < 7 else 0
        if left == 0 and right == 0:
            total += n
    return total


def doubled_count(board: chess.Board, color: chess.Color) -> int:
    return sum(max(0, n - 1) for n in pawn_files(board, color))


def passed_count(board: chess.Board, color: chess.Color) -> int:
    enemy = board.pieces(chess.PAWN, not color)
    total = 0
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        passed = True
        for ef in range(max(0, f - 1), min(7, f + 1) + 1):
            for er in range(8):
                esq = chess.square(ef, er)
                if esq not in enemy:
                    continue
                if color == chess.WHITE and er > r:
                    passed = False
                if color == chess.BLACK and er < r:
                    passed = False
                if not passed:
                    break
            if not passed:
                break
        if passed:
            total += 1
    return total


def king_shield(board: chess.Board, color: chess.Color) -> int:
    k = board.king(color)
    if k is None:
        return 0
    kf = chess.square_file(k)
    kr = chess.square_rank(k)
    direction = 1 if color == chess.WHITE else -1
    score = 0
    for df in (-1, 0, 1):
        f = kf + df
        r = kr + direction
        if 0 <= f < 8 and 0 <= r < 8 and board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color):
            score += 1
    return score


def features(board: chess.Board) -> np.ndarray:
    x = np.zeros(FEATURE_COUNT, dtype=np.float64)
    for color, sign in ((chess.WHITE, 1.0), (chess.BLACK, -1.0)):
        for pt in PIECE_TYPES:
            base = PT_INDEX[pt] * 64
            for sq in board.pieces(pt, color):
                x[base + canonical_sq(sq, color)] += sign

    e = PSQT_FEATURES
    for i, pt in enumerate(PIECE_TYPES[:5]):
        x[e + i] = len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
    x[e + 5] = (1 if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2 else 0) - (1 if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2 else 0)
    x[e + 6] = doubled_count(board, chess.WHITE) - doubled_count(board, chess.BLACK)
    x[e + 7] = isolated_count(board, chess.WHITE) - isolated_count(board, chess.BLACK)
    x[e + 8] = passed_count(board, chess.WHITE) - passed_count(board, chess.BLACK)
    white_castle = int(board.has_kingside_castling_rights(chess.WHITE)) + int(board.has_queenside_castling_rights(chess.WHITE))
    black_castle = int(board.has_kingside_castling_rights(chess.BLACK)) + int(board.has_queenside_castling_rights(chess.BLACK))
    x[e + 9] = white_castle - black_castle
    x[e + 10] = king_shield(board, chess.WHITE) - king_shield(board, chess.BLACK)
    x[e + 11] = 1.0 if board.turn == chess.WHITE else -1.0
    return x


def baseline_white_cp(board: chess.Board) -> int:
    values = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:0}
    score = 0
    for sq, piece in board.piece_map().items():
        v = values[piece.piece_type]
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            # Match current C++ baseline exactly; black is not mirrored there.
            center = 6 - (abs(f - 3) + abs(r - 3))
            v += center * 2
        score += v if piece.color == chess.WHITE else -v
    return score


def generate_positions(count: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    positions: list[chess.Board] = []
    seen: set[str] = set()
    while len(positions) < count:
        board = chess.Board()
        target = rng.randint(4, 90)
        for ply in range(target):
            if board.is_game_over(claim_draw=True):
                break
            moves = list(board.legal_moves)
            # Mildly prefer captures/checks 25% of the time so the corpus is not purely quiet.
            tactical = [m for m in moves if board.is_capture(m) or board.gives_check(m)]
            if tactical and rng.random() < 0.25:
                move = rng.choice(tactical)
            else:
                move = rng.choice(moves)
            board.push(move)
            if ply >= 3 and not board.is_game_over(claim_draw=True) and rng.random() < 0.16:
                key = board.board_fen() + (" w" if board.turn else " b") + " " + board.castling_xfen() + " " + (chess.square_name(board.ep_square) if board.ep_square is not None else "-")
                if key not in seen:
                    seen.add(key)
                    positions.append(board.copy(stack=False))
                    if len(positions) >= count:
                        break
    return positions


def teacher_score(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> int | None:
    info = engine.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_SCORE)
    score = info.get("score")
    if score is None:
        return None
    pov = score.pov(chess.WHITE)
    if pov.is_mate():
        mate = pov.mate()
        if mate is None:
            return None
        return 30000 - min(abs(mate), 1000) if mate > 0 else -30000 + min(abs(mate), 1000)
    cp = pov.score()
    return None if cp is None else int(np.clip(cp, -2000, 2000))


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "correlation": float(np.corrcoef(pred, y)[0, 1]) if len(y) > 1 else 0.0,
    }


def emit_header(path: Path, weights: np.ndarray, report: dict) -> None:
    rounded = np.rint(weights).astype(np.int32)
    psqt = rounded[:PSQT_FEATURES]
    extras = rounded[PSQT_FEATURES:]
    lines = [
        "#pragma once",
        "#include <array>",
        "#include <cstdint>",
        "namespace leviathan::distilled_eval {",
        f"inline constexpr int kTeacherDepth = {report['teacher_depth']};",
        f"inline constexpr int kTrainingSamples = {report['training_samples']};",
        f"inline constexpr int kHoldoutSamples = {report['holdout_samples']};",
        "inline constexpr std::array<int16_t, 384> kPsqt = {",
    ]
    for i in range(0, len(psqt), 16):
        lines.append("    " + ", ".join(str(int(v)) for v in psqt[i:i+16]) + ",")
    lines += ["};", "inline constexpr std::array<int16_t, 12> kExtra = {"]
    lines.append("    " + ", ".join(str(int(v)) for v in extras) + ",")
    lines += ["};", "} // namespace leviathan::distilled_eval", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--samples", type=int, default=2400)
    ap.add_argument("--holdout", type=int, default=600)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=8910)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    total = args.samples + args.holdout
    positions = generate_positions(total, args.seed)

    X = np.stack([features(b) for b in positions])
    baseline = np.array([baseline_white_cp(b) for b in positions], dtype=np.float64)
    y_values = []
    valid_rows = []
    engine = chess.engine.SimpleEngine.popen_uci(args.teacher)
    try:
        for i, b in enumerate(positions):
            score = teacher_score(engine, b, args.depth)
            if score is not None:
                valid_rows.append(i)
                y_values.append(score)
            if (i + 1) % 250 == 0:
                print(f"teacher labeled {i+1}/{total}")
    finally:
        engine.quit()

    X = X[valid_rows]
    baseline = baseline[valid_rows]
    y = np.array(y_values, dtype=np.float64)
    if len(y) < total * 0.9:
        raise SystemExit(f"too many unusable labels: {len(y)}/{total}")

    split = min(args.samples, len(y) - args.holdout)
    Xtr, Xho = X[:split], X[split:]
    ytr, yho = y[:split], y[split:]
    bho = baseline[split:]

    # Symmetric features make a zero intercept appropriate. Ridge stabilizes rarely seen squares.
    gram = Xtr.T @ Xtr
    reg = np.eye(FEATURE_COUNT) * args.ridge
    weights = np.linalg.solve(gram + reg, Xtr.T @ ytr)
    pred_float = Xho @ weights
    pred_int = Xho @ np.rint(weights)

    report = {
        "schema_version": 1,
        "seed": args.seed,
        "teacher": args.teacher,
        "teacher_depth": args.depth,
        "training_samples": int(len(ytr)),
        "holdout_samples": int(len(yho)),
        "ridge": args.ridge,
        "baseline": metrics(bho, yho),
        "distilled_float": metrics(pred_float, yho),
        "distilled_integer": metrics(pred_int, yho),
        "accepted": bool(metrics(pred_int, yho)["mae"] < metrics(bho, yho)["mae"]),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    emit_header(out / "distilled_eval_weights.h", weights, report)
    with (out / "holdout.jsonl").open("w", encoding="utf-8") as f:
        for b, truth, old, new in zip(positions[split:split+len(yho)], yho, bho, pred_int):
            f.write(json.dumps({"fen": b.fen(), "teacher_cp": int(truth), "baseline_cp": int(old), "distilled_cp": int(round(new))}) + "\n")
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

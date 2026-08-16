#!/usr/bin/env python3
"""Cheap Stockfish-teacher distillation for Leviathan.

This experiment intentionally does *not* reproduce NNUE. It asks a narrower
question: can a small CPU-only teacher run learn a useful positional correction
on top of Leviathan's material baseline? Candidate weights are accepted only on
an untouched held-out set.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess
import chess.engine
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PT_INDEX = {pt: i for i, pt in enumerate(PIECE_TYPES)}
PSQT_FEATURES = 6 * 64
EXTRA_NAMES = [
    "bishop_pair", "doubled_pawns", "isolated_pawns", "passed_pawns",
    "castling_rights", "king_shield", "tempo"
]
FEATURE_COUNT = PSQT_FEATURES + len(EXTRA_NAMES)
MAX_CORRECTION_CP = 600


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
        if n and (f == 0 or files[f - 1] == 0) and (f == 7 or files[f + 1] == 0):
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
            for esq in enemy:
                if chess.square_file(esq) != ef:
                    continue
                er = chess.square_rank(esq)
                if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                    passed = False
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
    kf, kr = chess.square_file(k), chess.square_rank(k)
    direction = 1 if color == chess.WHITE else -1
    score = 0
    for df in (-1, 0, 1):
        f, r = kf + df, kr + direction
        if 0 <= f < 8 and 0 <= r < 8:
            p = board.piece_at(chess.square(f, r))
            if p == chess.Piece(chess.PAWN, color):
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
    x[e + 0] = (len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2) - (len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2)
    x[e + 1] = doubled_count(board, chess.WHITE) - doubled_count(board, chess.BLACK)
    x[e + 2] = isolated_count(board, chess.WHITE) - isolated_count(board, chess.BLACK)
    x[e + 3] = passed_count(board, chess.WHITE) - passed_count(board, chess.BLACK)
    wc = int(board.has_kingside_castling_rights(chess.WHITE)) + int(board.has_queenside_castling_rights(chess.WHITE))
    bc = int(board.has_kingside_castling_rights(chess.BLACK)) + int(board.has_queenside_castling_rights(chess.BLACK))
    x[e + 4] = wc - bc
    x[e + 5] = king_shield(board, chess.WHITE) - king_shield(board, chess.BLACK)
    x[e + 6] = 1.0 if board.turn == chess.WHITE else -1.0
    return x


def baseline_white_cp(board: chess.Board) -> int:
    values = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:0}
    score = 0
    for sq, piece in board.piece_map().items():
        v = values[piece.piece_type]
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            center = 6 - (abs(f - 3) + abs(r - 3))
            v += center * 2
        score += v if piece.color == chess.WHITE else -v
    return score


def position_key(board: chess.Board) -> str:
    return board.board_fen() + (" w" if board.turn else " b") + " " + board.castling_xfen() + " " + (chess.square_name(board.ep_square) if board.ep_square is not None else "-")


def generate_guided_positions(engine: chess.engine.SimpleEngine, count: int, seed: int, generation_depth: int) -> list[chess.Board]:
    """Generate plausible diversity by sampling among shallow Stockfish MultiPV candidates."""
    rng = random.Random(seed)
    positions: list[chess.Board] = []
    seen: set[str] = set()
    rank_weights = [0.58, 0.25, 0.12, 0.05]

    while len(positions) < count:
        board = chess.Board()
        max_plies = rng.randint(28, 72)
        for ply in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            infos = engine.analyse(board, chess.engine.Limit(depth=generation_depth), multipv=4, info=chess.engine.INFO_PV)
            if isinstance(infos, dict):
                infos = [infos]
            candidates = [info["pv"][0] for info in infos if info.get("pv")]
            if not candidates:
                candidates = list(board.legal_moves)
            n = len(candidates)
            weights = rank_weights[:n]
            move = rng.choices(candidates, weights=weights, k=1)[0]
            board.push(move)

            if ply >= 5 and not board.is_game_over(claim_draw=True):
                # Sample more densely in middlegames than in the first few opening plies.
                chance = 0.48 if ply < 44 else 0.30
                if rng.random() < chance:
                    key = position_key(board)
                    if key not in seen:
                        seen.add(key)
                        positions.append(board.copy(stack=False))
                        if len(positions) >= count:
                            break
    return positions


def teacher_score(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int, max_abs_cp: int) -> int | None:
    info = engine.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_SCORE)
    score = info.get("score")
    if score is None:
        return None
    pov = score.pov(chess.WHITE)
    if pov.is_mate():
        return None
    cp = pov.score()
    if cp is None or abs(cp) > max_abs_cp:
        return None
    return int(cp)


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "correlation": float(np.corrcoef(pred, y)[0, 1]) if len(y) > 1 else 0.0,
    }


def prediction(baseline: np.ndarray, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    correction = np.clip(X @ weights, -MAX_CORRECTION_CP, MAX_CORRECTION_CP)
    return baseline + correction


def emit_header(path: Path, weights: np.ndarray, report: dict) -> None:
    rounded = np.rint(weights).astype(np.int32)
    rounded[:PSQT_FEATURES] = np.clip(rounded[:PSQT_FEATURES], -180, 180)
    rounded[PSQT_FEATURES:] = np.clip(rounded[PSQT_FEATURES:], -240, 240)
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
        f"inline constexpr int kMaxCorrection = {MAX_CORRECTION_CP};",
        "inline constexpr std::array<int16_t, 384> kPsqt = {",
    ]
    for i in range(0, len(psqt), 16):
        lines.append("    " + ", ".join(str(int(v)) for v in psqt[i:i+16]) + ",")
    lines += ["};", f"inline constexpr std::array<int16_t, {len(EXTRA_NAMES)}> kExtra = {{"]
    lines.append("    " + ", ".join(str(int(v)) for v in extras) + ",")
    lines += ["};", "} // namespace leviathan::distilled_eval", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--samples", type=int, default=2400)
    ap.add_argument("--holdout", type=int, default=600)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--generation-depth", type=int, default=3)
    ap.add_argument("--max-abs-cp", type=int, default=1200)
    ap.add_argument("--ridge", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=8910)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    wanted = args.samples + args.holdout

    engine = chess.engine.SimpleEngine.popen_uci(args.teacher)
    try:
        # Generate extra candidates because mate/extreme labels are deliberately discarded.
        candidates = generate_guided_positions(engine, int(wanted * 1.35) + 100, args.seed, args.generation_depth)
        rows: list[tuple[chess.Board, int]] = []
        for i, board in enumerate(candidates):
            score = teacher_score(engine, board, args.depth, args.max_abs_cp)
            if score is not None:
                rows.append((board, score))
                if len(rows) >= wanted:
                    break
            if (i + 1) % 250 == 0:
                print(f"teacher considered {i+1}/{len(candidates)}; accepted {len(rows)}/{wanted}")
    finally:
        engine.quit()

    if len(rows) < wanted:
        raise SystemExit(f"insufficient usable realistic labels: {len(rows)}/{wanted}")

    positions = [r[0] for r in rows]
    y = np.array([r[1] for r in rows], dtype=np.float64)
    X = np.stack([features(b) for b in positions])
    baseline = np.array([baseline_white_cp(b) for b in positions], dtype=np.float64)

    split = args.samples
    Xtr, Xho = X[:split], X[split:]
    ytr, yho = y[:split], y[split:]
    btr, bho = baseline[:split], baseline[split:]

    residual = ytr - btr
    gram = Xtr.T @ Xtr
    weights = np.linalg.solve(gram + np.eye(FEATURE_COUNT) * args.ridge, Xtr.T @ residual)

    # Bound learned terms before evaluation so the Python gate exactly matches the C++ candidate.
    integer_weights = np.rint(weights)
    integer_weights[:PSQT_FEATURES] = np.clip(integer_weights[:PSQT_FEATURES], -180, 180)
    integer_weights[PSQT_FEATURES:] = np.clip(integer_weights[PSQT_FEATURES:], -240, 240)

    pred_float = prediction(bho, Xho, weights)
    pred_integer = prediction(bho, Xho, integer_weights)
    baseline_metrics = metrics(bho, yho)
    float_metrics = metrics(pred_float, yho)
    integer_metrics = metrics(pred_integer, yho)
    improvement = (baseline_metrics["mae"] - integer_metrics["mae"]) / baseline_metrics["mae"]

    report = {
        "schema_version": 2,
        "seed": args.seed,
        "teacher": args.teacher,
        "teacher_depth": args.depth,
        "generation_depth": args.generation_depth,
        "training_samples": len(ytr),
        "holdout_samples": len(yho),
        "max_abs_cp": args.max_abs_cp,
        "ridge": args.ridge,
        "max_correction_cp": MAX_CORRECTION_CP,
        "baseline": baseline_metrics,
        "distilled_float": float_metrics,
        "distilled_integer": integer_metrics,
        "holdout_mae_improvement_fraction": float(improvement),
        "accepted": bool(integer_metrics["mae"] < baseline_metrics["mae"] and improvement >= 0.02),
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    emit_header(out / "distilled_eval_weights.h", integer_weights, report)
    with (out / "holdout.jsonl").open("w", encoding="utf-8") as f:
        for b, truth, old, new in zip(positions[split:], yho, bho, pred_integer):
            f.write(json.dumps({"fen": b.fen(), "teacher_cp": int(truth), "baseline_cp": int(old), "distilled_cp": int(round(new))}) + "\n")

    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shadow census for transferable counterfactual failure certificates.

For a quiet root rival, retain the opponent's best first refutation.  Construct
reachable two-ply near-neighbour positions and test whether that refutation stays
exactly best or remains within a bounded regret.  A small geometric dependency
footprint separates preserving from deliberately violating perturbations.

This tool never prunes or changes engine search.  It measures whether a reusable
representation has enough precision and coverage to justify an engine prototype.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

import chess
import chess.engine


MATE_SCORE = 100000


def clear(engine: chess.engine.SimpleEngine) -> None:
    if "Clear Hash" in engine.options:
        engine.configure({"Clear Hash": None})


def analyse(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    multipv: int = 1,
    root_moves: list[chess.Move] | None = None,
) -> list[dict[str, Any]]:
    result = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=multipv,
        root_moves=root_moves,
    )
    return result if isinstance(result, list) else [result]


def score_cp(info: dict[str, Any], pov: chess.Color) -> int:
    value = info["score"].pov(pov).score(mate_score=MATE_SCORE)
    return int(value if value is not None else 0)


def quiet_nonrights_move(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    return bool(
        piece
        and piece.piece_type not in (chess.PAWN, chess.KING, chess.ROOK)
        and not board.is_capture(move)
        and not board.gives_check(move)
        and not move.promotion
    )


def selected_rival(
    engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int
) -> tuple[chess.Move, int, int] | None:
    clear(engine)
    infos = analyse(engine, board, nodes, min(6, board.legal_moves.count()))
    if len(infos) < 2:
        return None
    best = score_cp(infos[0], board.turn)
    for info in infos[1:]:
        if not info.get("pv"):
            continue
        move = info["pv"][0]
        if board.is_capture(move) or board.gives_check(move) or move.promotion:
            continue
        gap = best - score_cp(info, board.turn)
        if 3 <= gap <= 120:
            return move, gap, score_cp(info, board.turn)
    return None


def first_refutation(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    rival: chess.Move,
    nodes: int,
    witness_plies: int,
) -> tuple[chess.Move, int, list[chess.Move]] | None:
    child = board.copy(stack=False)
    child.push(rival)
    clear(engine)
    info = analyse(engine, child, nodes)[0]
    if not info.get("pv"):
        return None
    witness = info["pv"][:witness_plies]
    return info["pv"][0], score_cp(info, child.turn), witness


def ray_squares(move: chess.Move) -> chess.SquareSet:
    return chess.SquareSet(chess.between(move.from_square, move.to_square))


def dependency_footprint(
    board: chess.Board, rival: chess.Move, witness: list[chess.Move]
) -> tuple[chess.SquareSet, chess.SquareSet, list[chess.Move]]:
    sequence = [rival, *witness]
    core = chess.SquareSet()
    state = board.copy(stack=False)
    states: list[chess.Board] = []
    for move in sequence:
        if move not in state.legal_moves:
            break
        states.append(state.copy(stack=False))
        core.add(move.from_square)
        core.add(move.to_square)
        core |= ray_squares(move)
        for color in (chess.WHITE, chess.BLACK):
            king = state.king(color)
            if king is not None:
                core.add(king)
        state.push(move)
    sequence = sequence[: len(states)]
    footprint = chess.SquareSet(core)
    for state in states:
        for square in core:
            footprint |= state.attackers(chess.WHITE, square)
            footprint |= state.attackers(chess.BLACK, square)
    return core, footprint, sequence


def same_witness_context(
    original: chess.Board,
    neighbour: chess.Board,
    core: chess.SquareSet,
    footprint: chess.SquareSet,
    sequence: list[chess.Move],
) -> bool:
    source = original.copy(stack=False)
    target = neighbour.copy(stack=False)
    for move in sequence:
        if move not in source.legal_moves or move not in target.legal_moves:
            return False
        for square in footprint:
            if source.piece_at(square) != target.piece_at(square):
                return False
        for square in core:
            for color in (chess.WHITE, chess.BLACK):
                if source.attackers(color, square) != target.attackers(color, square):
                    return False
        source.push(move)
        target.push(move)
    return True
    return core, footprint


def preserves_dependency(
    original: chess.Board,
    neighbour: chess.Board,
    core: chess.SquareSet,
    footprint: chess.SquareSet,
    mutation_squares: chess.SquareSet,
    sequence: list[chess.Move],
) -> bool:
    if mutation_squares & footprint:
        return False
    return same_witness_context(original, neighbour, core, footprint, sequence)


def line_remains_legal(
    board: chess.Board, rival: chess.Move, refutation: chess.Move
) -> bool:
    if rival not in board.legal_moves:
        return False
    child = board.copy(stack=False)
    child.push(rival)
    return refutation in child.legal_moves


def neighbour_candidates(
    board: chess.Board,
    rival: chess.Move,
    refutation: chess.Move,
    witness: list[chess.Move],
    per_class: int,
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    core, footprint, sequence = dependency_footprint(board, rival, witness)
    first_moves = [move for move in board.legal_moves if quiet_nonrights_move(board, move)]
    rng.shuffle(first_moves)
    result: dict[str, list[dict[str, Any]]] = {"preserve": [], "violate": []}
    seen: set[str] = set()
    for first in first_moves[:24]:
        intermediate = board.copy(stack=False)
        intermediate.push(first)
        replies = [
            move for move in intermediate.legal_moves if quiet_nonrights_move(intermediate, move)
        ]
        rng.shuffle(replies)
        for reply in replies[:24]:
            neighbour = intermediate.copy(stack=False)
            neighbour.push(reply)
            key = " ".join(neighbour.fen().split()[:4])
            if key in seen or not line_remains_legal(neighbour, rival, refutation):
                continue
            seen.add(key)
            touched = chess.SquareSet(
                [first.from_square, first.to_square, reply.from_square, reply.to_square]
            )
            label = (
                "preserve"
                if preserves_dependency(
                    board, neighbour, core, footprint, touched, sequence
                )
                else "violate"
            )
            if len(result[label]) >= per_class:
                continue
            result[label].append(
                {
                    "fen": neighbour.fen(),
                    "mutation": [first.uci(), reply.uci()],
                }
            )
            if all(len(items) >= per_class for items in result.values()):
                return result
    return result


def evaluate_transfer(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    rival: chess.Move,
    refutation: chess.Move,
    nodes: int,
) -> dict[str, Any]:
    child = board.copy(stack=False)
    child.push(rival)
    clear(engine)
    best_info = analyse(engine, child, nodes)[0]
    best_move = best_info["pv"][0]
    best_score = score_cp(best_info, child.turn)
    clear(engine)
    forced_info = analyse(engine, child, nodes, root_moves=[refutation])[0]
    forced_score = score_cp(forced_info, child.turn)
    regret = max(0, best_score - forced_score)
    return {
        "best_move": best_move.uci(),
        "original_refutation": refutation.uci(),
        "exact": best_move == refutation,
        "best_score_cp": best_score,
        "forced_score_cp": forced_score,
        "regret_cp": regret,
        "within_20cp": regret <= 20,
    }


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    transfers = [
        neighbour["transfer"]
        for row in rows
        for neighbour in row["neighbours"][label]
    ]
    regrets = [item["regret_cp"] for item in transfers]
    return {
        "certificates_with_class": sum(bool(row["neighbours"][label]) for row in rows),
        "neighbours": len(transfers),
        "exact_refutation_rate": sum(item["exact"] for item in transfers) / len(transfers)
        if transfers
        else None,
        "within_20cp_rate": sum(item["within_20cp"] for item in transfers) / len(transfers)
        if transfers
        else None,
        "mean_regret_cp": statistics.mean(regrets) if regrets else None,
        "median_regret_cp": statistics.median(regrets) if regrets else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--source-nodes", type=int, default=80000)
    parser.add_argument("--refutation-nodes", type=int, default=80000)
    parser.add_argument("--transfer-nodes", type=int, default=60000)
    parser.add_argument("--neighbours-per-class", type=int, default=3)
    parser.add_argument("--witness-plies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=444091)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fens = [line.strip() for line in args.positions.read_text(encoding="utf-8").splitlines() if line.strip()]
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 64})
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    attempted = 0
    try:
        for fen in fens:
            if len(rows) >= args.count:
                break
            attempted += 1
            board = chess.Board(fen)
            if board.is_check() or board.ep_square is not None or board.halfmove_clock >= 80:
                continue
            rival_data = selected_rival(engine, board, args.source_nodes)
            if rival_data is None:
                continue
            rival, rival_gap, rival_score = rival_data
            refutation_data = first_refutation(
                engine, board, rival, args.refutation_nodes, args.witness_plies
            )
            if refutation_data is None:
                continue
            refutation, refutation_score, witness = refutation_data
            neighbours = neighbour_candidates(
                board,
                rival,
                refutation,
                witness,
                args.neighbours_per_class,
                rng,
            )
            if not neighbours["preserve"] and not neighbours["violate"]:
                continue
            for items in neighbours.values():
                for item in items:
                    item["transfer"] = evaluate_transfer(
                        engine,
                        chess.Board(item["fen"]),
                        rival,
                        refutation,
                        args.transfer_nodes,
                    )
            core, footprint, sequence = dependency_footprint(board, rival, witness)
            row = {
                "source_fen": board.fen(),
                "rival": rival.uci(),
                "rival_gap_cp": rival_gap,
                "rival_score_cp": rival_score,
                "refutation": refutation.uci(),
                "refutation_score_cp": refutation_score,
                "witness": [move.uci() for move in sequence],
                "core_squares": [chess.square_name(square) for square in core],
                "footprint_squares": [chess.square_name(square) for square in footprint],
                "neighbours": neighbours,
            }
            rows.append(row)
            print(
                len(rows),
                rival.uci(),
                refutation.uci(),
                len(neighbours["preserve"]),
                len(neighbours["violate"]),
                flush=True,
            )
    finally:
        engine.quit()

    payload = {
        "schema": "LV_FAILURE_CERTIFICATE_CENSUS_V1",
        "settings": {
            "requested_certificates": args.count,
            "attempted_source_positions": attempted,
            "source_nodes": args.source_nodes,
            "refutation_nodes": args.refutation_nodes,
            "transfer_nodes_per_search": args.transfer_nodes,
            "neighbours_per_class": args.neighbours_per_class,
            "witness_plies": args.witness_plies,
            "seed": args.seed,
        },
        "certificates": len(rows),
        "mean_footprint_squares": statistics.mean(len(row["footprint_squares"]) for row in rows)
        if rows
        else None,
        "preserve": summarize(rows, "preserve"),
        "violate": summarize(rows, "violate"),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    if len(rows) < max(5, args.count // 2):
        raise SystemExit("insufficient certificate coverage")


if __name__ == "__main__":
    main()

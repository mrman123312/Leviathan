#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import chess
import chess.engine

MATE_SCORE = 100000
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def cfg(engine: chess.engine.SimpleEngine, fundamentals: bool) -> None:
    opts = {}
    for name, value in (("Threads", 1), ("Hash", 64)):
        if name in engine.options:
            opts[name] = value
    if fundamentals:
        for name, value in (
            ("Leviathan Fundamentals", True),
            ("Leviathan Fundamentals Authority", 1),
            ("Leviathan Quiet Overdrive", 0),
        ):
            if name in engine.options:
                opts[name] = value
    if opts:
        engine.configure(opts)


def root_score(info: dict, color: chess.Color) -> int:
    return int(info["score"].pov(color).score(mate_score=MATE_SCORE) or 0)


def load_positions(path: str) -> list[tuple[str, chess.Board]]:
    raw = json.loads(Path(path).read_text())
    out = []
    for i, item in enumerate(raw, 1):
        board = chess.Board()
        for u in item["moves"]:
            board.push_uci(u)
        out.append((item.get("name", f"pos-{i}"), board))
    return out


def board_features(board: chess.Board) -> dict[str, float]:
    piece_count = len(board.piece_map())
    nonpawn_count = sum(
        1 for p in board.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING)
    )
    stm = board.turn
    material = 0
    for piece in board.piece_map().values():
        v = PIECE_VALUE[piece.piece_type]
        material += v if piece.color == stm else -v
    return {
        "game_ply": float(board.ply()),
        "legal_count": float(board.legal_moves.count()),
        "piece_count": float(piece_count),
        "nonpawn_count": float(nonpawn_count),
        "material_stm": float(material),
        "in_check": float(board.is_check()),
    }


def one_info(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    token,
    multipv: int = 1,
):
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=token, multipv=multipv)
    infos = info if isinstance(info, list) else [info]
    if not infos or not infos[0].get("pv"):
        raise RuntimeError("engine returned no PV")
    return infos


def analyse_candidate(engine, board, n1, n2, n3, token):
    i1s = one_info(engine, board, n1, token, multipv=2)
    i1 = i1s[0]
    i2 = one_info(engine, board, n2, token, multipv=1)[0]
    i3 = one_info(engine, board, n3, token, multipv=1)[0]

    m1 = i1["pv"][0]
    m2 = i2["pv"][0]
    m3 = i3["pv"][0]
    s1 = root_score(i1, board.turn)
    s2 = root_score(i2, board.turn)
    s3 = root_score(i3, board.turn)
    if len(i1s) > 1 and i1s[1].get("pv"):
        s1b = root_score(i1s[1], board.turn)
        top2_gap = max(0, s1 - s1b)
    else:
        s1b = s1
        top2_gap = 0

    f = board_features(board)
    f.update(
        {
            "score_n1": float(s1),
            "score_n2": float(s2),
            "score_n3": float(s3),
            "abs_score_n3": float(abs(s3)),
            "score_drift_12": float(s2 - s1),
            "score_drift_23": float(s3 - s2),
            "score_drift_13": float(s3 - s1),
            "abs_drift_12": float(abs(s2 - s1)),
            "abs_drift_23": float(abs(s3 - s2)),
            "top2_gap_n1": float(top2_gap),
            "stable_12": float(m1 == m2),
            "stable_23": float(m2 == m3),
            "stable_13": float(m1 == m3),
            "depth_n1": float(i1.get("depth", 0)),
            "depth_n2": float(i2.get("depth", 0)),
            "depth_n3": float(i3.get("depth", 0)),
            "seldepth_n1": float(i1.get("seldepth", 0)),
            "seldepth_n2": float(i2.get("seldepth", 0)),
            "seldepth_n3": float(i3.get("seldepth", 0)),
            "pv_len_n3": float(len(i3.get("pv", []))),
        }
    )
    return m3, s3, f, {
        "n1": m1.uci(),
        "n2": m2.uci(),
        "n3": m3.uci(),
        "n1_second_score": s1b,
    }


def oracle_grade(select_engine, best_engine, cand_engine, board, move, nodes, token):
    best_info = one_info(select_engine, board, nodes, ("select", token), multipv=1)[0]
    best_move = best_info["pv"][0]
    best_grade = best_engine.analyse(
        board, chess.engine.Limit(nodes=nodes), game=("best-root", token), root_moves=[best_move]
    )
    cand_grade = cand_engine.analyse(
        board, chess.engine.Limit(nodes=nodes), game=("cand-root", token), root_moves=[move]
    )
    best_score = root_score(best_grade, board.turn)
    cand_score = root_score(cand_grade, board.turn)
    return best_move, best_score, cand_score, max(0, best_score - cand_score)


def collect(args) -> int:
    positions = load_positions(args.openings_json)
    cand = chess.engine.SimpleEngine.popen_uci(args.candidate, timeout=30)
    oracle_select = chess.engine.SimpleEngine.popen_uci(args.oracle, timeout=30)
    oracle_best = chess.engine.SimpleEngine.popen_uci(args.oracle, timeout=30)
    oracle_cand = chess.engine.SimpleEngine.popen_uci(args.oracle, timeout=30)
    cfg(cand, args.candidate_fundamentals)
    for e in (oracle_select, oracle_best, oracle_cand):
        cfg(e, False)

    rows = []
    try:
        for i, (name, board) in enumerate(positions, 1):
            token = (args.shard_label, i, board.fen())
            move, shallow_score, features, trajectory = analyse_candidate(
                cand, board, args.nodes1, args.nodes2, args.nodes3, ("candidate", token)
            )
            om, oscore, cscore, regret = oracle_grade(
                oracle_select, oracle_best, oracle_cand, board, move, args.oracle_nodes, token
            )
            row = {
                "schema": "LV_SEARCH_ERROR_ATLAS_ROW_V1",
                "shard": args.shard_label,
                "position": i,
                "name": name,
                "fen": board.fen(),
                "candidate_move": move.uci(),
                "candidate_shallow_score": shallow_score,
                "oracle_move": om.uci(),
                "oracle_score": oscore,
                "candidate_oracle_score": cscore,
                "regret_cp": regret,
                "matches_oracle": move == om,
                "trajectory": trajectory,
                "features": features,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        for e in (cand, oracle_select, oracle_best, oracle_cand):
            e.quit()

    Path(args.output).write_text(
        json.dumps({"schema": "LV_SEARCH_ERROR_ATLAS_ROWS_V1", "rows": rows}, indent=2, sort_keys=True)
        + "\n"
    )
    return 0


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def standardize(train_x, test_x):
    cols = len(train_x[0])
    means = [statistics.fmean(r[j] for r in train_x) for j in range(cols)]
    stds = []
    for j in range(cols):
        vals = [r[j] for r in train_x]
        sd = statistics.pstdev(vals)
        stds.append(sd if sd > 1e-9 else 1.0)

    def tx(rows):
        return [[(r[j] - means[j]) / stds[j] for j in range(cols)] for r in rows]

    return tx(train_x), tx(test_x)


def train_logistic(x, y, l2=0.05, steps=1800, lr=0.08):
    n = len(x)
    p = len(x[0])
    w = [0.0] * (p + 1)
    for step in range(steps):
        g = [0.0] * (p + 1)
        for row, target in zip(x, y):
            z = w[0] + sum(w[j + 1] * row[j] for j in range(p))
            err = sigmoid(z) - target
            g[0] += err
            for j in range(p):
                g[j + 1] += err * row[j]
        g[0] /= n
        for j in range(p):
            g[j + 1] = g[j + 1] / n + l2 * w[j + 1]
        eta = lr / math.sqrt(1.0 + step / 250.0)
        for j in range(p + 1):
            w[j] -= eta * g[j]
    return w


def predict(w, rows):
    return [
        sigmoid(w[0] + sum(w[j + 1] * row[j] for j in range(len(row)))) for row in rows
    ]


def auc_score(y, scores):
    pos = [s for s, t in zip(scores, y) if t == 1]
    neg = [s for s, t in zip(scores, y) if t == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps in pos:
        for ns in neg:
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return wins / (len(pos) * len(neg))


def top_fraction_metrics(rows, scores, label_threshold, frac=0.25):
    n = len(rows)
    k = max(1, round(n * frac))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top = order[:k]
    rest = order[k:]
    labels = [int(r["regret_cp"] >= label_threshold) for r in rows]
    positives = sum(labels)
    caught = sum(labels[i] for i in top)
    return {
        "fraction": frac,
        "top_n": k,
        "positive_count": positives,
        "positive_capture_rate": caught / positives if positives else None,
        "top_positive_rate": caught / k,
        "base_positive_rate": positives / n,
        "top_mean_regret_cp": statistics.fmean(rows[i]["regret_cp"] for i in top),
        "rest_mean_regret_cp": statistics.fmean(rows[i]["regret_cp"] for i in rest) if rest else None,
    }


def fit(args) -> int:
    rows = []
    for path in args.inputs:
        obj = json.loads(Path(path).read_text())
        rows.extend(obj["rows"])
    if len(rows) < 20:
        raise SystemExit("need at least 20 rows")

    feature_names = sorted(rows[0]["features"].keys())
    x = [[float(r["features"][k]) for k in feature_names] for r in rows]
    threshold = args.regret_threshold
    y = [int(r["regret_cp"] >= threshold) for r in rows]
    folds = max(2, min(args.folds, len(rows)))
    oof = [0.0] * len(rows)
    fold_records = []
    for fold in range(folds):
        tr = [i for i in range(len(rows)) if i % folds != fold]
        te = [i for i in range(len(rows)) if i % folds == fold]
        xtr = [x[i] for i in tr]
        xte = [x[i] for i in te]
        sxtr, sxte = standardize(xtr, xte)
        w = train_logistic(sxtr, [y[i] for i in tr])
        ps = predict(w, sxte)
        for i, p in zip(te, ps):
            oof[i] = p
        fold_records.append(
            {
                "fold": fold,
                "train_n": len(tr),
                "test_n": len(te),
                "test_positives": sum(y[i] for i in te),
            }
        )

    auc = auc_score(y, oof)
    instability = [1.0 - r["features"]["stable_13"] for r in rows]
    drift = [r["features"]["abs_drift_23"] for r in rows]
    inv_gap = [-r["features"]["top2_gap_n1"] for r in rows]
    metrics = {
        "schema": "LV_SEARCH_ERROR_ATLAS_MODEL_V1",
        "positions": len(rows),
        "regret_threshold_cp": threshold,
        "positive_count": sum(y),
        "positive_rate": sum(y) / len(y),
        "feature_names": feature_names,
        "folds": fold_records,
        "oof_auc_logistic": auc,
        "baseline_auc_move_instability": auc_score(y, instability),
        "baseline_auc_recent_score_drift": auc_score(y, drift),
        "baseline_auc_inverse_top2_gap": auc_score(y, inv_gap),
        "risk_concentration": top_fraction_metrics(rows, oof, threshold, 0.25),
        "all_regret": {
            "mean": statistics.fmean(r["regret_cp"] for r in rows),
            "median": statistics.median(r["regret_cp"] for r in rows),
            "max": max(r["regret_cp"] for r in rows),
        },
    }
    enriched = []
    for r, p in zip(rows, oof):
        z = dict(r)
        z["oof_risk"] = p
        enriched.append(z)
    metrics["rows"] = enriched
    Path(args.output).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print("SUMMARY", json.dumps({k: v for k, v in metrics.items() if k != "rows"}, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--candidate", required=True)
    c.add_argument("--oracle", required=True)
    c.add_argument("--openings-json", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--shard-label", required=True)
    c.add_argument("--candidate-fundamentals", action="store_true")
    c.add_argument("--nodes1", type=int, default=12000)
    c.add_argument("--nodes2", type=int, default=40000)
    c.add_argument("--nodes3", type=int, default=100000)
    c.add_argument("--oracle-nodes", type=int, default=700000)
    c.set_defaults(func=collect)

    f = sub.add_parser("fit")
    f.add_argument("--inputs", nargs="+", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--regret-threshold", type=int, default=15)
    f.add_argument("--folds", type=int, default=5)
    f.set_defaults(func=fit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

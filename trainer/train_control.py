"""Train/export Leviathan MetaSearch or selective-search risk ensembles.

The runtime formats are deliberately tiny linear ensembles. They are cheap
enough to query inside search and expose disagreement between independently
trained heads as epistemic uncertainty.

Meta input: generate_metasearch_dataset.py JSONL. Risk input: optional LMR trace
JSONL emitted by an instrumented Leviathan build; each risk row must contain a
12-element `features` vector plus `regret`/`dangerous`/`need_full_search` label.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

META_FEATURES = 8
RISK_FEATURES = 12


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def derive_meta(rows: list[dict]) -> tuple[list[list[float]], list[float]]:
    by_pos: dict[str, list[dict]] = {}
    for row in rows:
        by_pos.setdefault(str(row.get("position_id", "")), []).append(row)

    xs: list[list[float]] = []
    ys: list[float] = []
    for group in by_pos.values():
        group.sort(key=lambda r: int(r["budget_nodes"]))
        last_move = None
        last_score = None
        last_change_depth = 0
        changes = 0
        for row in group:
            low = row["low"]
            move = low.get("best_move")
            depth = int(low.get("depth", 0))
            score = int(low.get("score_cp", 0))
            if last_move is not None and move != last_move:
                changes += 1
                last_change_depth = depth
            stable_depth = max(0, depth - last_change_depth)
            delta = 0 if last_score is None else abs(score - last_score)
            # Exact root effort is available in live Stockfish but not standard
            # UCI MultiPV output. Use zero here so the initial model cannot learn
            # a spurious proxy; later trace datasets can populate it exactly.
            nodes_effort = 0
            try:
                import chess
                root_count = chess.Board(row["fen"]).legal_moves.count()
            except Exception:
                root_count = 0
            decisive = 1 if abs(score) >= 30000 else 0
            feat = [
                clamp(depth, 0, 128),
                clamp(stable_depth, 0, 64),
                clamp(delta // 8, 0, 256),
                clamp(changes * 16, 0, 256),
                clamp(nodes_effort // 1000, 0, 128),
                clamp(root_count, 0, 64),
                clamp(abs(score) // 32, 0, 256),
                32 if decisive else 0,
            ]
            xs.append([float(v) for v in feat])
            ys.append(500.0 if bool(row.get("need_more_search")) else -500.0)
            last_move, last_score = move, score
    return xs, ys


def derive_risk(rows: list[dict]) -> tuple[list[list[float]], list[float]]:
    xs, ys = [], []
    for row in rows:
        feat = row.get("features")
        if not isinstance(feat, list) or len(feat) != RISK_FEATURES:
            continue
        label = bool(row.get("regret", row.get("dangerous", row.get("need_full_search", False))))
        xs.append([float(clamp(int(v), -4096, 4096)) for v in feat])
        ys.append(500.0 if label else -500.0)
    return xs, ys


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_head(x: torch.Tensor, y: torch.Tensor, seed: int, epochs: int, lr: float) -> torch.nn.Linear:
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    idx = torch.randint(0, n, (n,), generator=g)
    xb, yb = x[idx], y[idx]
    model = torch.nn.Linear(x.shape[1], 1)
    torch.manual_seed(seed)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        pred = model(xb).squeeze(1)
        loss = torch.mean((pred - yb) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        opt.step()
    return model


def export(models: list[torch.nn.Linear], path: Path, magic: str, scale: int) -> None:
    feature_count = models[0].in_features
    lines = [f"{magic} {feature_count} {len(models)} {scale}"]
    for model in models:
        w = model.weight.detach().cpu().view(-1).tolist()
        b = float(model.bias.detach().cpu().item())
        qw = [clamp(round(v * scale), -32768, 32767) for v in w]
        qb = clamp(round(b * scale), -(2**31), 2**31 - 1)
        lines.append(" ".join(str(int(v)) for v in qw) + f" {int(qb)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["meta", "risk"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--scale", type=int, default=256)
    ap.add_argument("--seed", type=int, default=8910)
    args = ap.parse_args()
    if not 1 <= args.heads <= 4:
        raise SystemExit("--heads must be 1..4")
    if args.scale < 1 or args.scale > 1_000_000:
        raise SystemExit("invalid --scale")

    rows = load_rows(args.data)
    xs, ys = derive_meta(rows) if args.task == "meta" else derive_risk(rows)
    if len(xs) < 8:
        raise SystemExit(f"not enough usable rows: {len(xs)}")

    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    models = [train_head(x, y, args.seed + i * 997, args.epochs, args.lr) for i in range(args.heads)]
    magic = "LVTM1" if args.task == "meta" else "LVTR1"
    export(models, args.out, magic, args.scale)

    with torch.no_grad():
        pred = torch.stack([m(x).squeeze(1) for m in models]).mean(0)
        acc = ((pred >= 0) == (y >= 0)).float().mean().item()
        mse = torch.mean((pred - y) ** 2).item()
    print(json.dumps({"rows": len(xs), "heads": args.heads, "accuracy": acc, "mse": mse, "out": str(args.out)}))


if __name__ == "__main__":
    main()

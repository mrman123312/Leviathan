"""Train Leviathan's tiny move-policy network from JSONL teacher data.

Each input row must contain:
  {"features": [12 integers], "target": float, "game_id": "..."}

`target` is a search-utility logit/score where larger means the move deserved
earlier search. Dataset generation tools can derive it from teacher MultiPV
scores or later from direct cutoff instrumentation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import FEATURE_COUNT, LeviathanPolicy


class JsonlDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows: list[tuple[list[float], float]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                features = row["features"]
                if len(features) != FEATURE_COUNT:
                    raise ValueError(f"{path}:{line_no}: expected {FEATURE_COUNT} features")
                self.rows.append(([float(v) for v in features], float(row["target"])))
        if not self.rows:
            raise ValueError(f"empty dataset: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        x, y = self.rows[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    loss_fn = nn.SmoothL1Loss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += float(loss_fn(model(x), y))
            count += y.numel()
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("networks/policy.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=8910)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = JsonlDataset(args.train)
    valid_ds = JsonlDataset(args.valid)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LeviathanPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    best = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss) * y.numel()
            seen += y.numel()

        valid = evaluate(model, valid_loader, device)
        train_loss = running / max(seen, 1)
        print(f"epoch={epoch} train={train_loss:.6f} valid={valid:.6f}")
        if valid < best:
            best = valid
            torch.save({"state_dict": model.state_dict(), "valid_loss": valid}, args.out)

    print(f"best_valid={best:.6f} checkpoint={args.out}")


if __name__ == "__main__":
    main()

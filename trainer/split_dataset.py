"""Split Leviathan JSONL data by game_id so related positions never leak."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def bucket(game_id: str, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--seed", default="leviathan-8910")
    args = ap.parse_args()

    for p in (args.train, args.valid, args.test):
        p.parent.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "valid": 0, "test": 0}
    with args.input.open("r", encoding="utf-8") as src, \
         args.train.open("w", encoding="utf-8") as train, \
         args.valid.open("w", encoding="utf-8") as valid, \
         args.test.open("w", encoding="utf-8") as test:
        outs = {"train": train, "valid": valid, "test": test}
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            b = bucket(str(row["game_id"]), args.seed)
            split = "train" if b < 80 else "valid" if b < 90 else "test"
            outs[split].write(line)
            counts[split] += 1

    print(counts)


if __name__ == "__main__":
    main()

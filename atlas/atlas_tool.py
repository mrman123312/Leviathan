"""Build and verify Project Leviathan Chess Atlas artifacts.

Input rows are engine-native JSONL and must contain `position_key` (or `key`)
and `move_raw`. This avoids pretending Python's Zobrist/move representation is
identical to Stockfish's. Optional fields: bonus, confidence, kind, source.

The runtime export is LVTA1. A sidecar manifest is SHA-256 content addressed so
Atlas knowledge remains reproducible across Leviathan generations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(paths: list[Path]) -> list[dict]:
    out: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def normalize(row: dict) -> tuple[int, int, int, int, str, str]:
    key = int(row.get("position_key", row.get("key")))
    raw = int(row["move_raw"])
    bonus = int(row.get("bonus", 2048))
    confidence = int(row.get("confidence", 500))
    kind = str(row.get("kind", "episode"))
    source = str(row.get("source", "unknown"))
    if not 0 <= key < 2**64:
        raise ValueError("position key out of range")
    if not 0 <= raw <= 65535:
        raise ValueError("move_raw out of range")
    if not -32768 <= bonus <= 32768:
        raise ValueError("bonus out of range")
    if not 0 <= confidence <= 1000:
        raise ValueError("confidence out of range")
    if kind not in {"episode", "skill", "exact"}:
        raise ValueError(f"invalid kind: {kind}")
    return key, raw, bonus, confidence, kind, source


def build(inputs: list[Path], out: Path) -> None:
    # Deduplicate by engine-native position/move/kind. Keep the strongest
    # confidence and, on ties, the larger absolute ordering evidence.
    best: dict[tuple[int, int, str], tuple[int, int, int, int, str, str]] = {}
    for row in load_jsonl(inputs):
        item = normalize(row)
        key = (item[0], item[1], item[4])
        old = best.get(key)
        if old is None or (item[3], abs(item[2])) > (old[3], abs(old[2])):
            best[key] = item

    lines = ["LVTA1"]
    provenance: dict[str, int] = {}
    for item in sorted(best.values(), key=lambda x: (x[0], x[1], x[4])):
        key, raw, bonus, confidence, kind, source = item
        lines.append(f"{key} {raw} {bonus} {confidence} {kind}")
        provenance[source] = provenance.get(source, 0) + 1

    payload = ("\n".join(lines) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    manifest = {
        "format": "LVTA1",
        "sha256": digest,
        "entries": len(lines) - 1,
        "sources": provenance,
        "inputs": [str(p) for p in inputs],
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


def verify(path: Path) -> None:
    payload = path.read_bytes()
    lines = payload.decode().splitlines()
    if not lines or lines[0] != "LVTA1":
        raise SystemExit("invalid LVTA1 magic")
    for i, line in enumerate(lines[1:], 2):
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"line {i}: expected 5 fields")
        normalize({
            "position_key": int(parts[0]),
            "move_raw": int(parts[1]),
            "bonus": int(parts[2]),
            "confidence": int(parts[3]),
            "kind": parts[4],
        })
    digest = hashlib.sha256(payload).hexdigest()
    print(json.dumps({"entries": max(0, len(lines) - 1), "sha256": digest}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--input", type=Path, action="append", required=True)
    b.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("path", type=Path)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.input, args.out)
    else:
        verify(args.path)


if __name__ == "__main__":
    main()

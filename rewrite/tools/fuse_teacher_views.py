#!/usr/bin/env python3
"""Fuse normalized Lc0-native and Stockfish-teacher JSONL without double-counting samples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path, label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        rec = json.loads(raw)
        key = rec.get("position_id") or rec.get("fen")
        if not key:
            raise SystemExit(f"{label}:{lineno}: record needs position_id or fen")
        key = str(key)
        if key in out:
            raise SystemExit(f"{label}:{lineno}: duplicate position key {key}")
        out[key] = rec
    return out


def canonical_id(key: str, lc0: dict[str, Any] | None, sf: dict[str, Any] | None) -> str:
    for rec in (lc0, sf):
        if rec and rec.get("position_id"):
            return str(rec["position_id"])
    return "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()


def extract_split_group(rec: dict[str, Any] | None) -> str | None:
    if not rec:
        return None
    return rec.get("source_split_group") or rec.get("game_id") or rec.get("source_shard")


def stockfish_view(rec: dict[str, Any]) -> dict[str, Any]:
    # Accept either one normalized teacher record or label_corpus.py output.
    if "teachers" in rec:
        teachers = rec["teachers"]
        sf_names = [name for name in teachers if "stockfish" in name.lower()]
        if len(sf_names) != 1:
            raise SystemExit("Stockfish input with teachers must contain exactly one Stockfish-named teacher")
        name = sf_names[0]
        return {"teacher_name": name, **teachers[name]}
    return rec.get("stockfish_teacher", rec)


def lc0_view(rec: dict[str, Any]) -> dict[str, Any]:
    return rec.get("lc0_native", rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc0", required=True, help="normalized Lc0-native JSONL")
    ap.add_argument("--stockfish", required=True, help="Stockfish teacher-label JSONL")
    ap.add_argument("--output", required=True)
    ap.add_argument("--require-complete-overlap", action="store_true")
    args = ap.parse_args()

    lc0 = load_jsonl(Path(args.lc0), "lc0")
    sf = load_jsonl(Path(args.stockfish), "stockfish")
    if args.require_complete_overlap and set(lc0) != set(sf):
        missing_sf = sorted(set(lc0) - set(sf))[:10]
        missing_lc0 = sorted(set(sf) - set(lc0))[:10]
        raise SystemExit(f"view coverage mismatch; missing Stockfish={missing_sf}, missing Lc0={missing_lc0}")

    keys = sorted(set(lc0) | set(sf))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_sample_count = 0
    dual_view_count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for key in keys:
            lrec = lc0.get(key)
            srec = sf.get(key)
            raw_sample_count += 1  # One source position regardless of number of views.
            if lrec and srec:
                dual_view_count += 1

            split_l = extract_split_group(lrec)
            split_s = extract_split_group(srec)
            if split_l and split_s and split_l != split_s:
                raise SystemExit(f"split-group ancestry mismatch for {key}: {split_l} vs {split_s}")
            split_group = split_l or split_s
            if lrec and not split_group:
                raise SystemExit(f"Lc0 source record lacks source_split_group/game_id/source_shard: {key}")

            fen = (lrec or {}).get("fen") or (srec or {}).get("fen")
            record = {
                "schema_version": 2,
                "position_id": canonical_id(key, lrec, srec),
                "source_split_group": split_group,
                "position": {"fen": fen} if fen else {},
                "source": {
                    "raw_sample_count": 1,
                    "lc0_present": lrec is not None,
                    "stockfish_view_present": srec is not None,
                    "lineage_note": "Stockfish annotation is a derived teacher view and does not create another independent raw sample."
                },
                "views": {
                    "lc0_native": lc0_view(lrec) if lrec else None,
                    "stockfish_teacher": stockfish_view(srec) if srec else None,
                },
                "derived": {
                    "dual_view": bool(lrec and srec),
                    "teacher_bestmove_disagreement": None,
                    "calibrated_value_disagreement": None,
                    "frontier_priority": None,
                },
            }
            out.write(json.dumps(record, sort_keys=True) + "\n")

    print(json.dumps({
        "unique_source_positions": raw_sample_count,
        "lc0_records": len(lc0),
        "stockfish_records": len(sf),
        "dual_view_records": dual_view_count,
        "output": str(out_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

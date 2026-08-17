#!/usr/bin/env python3
"""Convert Leviathan isolated-oracle regret artifacts into reusable hard examples.

This tool does not run engines. It harvests information already paid for by prior
oracle experiments so those positions can feed later evaluator/policy curricula.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(row: dict) -> str:
    b = int(row.get("baseline_regret_cp", 0))
    c = int(row.get("candidate_regret_cp", 0))
    bm = row.get("baseline_move")
    cm = row.get("candidate_move")
    om = row.get("oracle_move")
    if c < b:
        return "candidate_rescue"
    if c > b:
        return "candidate_harm"
    if bm != cm:
        return "changed_tie"
    if bm == om:
        return "shared_oracle_move"
    return "shared_nonoracle_tie"


def convert(path: Path, row: dict, schema: str, choice_nodes: int, oracle_nodes: int) -> dict:
    b = int(row.get("baseline_regret_cp", 0))
    c = int(row.get("candidate_regret_cp", 0))
    gap = b - c
    changed = row.get("baseline_move") != row.get("candidate_move")
    # Priority emphasizes actual root-choice mistakes and large intervention effects.
    priority = max(b, c) + abs(gap) + (8 if changed else 0)
    return {
        "schema": "LV_HARD_EXAMPLE_V1",
        "source_artifact": str(path),
        "source_schema": schema,
        "position": row.get("position"),
        "name": row.get("name"),
        "fen": row["fen"],
        "category": classify(row),
        "root_move_changed": changed,
        "baseline_move": row.get("baseline_move"),
        "candidate_move": row.get("candidate_move"),
        "oracle_move": row.get("oracle_move"),
        "baseline_regret_cp": b,
        "candidate_regret_cp": c,
        "regret_delta_candidate_minus_baseline_cp": c - b,
        "oracle_score_cp": row.get("oracle_score"),
        "baseline_score_cp": row.get("baseline_score"),
        "candidate_score_cp": row.get("candidate_score"),
        "choice_nodes": choice_nodes,
        "oracle_nodes_per_search": oracle_nodes,
        "priority": priority,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--min-priority", type=int, default=1)
    ap.add_argument("--changed-only", action="store_true")
    args = ap.parse_args()

    out: list[dict] = []
    sources: list[dict] = []
    for path in args.inputs:
        doc = json.loads(path.read_text())
        schema = doc.get("schema", "UNKNOWN")
        if not isinstance(doc.get("rows"), list):
            raise SystemExit(f"{path}: missing rows list")
        choice_nodes = int(doc.get("choice_nodes", 0) or 0)
        oracle_nodes = int(doc.get("oracle_nodes_per_search", 0) or 0)
        sources.append({"path": str(path), "sha256": sha256(path), "schema": schema})
        for row in doc["rows"]:
            ex = convert(path, row, schema, choice_nodes, oracle_nodes)
            if args.changed_only and not ex["root_move_changed"]:
                continue
            if ex["priority"] < args.min_priority:
                continue
            out.append(ex)

    # Stable, deterministic order: hardest first, then FEN as tie-breaker.
    out.sort(key=lambda x: (-x["priority"], x["fen"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in out:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "schema": "LV_HARD_EXAMPLE_MANIFEST_V1",
        "sources": sources,
        "records": len(out),
        "changed_only": args.changed_only,
        "min_priority": args.min_priority,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "category_counts": {k: sum(x["category"] == k for x in out) for k in sorted({x["category"] for x in out})},
        "rule": "Derived curriculum only; source oracle evidence retains its original scope and does not become a strength claim by being mined."
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

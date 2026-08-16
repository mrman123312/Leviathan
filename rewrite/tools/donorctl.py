#!/usr/bin/env python3
"""Small CLI for the Leviathan donor ecology.

Examples:
  python3 rewrite/tools/donorctl.py list
  python3 rewrite/tools/donorctl.py show caissa
  python3 rewrite/tools/donorctl.py scaffold caissa src/PackedNeuralNetwork.cpp deadbeef evaluation
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "donors" / "DONOR_REGISTRY.json"
IMPORTS = ROOT / "imports"


def registry():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = data.get("engines", []) + data.get("assets_and_infrastructure", [])
    return data, {x["id"]: x for x in entries}


def policy(entry):
    return entry.get("direct_code_policy", entry.get("direct_use_policy", "blocked"))


def cmd_list(_args):
    _data, entries = registry()
    for donor_id in sorted(entries):
        e = entries[donor_id]
        repo = e.get("repo", e.get("source", "-"))
        print(f"{donor_id:24} {e['license']:20} {policy(e):24} {repo}")


def cmd_show(args):
    _data, entries = registry()
    if args.donor_id not in entries:
        raise SystemExit(f"unknown donor: {args.donor_id}")
    print(json.dumps(entries[args.donor_id], indent=2))


def cmd_scaffold(args):
    _data, entries = registry()
    if args.donor_id not in entries:
        raise SystemExit(f"unknown donor: {args.donor_id}")
    e = entries[args.donor_id]
    p = policy(e)
    if p in {"reference_only", "blocked"}:
        raise SystemExit(f"{args.donor_id} is {p}; direct source import is forbidden by current policy")

    name = pathlib.Path(args.source_path).name
    donor_dir = IMPORTS / args.donor_id
    donor_dir.mkdir(parents=True, exist_ok=True)
    sidecar = donor_dir / f"{name}.provenance.json"
    if sidecar.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {sidecar}; use --force if intentional")

    record = {
        "donor_id": args.donor_id,
        "source_repo": e.get("repo", e.get("source", "")),
        "source_revision": args.source_revision,
        "source_path": args.source_path,
        "license": e["license"],
        "reuse_mode": args.reuse_mode,
        "leviathan_owner": args.owner,
        "import_reason": args.reason or "TODO: state unique capability",
        "tests_required": ["unit", "A/B", "fixed-node"],
        "version_gate_verified": p != "allowed_version_gated",
        "changes": "TODO",
        "attribution": "TODO: preserve upstream copyright/license notice"
    }
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(sidecar)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show")
    p.add_argument("donor_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("scaffold")
    p.add_argument("donor_id")
    p.add_argument("source_path")
    p.add_argument("source_revision", help="immutable commit SHA or tag")
    p.add_argument("owner", help="Leviathan subsystem, e.g. evaluation/search/tt")
    p.add_argument("--reuse-mode", choices=["copied", "adapted", "reimplemented", "model", "dataset", "tool"], default="adapted")
    p.add_argument("--reason")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_scaffold)

    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

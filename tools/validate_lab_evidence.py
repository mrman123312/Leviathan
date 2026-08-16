#!/usr/bin/env python3
"""Fail closed on common Project Leviathan laboratory provenance mistakes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
LEDGER_ID = re.compile(r"^\|\s*(W\d{3})\s*\|", re.MULTILINE)
USES = re.compile(r"^\s*-?\s*uses:\s*[^@\s]+@([^\s#]+)", re.MULTILINE)


def run_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        match = re.match(r"^(\s*)run:\s*\|\s*$", lines[index])
        if not match:
            index += 1
            continue
        indent = len(match.group(1))
        block = []
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            block.append(line)
            index += 1
        blocks.append("\n".join(block))
    return blocks


def validate_workflow(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors = []
    for token in (
        "permissions:",
        "timeout-minutes:",
        "retention-days:",
        "if-no-files-found: error",
    ):
        if token not in text:
            errors.append(f"{path}: missing {token}")
    for ref in USES.findall(text):
        if not FULL_SHA.fullmatch(ref):
            errors.append(f"{path}: action ref is not a full SHA: {ref}")
    for number, block in enumerate(run_blocks(text), start=1):
        if "| tee " in block and "set -euo pipefail" not in block:
            errors.append(f"{path}: run block {number} pipes to tee without pipefail")
    if "actions/upload-artifact@" in text and "if: always()" not in text:
        errors.append(f"{path}: evidence artifact is not retained on failure")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    workflows = sorted((root / ".github/workflows").glob("lab-*.yml"))
    manifests = sorted((root / "experiments/manifests").glob("*.json"))
    ledger_text = (root / "experiments/WORK_LAB_LEDGER.md").read_text(encoding="utf-8")
    ledger_ids = set(LEDGER_ID.findall(ledger_text))
    errors = []
    manifest_ids: dict[str, str] = {}
    for path in workflows:
        errors.extend(validate_workflow(path))
    for path in manifests:
        value = json.loads(path.read_text(encoding="utf-8"))
        experiment_id = value.get("id")
        if not isinstance(experiment_id, str) or not re.fullmatch(r"W\d{3}", experiment_id):
            errors.append(f"{path}: missing valid Wnnn id")
            continue
        if experiment_id in manifest_ids:
            errors.append(
                f"{path}: duplicate {experiment_id}, first in {manifest_ids[experiment_id]}"
            )
        manifest_ids[experiment_id] = str(path)
        if experiment_id not in ledger_ids:
            errors.append(f"{path}: {experiment_id} missing from permanent ledger")
    payload = {
        "schema": "LV_LAB_VALIDATION_V1",
        "workflows_checked": len(workflows),
        "manifests_checked": len(manifests),
        "ledger_entries": len(ledger_ids),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

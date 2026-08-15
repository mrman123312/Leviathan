"""Mechanical lineage state for Project Leviathan.

The lineage state remembers candidate ancestry and lifecycle independently of
Git history. It never declares playing strength from architecture alone. A
candidate may become ACTIVE only after explicitly supplied promotion evidence
passes the configured gates. Rollback is a first-class operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"format": "LVLINE1", "active": None, "generation": 0, "candidates": {}, "history": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "LVLINE1":
        raise ValueError("invalid lineage format")
    return data


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(args: argparse.Namespace) -> None:
    data = load(args.state)
    candidate_id = args.id or digest_file(args.artifact)[:16]
    if candidate_id in data["candidates"]:
        raise SystemExit(f"candidate already exists: {candidate_id}")
    parent = args.parent if args.parent != "ACTIVE" else data.get("active")
    entry = {
        "id": candidate_id,
        "parent": parent,
        "artifact": str(args.artifact),
        "artifact_sha256": digest_file(args.artifact),
        "status": "PROVISIONAL",
        "created_at": now(),
        "evidence": [],
    }
    data["candidates"][candidate_id] = entry
    data["history"].append({"at": now(), "event": "REGISTER", "candidate": candidate_id, "parent": parent})
    save(args.state, data)
    print(json.dumps(entry, sort_keys=True))


def base_gates_pass(e: dict[str, Any]) -> bool:
    return (
        bool(e.get("compile_ok", False))
        and bool(e.get("parent_signature_ok", False))
        and float(e.get("nps_delta_pct", 0.0)) >= float(e.get("min_nps_delta_pct", -5.0))
    )


def evidence_passes(e: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not bool(e.get("compile_ok", False)):
        failures.append("compile")
    if not bool(e.get("parent_signature_ok", False)):
        failures.append("parent_signature")
    if float(e.get("nps_delta_pct", 0.0)) < float(e.get("min_nps_delta_pct", -5.0)):
        failures.append("nps")
    # Final promotion requires the explicit high-quality gates. Screening alone
    # can advance a candidate to SCREENED but never ACTIVE.
    for gate in ("stc_pass", "ltc_pass", "multi_hardware_pass"):
        if not bool(e.get(gate, False)):
            failures.append(gate)
    if float(e.get("elo", 0.0)) <= 0.0:
        failures.append("positive_elo")
    return not failures, failures


def screen_passes(e: dict[str, Any]) -> bool:
    return base_gates_pass(e) and str(e.get("screening_verdict", "")) in {"ACCEPT_H1", "PASS"}


def apply_evidence(args: argparse.Namespace) -> None:
    data = load(args.state)
    if args.id not in data["candidates"]:
        raise SystemExit(f"unknown candidate: {args.id}")
    e = json.loads(args.evidence.read_text(encoding="utf-8"))
    entry = data["candidates"][args.id]
    entry["evidence"].append({"at": now(), "file": str(args.evidence), "data": e})
    passed, failures = evidence_passes(e)
    verdict = str(e.get("screening_verdict", ""))

    if passed:
        previous = data.get("active")
        if previous and previous != args.id and previous in data["candidates"]:
            data["candidates"][previous]["status"] = "RETIRED"
            data["candidates"][previous]["retired_at"] = now()
        entry["status"] = "ACTIVE"
        entry["promoted_at"] = now()
        data["active"] = args.id
        data["generation"] = int(data.get("generation", 0)) + 1
        event = "PROMOTE"
    elif screen_passes(e):
        entry["status"] = "SCREENED"
        event = "SCREEN"
    elif base_gates_pass(e) and verdict in {"CONTINUE", "RETEST", ""}:
        # Inconclusive evidence is not negative evidence. Preserve the candidate
        # and its ancestry so more games can be appended later.
        entry["status"] = "PROVISIONAL"
        event = "RETEST"
    else:
        entry["status"] = "CONTESTED" if entry.get("status") == "ACTIVE" else "REJECTED"
        entry["failures"] = failures
        event = entry["status"]

    data["history"].append({"at": now(), "event": event, "candidate": args.id, "failures": failures})
    save(args.state, data)
    print(json.dumps({"candidate": args.id, "status": entry["status"], "event": event, "failures": failures}, sort_keys=True))


def contest(args: argparse.Namespace) -> None:
    data = load(args.state)
    if args.id not in data["candidates"]:
        raise SystemExit(f"unknown candidate: {args.id}")
    entry = data["candidates"][args.id]
    entry["status"] = "CONTESTED"
    entry["contest_reason"] = args.reason
    data["history"].append({"at": now(), "event": "CONTEST", "candidate": args.id, "reason": args.reason})
    save(args.state, data)


def rollback(args: argparse.Namespace) -> None:
    data = load(args.state)
    current = data.get("active")
    if not current or current not in data["candidates"]:
        raise SystemExit("no active candidate to roll back")
    parent = data["candidates"][current].get("parent")
    if not parent or parent not in data["candidates"]:
        raise SystemExit("active candidate has no registered rollback parent")
    data["candidates"][current]["status"] = "RETIRED"
    data["candidates"][current]["retired_at"] = now()
    data["candidates"][parent]["status"] = "ACTIVE"
    data["active"] = parent
    data["generation"] = int(data.get("generation", 0)) + 1
    data["history"].append({"at": now(), "event": "ROLLBACK", "from": current, "to": parent, "reason": args.reason})
    save(args.state, data)
    print(json.dumps({"active": parent, "rolled_back": current}, sort_keys=True))


def status(args: argparse.Namespace) -> None:
    data = load(args.state)
    counts: dict[str, int] = {}
    for c in data["candidates"].values():
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    print(json.dumps({"active": data.get("active"), "generation": data.get("generation", 0), "counts": counts}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--state", type=Path, required=True)
    r.add_argument("--artifact", type=Path, required=True)
    r.add_argument("--id")
    r.add_argument("--parent", default="ACTIVE")
    e = sub.add_parser("evidence")
    e.add_argument("--state", type=Path, required=True)
    e.add_argument("--id", required=True)
    e.add_argument("--evidence", type=Path, required=True)
    c = sub.add_parser("contest")
    c.add_argument("--state", type=Path, required=True)
    c.add_argument("--id", required=True)
    c.add_argument("--reason", required=True)
    rb = sub.add_parser("rollback")
    rb.add_argument("--state", type=Path, required=True)
    rb.add_argument("--reason", required=True)
    s = sub.add_parser("status")
    s.add_argument("--state", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "register":
        register(args)
    elif args.cmd == "evidence":
        apply_evidence(args)
    elif args.cmd == "contest":
        contest(args)
    elif args.cmd == "rollback":
        rollback(args)
    else:
        status(args)


if __name__ == "__main__":
    main()

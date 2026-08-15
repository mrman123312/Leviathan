"""Offline Search Foundry for bounded Leviathan Search-DSL candidates.

This is intentionally not runtime self-modification. The Foundry creates typed,
bounded candidate programs, performs static/range validation, records ancestry,
and emits promotion decisions from measured evidence. Engine games remain the
source of strength truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

FEATURES = 12
MAX_INS = 32


@dataclass
class Ins:
    op: str
    feature: int = 0
    a: int = 0
    b: int = 0


def parse_program(path: Path) -> list[Ins]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines or lines[0] != "LVSD1":
        raise ValueError("invalid LVSD1 magic")
    out: list[Ins] = []
    for line in lines[1:]:
        p = line.split()
        op = p[0]
        if op == "ADD" and len(p) == 2:
            out.append(Ins(op, a=int(p[1])))
        elif op == "MULADD" and len(p) == 4:
            out.append(Ins(op, int(p[1]), int(p[2]), int(p[3])))
        elif op in {"IFGT", "IFLT"} and len(p) == 4:
            out.append(Ins(op, int(p[1]), int(p[2]), int(p[3])))
        elif op == "CLAMP" and len(p) == 3:
            out.append(Ins(op, a=int(p[1]), b=int(p[2])))
        else:
            raise ValueError(f"bad instruction: {line}")
    validate_syntax(out)
    return out


def emit(program: list[Ins]) -> str:
    lines = ["LVSD1"]
    for i in program:
        if i.op == "ADD":
            lines.append(f"ADD {i.a}")
        elif i.op == "MULADD":
            lines.append(f"MULADD {i.feature} {i.a} {i.b}")
        elif i.op in {"IFGT", "IFLT"}:
            lines.append(f"{i.op} {i.feature} {i.a} {i.b}")
        elif i.op == "CLAMP":
            lines.append(f"CLAMP {i.a} {i.b}")
    return "\n".join(lines) + "\n"


def validate_syntax(program: list[Ins]) -> None:
    if not 1 <= len(program) <= MAX_INS:
        raise ValueError("program length out of range")
    for i in program:
        if i.op not in {"ADD", "MULADD", "IFGT", "IFLT", "CLAMP"}:
            raise ValueError("unknown op")
        if not 0 <= i.feature < FEATURES and i.op in {"MULADD", "IFGT", "IFLT"}:
            raise ValueError("feature out of range")
        if abs(i.a) > 65536 or abs(i.b) > 65536:
            raise ValueError("operand out of range")
        if i.op == "MULADD" and not 1 <= i.b <= 65536:
            raise ValueError("MULADD divisor out of range")
        if i.op == "CLAMP" and i.a > i.b:
            raise ValueError("invalid clamp")


def evaluate(program: list[Ins], x: list[int]) -> int:
    out = 0
    for i in program:
        if i.op == "ADD":
            out += i.a
        elif i.op == "MULADD":
            out += max(-4096, min(4096, x[i.feature])) * i.a // i.b
        elif i.op == "IFGT":
            out += i.b if x[i.feature] > i.a else 0
        elif i.op == "IFLT":
            out += i.b if x[i.feature] < i.a else 0
        elif i.op == "CLAMP":
            out = max(i.a, min(i.b, out))
        out = max(-4096, min(2048, out))
    return out


def range_test(program: list[Ins], seed: int = 8910, samples: int = 10000) -> dict:
    rng = random.Random(seed)
    lo, hi = 10**9, -10**9
    positive = negative = 0
    for _ in range(samples):
        x = [rng.randint(-256, 256) for _ in range(FEATURES)]
        # Boolean-ish control features use their live encoding.
        for idx in (4, 5, 6, 7, 8):
            x[idx] = rng.choice([0, 32])
        y = evaluate(program, x)
        lo, hi = min(lo, y), max(hi, y)
        positive += y > 0
        negative += y < 0
        if y < -4096 or y > 2048:
            raise ValueError("runtime range invariant violated")
    return {"min": lo, "max": hi, "positive": positive, "negative": negative, "samples": samples}


def random_ins(rng: random.Random) -> Ins:
    op = rng.choice(["ADD", "MULADD", "IFGT", "IFLT"])
    if op == "ADD":
        return Ins(op, a=rng.randint(-512, 256))
    if op == "MULADD":
        return Ins(op, rng.randrange(FEATURES), rng.randint(-256, 128), rng.choice([16, 32, 64, 128, 256]))
    return Ins(op, rng.randrange(FEATURES), rng.randint(-96, 128), rng.randint(-768, 384))


def new_program(rng: random.Random, length: int) -> list[Ins]:
    p = [random_ins(rng) for _ in range(max(1, min(length, MAX_INS - 1)))]
    p.append(Ins("CLAMP", a=-2048, b=1024))
    return p


def mutate(parent: list[Ins], rng: random.Random) -> list[Ins]:
    p = [Ins(**asdict(i)) for i in parent]
    action = rng.choice(["replace", "insert", "delete", "nudge"])
    if action == "replace":
        p[rng.randrange(len(p))] = random_ins(rng)
    elif action == "insert" and len(p) < MAX_INS:
        p.insert(rng.randrange(len(p) + 1), random_ins(rng))
    elif action == "delete" and len(p) > 1:
        del p[rng.randrange(len(p))]
    else:
        j = rng.randrange(len(p))
        if p[j].op == "CLAMP":
            p[j].a = max(-4096, p[j].a + rng.randint(-128, 128))
            p[j].b = min(2048, p[j].b + rng.randint(-128, 128))
            if p[j].a > p[j].b:
                p[j].a, p[j].b = p[j].b, p[j].a
        else:
            p[j].a = max(-65536, min(65536, p[j].a + rng.randint(-64, 64)))
    if not any(i.op == "CLAMP" for i in p):
        p.append(Ins("CLAMP", a=-2048, b=1024))
    validate_syntax(p)
    return p


def write_candidate(program: list[Ins], out: Path, parent: str, seed: int) -> None:
    text = emit(program)
    digest = hashlib.sha256(text.encode()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    meta = {
        "candidate_id": digest[:16],
        "sha256": digest,
        "parent": parent,
        "seed": seed,
        "range_test": range_test(program),
        "status": "PROVISIONAL",
        "authority": "veto-only-first",
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


def decide(evidence: Path) -> None:
    e = json.loads(evidence.read_text(encoding="utf-8"))
    nps = float(e.get("nps_delta_pct", 0.0))
    elo = float(e.get("elo", 0.0))
    los = float(e.get("los", 0.5))
    games = int(e.get("games", 0))
    if nps < -5.0:
        decision = "REJECT_NPS"
    elif games < 100:
        decision = "MORE_GAMES"
    elif elo > 0 and los >= 0.95:
        decision = "PROMOTE_SCREENING"
    elif elo < 0 and los <= 0.05:
        decision = "REJECT"
    else:
        decision = "RETEST"
    print(json.dumps({"decision": decision, "evidence": e}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--out", type=Path, required=True)
    n.add_argument("--length", type=int, default=5)
    n.add_argument("--seed", type=int, default=8910)
    m = sub.add_parser("mutate")
    m.add_argument("--parent", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--seed", type=int, default=8910)
    v = sub.add_parser("validate")
    v.add_argument("path", type=Path)
    d = sub.add_parser("decide")
    d.add_argument("evidence", type=Path)
    args = ap.parse_args()

    if args.cmd == "new":
        rng = random.Random(args.seed)
        write_candidate(new_program(rng, args.length), args.out, "NONE", args.seed)
    elif args.cmd == "mutate":
        rng = random.Random(args.seed)
        parent = parse_program(args.parent)
        write_candidate(mutate(parent, rng), args.out, str(args.parent), args.seed)
    elif args.cmd == "validate":
        p = parse_program(args.path)
        print(json.dumps(range_test(p), sort_keys=True))
    else:
        decide(args.evidence)


if __name__ == "__main__":
    main()

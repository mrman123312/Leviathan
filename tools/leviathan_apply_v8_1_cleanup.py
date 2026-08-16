#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)

p = Path("src/search.cpp")
s = p.read_text()

s = replace_once(
    s,
    "    bool  capture, ttCapture, leviathanNullFragile;\n",
    "    bool  capture, ttCapture;\n",
    "obsolete null-fragility declaration",
)
s = replace_once(
    s,
    "    leviathanNullFragile = false;\n",
    "",
    "obsolete null-fragility initialization",
)
s = replace_once(
    s,
    "            leviathanNullFragile = true;\n            leviathanEvidence.add(Leviathan::Evidence::Kind::NULL_FRAGILITY, 2);\n",
    "            leviathanEvidence.add(Leviathan::Evidence::Kind::NULL_FRAGILITY, 2);\n",
    "obsolete null-fragility assignment",
)

p.write_text(s)

final = p.read_text()
if "leviathanNullFragile" in final:
    raise SystemExit("obsolete leviathanNullFragile symbol still present")
if "Kind::NULL_FRAGILITY" not in final:
    raise SystemExit("typed NULL_FRAGILITY evidence was accidentally removed")
print("V8.1 cleanup applied: obsolete scalar null-fragility state removed")

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def materialize(root: Path) -> None:
    p = root / "src" / "leviathan_fundamentals.h"
    s = p.read_text()

    start = s.index("inline int lmr_adjustment(")
    end = s.index("\ninline int quiet_ordering_bonus", start)
    f = s[start:end]

    f = replace_once(
        f,
        "    if (!ready())\n        return 0;",
        "    const State& cfg = state();\n    if (!cfg.enabled || cfg.authority <= 0)\n        return 0;",
        "lmr ready",
    )
    f = f.replace("state().forcingBuyback", "cfg.forcingBuyback")
    f = replace_once(f, "if (recapture(move, prevSq, capture))", "if (isRecapture)", "recapture use")
    f = f.replace("state().recaptureBuyback", "cfg.recaptureBuyback")
    f = replace_once(
        f,
        "if (move.type_of() == PROMOTION || advanced_pawn_move(pos, move))",
        "if (isPromotion || isAdvancedPawn)",
        "promotion/pawn use",
    )
    f = f.replace("state().passerBuyback", "cfg.passerBuyback")
    f = replace_once(f, "if (low_material(pos) &&", "if (isLowMaterial &&", "low material use")
    f = f.replace("state().endgameBuyback", "cfg.endgameBuyback")
    f = f.replace("state().authority >= 2", "cfg.authority >= 2")
    f = replace_once(
        f,
        "move.type_of() != PROMOTION && !advanced_pawn_move(pos, move)",
        "!isPromotion && !isAdvancedPawn",
        "quiet promotion/pawn use",
    )
    f = replace_once(f, "!low_material(pos)", "!isLowMaterial", "quiet low material use")
    f = f.replace("state().quietOverdrive", "cfg.quietOverdrive")
    f = replace_once(
        f,
        "    int delta = 0;",
        "    const bool isRecapture = recapture(move, prevSq, capture);\n"
        "    const bool isPromotion = move.type_of() == PROMOTION;\n"
        "    const bool isAdvancedPawn = advanced_pawn_move(pos, move);\n"
        "    const bool isLowMaterial = low_material(pos);\n\n"
        "    int delta = 0;",
        "classification cache",
    )
    if "const bool isRecapture = isRecapture" in f or "const bool isLowMaterial = isLowMaterial" in f:
        raise SystemExit("self-reference introduced")
    s = s[:start] + f + s[end:]

    s = replace_once(
        s,
        "inline bool ready() { return state().enabled && state().authority > 0; }",
        "inline bool ready() {\n    const State& cfg = state();\n    return cfg.enabled && cfg.authority > 0;\n}",
        "ready",
    )
    s = replace_once(
        s,
        "inline bool zugzwang_risk(const Position& pos) {\n    if (!ready() || !state().zugzwangGuard)\n        return false;",
        "inline bool zugzwang_risk(const Position& pos) {\n    const State& cfg = state();\n"
        "    if (!cfg.enabled || cfg.authority <= 0 || !cfg.zugzwangGuard)\n        return false;",
        "zugzwang",
    )

    start = s.index("inline int quiet_ordering_bonus(")
    end = s.index("\ninline bool allow_null_move", start)
    q = s[start:end]
    q = replace_once(
        q,
        "    if (!ready() || !state().rule50Pressure || pos.rule50_count() < 70)",
        "    const State& cfg = state();\n"
        "    if (!cfg.enabled || cfg.authority <= 0 || !cfg.rule50Pressure || pos.rule50_count() < 70)",
        "quiet ordering ready",
    )
    q = q.replace("state().rule50PawnBonus", "cfg.rule50PawnBonus")
    s = s[:start] + q + s[end:]

    p.write_text(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    args = ap.parse_args()
    materialize(Path(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Guarded, idempotent integration for Leviathan Fundamentals v2."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{path}: expected one anchor for {marker!r}, found {n}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_engine() -> bool:
    p = ROOT / "src" / "engine.cpp"
    changed = False
    changed |= replace_once(
        p,
        '#include "leviathan_dsl.h"\n',
        '#include "leviathan_dsl.h"\n#include "leviathan_fundamentals.h"\n',
        '#include "leviathan_fundamentals.h"',
    )

    anchor = '''    options.add("Leviathan Specialist Veto", Option(768, 0, 4096, [](const Option& o) {\n        Leviathan::Control::set_specialist_veto(int(o));\n        return std::nullopt;\n    }));\n\n    options.add("Leviathan Atlas", Option(false, [](const Option& o) {\n'''
    block = '''    options.add("Leviathan Specialist Veto", Option(768, 0, 4096, [](const Option& o) {\n        Leviathan::Control::set_specialist_veto(int(o));\n        return std::nullopt;\n    }));\n\n    // Fundamentals v2: deterministic, pre-training search regime controls.\n    options.add("Leviathan Fundamentals", Option(false, [](const Option& o) {\n        Leviathan::Fundamentals::set_enabled(int(o) != 0);\n        return std::nullopt;\n    }));\n    options.add("Leviathan Fundamentals Authority", Option(0, 0, 2, [](const Option& o) {\n        Leviathan::Fundamentals::set_authority(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Forcing Buyback", Option(384, 0, 2048, [](const Option& o) {\n        Leviathan::Fundamentals::set_forcing_buyback(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Recapture Buyback", Option(256, 0, 2048, [](const Option& o) {\n        Leviathan::Fundamentals::set_recapture_buyback(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Passer Buyback", Option(320, 0, 2048, [](const Option& o) {\n        Leviathan::Fundamentals::set_passer_buyback(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Endgame Buyback", Option(128, 0, 1024, [](const Option& o) {\n        Leviathan::Fundamentals::set_endgame_buyback(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Quiet Overdrive", Option(160, 0, 1024, [](const Option& o) {\n        Leviathan::Fundamentals::set_quiet_overdrive(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Rule50 Pawn Bonus", Option(3072, 0, 16384, [](const Option& o) {\n        Leviathan::Fundamentals::set_rule50_pawn_bonus(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Zugzwang Guard", Option(true, [](const Option& o) {\n        Leviathan::Fundamentals::set_zugzwang_guard(int(o) != 0);\n        return std::nullopt;\n    }));\n    options.add("Leviathan Sacrifice Rescue", Option(true, [](const Option& o) {\n        Leviathan::Fundamentals::set_sacrifice_rescue(int(o) != 0);\n        return std::nullopt;\n    }));\n    options.add("Leviathan Rule50 Pressure", Option(true, [](const Option& o) {\n        Leviathan::Fundamentals::set_rule50_pressure(int(o) != 0);\n        return std::nullopt;\n    }));\n\n    options.add("Leviathan Atlas", Option(false, [](const Option& o) {\n'''
    changed |= replace_once(p, anchor, block, 'options.add("Leviathan Fundamentals"')
    return changed


def patch_movepick() -> bool:
    p = ROOT / "src" / "movepick.cpp"
    changed = False
    changed |= replace_once(
        p,
        '#include "leviathan_atlas.h"\n',
        '#include "leviathan_atlas.h"\n#include "leviathan_fundamentals.h"\n',
        '#include "leviathan_fundamentals.h"',
    )
    changed |= replace_once(
        p,
        '            m.value += Leviathan::Atlas::ordering_bonus(pos, m);\n',
        '            m.value += Leviathan::Atlas::ordering_bonus(pos, m);\n            m.value += Leviathan::Fundamentals::quiet_ordering_bonus(pos, m);\n',
        'Fundamentals::quiet_ordering_bonus',
    )
    return changed


def patch_search() -> bool:
    p = ROOT / "src" / "search.cpp"
    changed = False
    changed |= replace_once(
        p,
        '#include "leviathan_dsl.h"\n',
        '#include "leviathan_dsl.h"\n#include "leviathan_fundamentals.h"\n',
        '#include "leviathan_fundamentals.h"',
    )

    # Guard null-move pruning in sparse zugzwang-prone positions.
    changed |= replace_once(
        p,
        '        && pos.non_pawn_material(us) && ss->ply >= nmpMinPly && beta >= -2000)\n',
        '        && pos.non_pawn_material(us) && ss->ply >= nmpMinPly && beta >= -2000\n        && Leviathan::Fundamentals::allow_null_move(pos))\n',
        'Fundamentals::allow_null_move(pos)',
    )

    # Rescue narrow classes from SEE pruning (checking sacrifices, recaptures, promotions).
    old_see = '''                if ((alpha >= VALUE_DRAW || pos.non_pawn_material(us) != PieceValue[movedPiece])\n                    && !pos.see_ge(move, -margin))\n                    continue;\n'''
    new_see = '''                if ((alpha >= VALUE_DRAW || pos.non_pawn_material(us) != PieceValue[movedPiece])\n                    && !pos.see_ge(move, -margin)\n                    && !Leviathan::Fundamentals::rescue_bad_see(\n                      pos, move, prevSq, capture, givesCheck))\n                    continue;\n'''
    changed |= replace_once(p, old_see, new_see, 'Fundamentals::rescue_bad_see')

    # Protect advanced pawn pushes from the quiet shallow-pruning stack.
    old_quiet = '''            else if (!ss->followPV || !PvNode)\n            {\n                int dIndex  = std::min(int(depth), int(lmrDivisor.size())) - 1;\n                int history = (*contHist[0])[movedPiece][move.to_sq()]\n                            + (*contHist[1])[movedPiece][move.to_sq()]\n                            + sharedHistory.pawn_entry(pos)[movedPiece][move.to_sq()];\n\n                // Continuation history based pruning\n                if (history < -4136 * depth)\n                    continue;\n'''
    new_quiet = '''            else if (!ss->followPV || !PvNode)\n            {\n                const bool leviathanScopeProtected =\n                  Leviathan::Fundamentals::protected_scope_move(\n                    pos, move, prevSq, capture, givesCheck);\n                int dIndex  = std::min(int(depth), int(lmrDivisor.size())) - 1;\n                int history = (*contHist[0])[movedPiece][move.to_sq()]\n                            + (*contHist[1])[movedPiece][move.to_sq()]\n                            + sharedHistory.pawn_entry(pos)[movedPiece][move.to_sq()];\n\n                // Continuation history based pruning\n                if (!leviathanScopeProtected && history < -4136 * depth)\n                    continue;\n'''
    changed |= replace_once(p, old_quiet, new_quiet, 'leviathanScopeProtected')

    changed |= replace_once(
        p,
        '                if (!ss->inCheck && lmrDepth < 12 && futilityValue <= alpha)\n',
        '                if (!leviathanScopeProtected && !ss->inCheck && lmrDepth < 12\n                    && futilityValue <= alpha)\n',
        '!leviathanScopeProtected && !ss->inCheck',
    )
    changed |= replace_once(
        p,
        '                if (!pos.see_ge(move, -23 * lmrDepth * lmrDepth))\n                    continue;\n',
        '                if (!leviathanScopeProtected\n                    && !pos.see_ge(move, -23 * lmrDepth * lmrDepth))\n                    continue;\n',
        '!leviathanScopeProtected\n                    && !pos.see_ge',
    )

    # Add the balanced deterministic allocator after the learned/DSL controls.
    old_lmr = '''        r += Leviathan::DSL::lmr_adjustment(\n          depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,\n          givesCheck, ttData.depth, ss->staticEval, alpha);\n\n        Value leviathanReducedValue = VALUE_NONE;\n'''
    new_lmr = '''        r += Leviathan::DSL::lmr_adjustment(\n          depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,\n          givesCheck, ttData.depth, ss->staticEval, alpha);\n        r += Leviathan::Fundamentals::lmr_adjustment(\n          pos, move, prevSq, depth, moveCount, PvNode, capture, givesCheck);\n\n        Value leviathanReducedValue = VALUE_NONE;\n'''
    changed |= replace_once(p, old_lmr, new_lmr, 'Fundamentals::lmr_adjustment(')
    return changed


def main() -> None:
    changed = []
    if patch_engine(): changed.append("src/engine.cpp")
    if patch_movepick(): changed.append("src/movepick.cpp")
    if patch_search(): changed.append("src/search.cpp")
    print("patched=" + (",".join(changed) if changed else "none"))


if __name__ == "__main__":
    main()

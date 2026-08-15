"""Apply the small source hooks required by Leviathan Immortal v1.

The connector cannot apply text patches directly, so this script is run in CI on
the pinned research branch. Every edit is guarded by an exact Stockfish anchor;
if upstream structure changes, it fails instead of guessing. It is idempotent.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one anchor for {marker!r}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_engine() -> bool:
    p = ROOT / "src" / "engine.cpp"
    changed = False
    changed |= replace_once(
        p,
        '#include "evaluate.h"\n',
        '#include "evaluate.h"\n#include "leviathan_atlas.h"\n#include "leviathan_control.h"\n#include "leviathan_dsl.h"\n#include "leviathan_policy.h"\n#include "leviathan_trace.h"\n',
        '#include "leviathan_control.h"',
    )

    anchor = '''    options.add("SyzygyProbeLimit", Option(7, 0, 7));\n\n    options.add(  //\n      "EvalFile", Option(EvalFileDefaultName, [this](const Option& o) {\n'''
    block = '''    options.add("SyzygyProbeLimit", Option(7, 0, 7));\n\n    // Project Leviathan research controls. Every organ defaults to disabled.\n    options.add("Leviathan Policy", Option(false, [](const Option& o) {\n        Leviathan::Policy::set_enabled(int(o) != 0);\n        return std::nullopt;\n    }));\n    options.add("Leviathan Policy File", Option("", [](const Option& o) {\n        const std::string path = std::string(o);\n        const bool ok = Leviathan::Policy::set_model_path(path);\n        return !path.empty() && !ok ? std::optional<std::string>("Leviathan: invalid policy model")\n                                    : std::nullopt;\n    }));\n    options.add("Leviathan Policy Weight", Option(100, 0, 400, [](const Option& o) {\n        Leviathan::Policy::set_weight(int(o));\n        return std::nullopt;\n    }));\n\n    options.add("Leviathan MetaSearch", Option(false, [](const Option& o) {\n        Leviathan::Control::set_meta_enabled(int(o) != 0);\n        return std::nullopt;\n    }));\n    options.add("Leviathan Meta File", Option("", [](const Option& o) {\n        const std::string path = std::string(o);\n        const bool ok = Leviathan::Control::set_meta_file(path);\n        return !path.empty() && !ok ? std::optional<std::string>("Leviathan: invalid MetaSearch model")\n                                    : std::nullopt;\n    }));\n    options.add("Leviathan Meta Authority", Option(0, 0, 2, [](const Option& o) {\n        Leviathan::Control::set_meta_authority(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Meta Max Percent", Option(30, 0, 100, [](const Option& o) {\n        Leviathan::Control::set_meta_max_percent(int(o));\n        return std::nullopt;\n    }));\n\n    options.add("Leviathan Risk", Option(false, [](const Option& o) {\n        Leviathan::Control::set_risk_enabled(int(o) != 0);\n        return std::nullopt;\n    }));\n    options.add("Leviathan Risk File", Option("", [](const Option& o) {\n        const std::string path = std::string(o);\n        const bool ok = Leviathan::Control::set_risk_file(path);\n        return !path.empty() && !ok ? std::optional<std::string>("Leviathan: invalid risk model")\n                                    : std::nullopt;\n    }));\n    options.add("Leviathan Risk Authority", Option(0, 0, 2, [](const Option& o) {\n        Leviathan::Control::set_risk_authority(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Risk Threshold", Option(650, 0, 1000, [](const Option& o) {\n        Leviathan::Control::set_risk_threshold(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Risk Veto", Option(1536, 0, 4096, [](const Option& o) {\n        Leviathan::Control::set_risk_veto(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Specialist", Option(false, [](const Option& o) {\n        Leviathan::Control::set_specialist_enabled(int(o) != 0);\n        return std::nullopt;\n    }));\n    options.add("Leviathan Specialist Veto", Option(768, 0, 4096, [](const Option& o) {\n        Leviathan::Control::set_specialist_veto(int(o));\n        return std::nullopt;\n    }));\n\n    options.add("Leviathan Atlas", Option(false, [](const Option& o) {\n        Leviathan::Atlas::set_enabled(int(o) != 0);\n        return std::nullopt;\n    }));\n    options.add("Leviathan Atlas File", Option("", [](const Option& o) {\n        const std::string path = std::string(o);\n        const bool ok = Leviathan::Atlas::set_file(path);\n        return !path.empty() && !ok ? std::optional<std::string>("Leviathan: invalid Atlas")\n                                    : std::nullopt;\n    }));\n    options.add("Leviathan Atlas Weight", Option(100, 0, 400, [](const Option& o) {\n        Leviathan::Atlas::set_weight(int(o));\n        return std::nullopt;\n    }));\n\n    options.add("Leviathan Search DSL", Option(false, [](const Option& o) {\n        Leviathan::DSL::set_enabled(int(o) != 0);\n        return std::nullopt;\n    }));\n    options.add("Leviathan Search DSL File", Option("", [](const Option& o) {\n        const std::string path = std::string(o);\n        const bool ok = Leviathan::DSL::set_file(path);\n        return !path.empty() && !ok ? std::optional<std::string>("Leviathan: invalid Search DSL")\n                                    : std::nullopt;\n    }));\n    options.add("Leviathan Search DSL Authority", Option(0, 0, 2, [](const Option& o) {\n        Leviathan::DSL::set_authority(int(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Search DSL Weight", Option(100, 0, 400, [](const Option& o) {\n        Leviathan::DSL::set_weight(int(o));\n        return std::nullopt;\n    }));\n\n    options.add("Leviathan Trace File", Option("", [](const Option& o) {\n        Leviathan::Trace::set_file(std::string(o));\n        return std::nullopt;\n    }));\n    options.add("Leviathan Trace Sample Permille", Option(0, 0, 1000, [](const Option& o) {\n        Leviathan::Trace::set_sample_permille(int(o));\n        return std::nullopt;\n    }));\n\n    options.add(  //\n      "EvalFile", Option(EvalFileDefaultName, [this](const Option& o) {\n'''
    changed |= replace_once(p, anchor, block, 'options.add("Leviathan Policy"')
    return changed


def patch_movepick() -> bool:
    p = ROOT / "src" / "movepick.cpp"
    changed = False
    changed |= replace_once(
        p,
        '#include "bitboard.h"\n#include "leviathan_policy.h"\n',
        '#include "bitboard.h"\n#include "leviathan_atlas.h"\n#include "leviathan_policy.h"\n',
        '#include "leviathan_atlas.h"',
    )
    changed |= replace_once(
        p,
        '            m.value += Leviathan::Policy::ordering_bonus(pos, m);\n',
        '            m.value += Leviathan::Policy::ordering_bonus(pos, m);\n            m.value += Leviathan::Atlas::ordering_bonus(pos, m);\n',
        'Leviathan::Atlas::ordering_bonus(pos, m)',
    )
    return changed


def patch_search() -> bool:
    p = ROOT / "src" / "search.cpp"
    changed = False
    changed |= replace_once(
        p,
        '#include "history.h"\n',
        '#include "history.h"\n#include "leviathan_control.h"\n#include "leviathan_dsl.h"\n#include "leviathan_trace.h"\n',
        '#include "leviathan_dsl.h"',
    )

    old_time = '''            double totalTime = mainThread->tm.optimum() * fallingEval * reduction\n                             * bestMoveInstability * highBestMoveEffort;\n\n            if (rootMoves.size() == 1)\n'''
    new_time = '''            double totalTime = mainThread->tm.optimum() * fallingEval * reduction\n                             * bestMoveInstability * highBestMoveEffort;\n\n            // Leviathan P003: MetaSearch first receives bounded time-allocation\n            // authority. Authority 1 can only buy more verification time.\n            totalTime *= Leviathan::Control::meta_time_factor(\n              rootDepth, rootDepth - lastBestMoveDepth, rootMoves[0].previousScore, bestValue,\n              totBestMoveChanges, int(nodesEffort), rootMoves.size(), is_decisive(bestValue));\n\n            if (rootMoves.size() == 1)\n'''
    changed |= replace_once(p, old_time, new_time, 'Leviathan P003: MetaSearch')

    old_key = '''        u64 nodeCount = rootNode ? u64(nodes) : 0;\n\n        // Step 16. Make the move\n'''
    new_key = '''        u64 nodeCount = rootNode ? u64(nodes) : 0;\n        const u64 leviathanParentKey = u64(pos.key());\n\n        // Step 16. Make the move\n'''
    changed |= replace_once(p, old_key, new_key, 'leviathanParentKey')

    old_lmr = '''        // Scale up reductions for expected ALL nodes\n        if (allNode)\n            r += r * 276 / (256 * depth + 268);\n\n        // Step 17. Late moves reduction / extension (LMR)\n'''
    new_lmr = '''        // Scale up reductions for expected ALL nodes\n        if (allNode)\n            r += r * 276 / (256 * depth + 268);\n\n        // Leviathan P004/P005/P007: learned risk may first veto unsafe\n        // reductions; Search-DSL candidates are separately bounded and gated.\n        r += Leviathan::Control::lmr_adjustment(\n          depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,\n          givesCheck, ttData.depth, ss->staticEval, alpha);\n        r += Leviathan::DSL::lmr_adjustment(\n          depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,\n          givesCheck, ttData.depth, ss->staticEval, alpha);\n\n        Value leviathanReducedValue = VALUE_NONE;\n        Depth leviathanReducedDepth = DEPTH_NONE;\n        bool  leviathanResearched   = false;\n\n        // Step 17. Late moves reduction / extension (LMR)\n'''
    changed |= replace_once(p, old_lmr, new_lmr, 'Leviathan P004/P005/P007')

    old_d = '''            Depth d = std::max(1, std::min(newDepth - r / 1024, newDepth + 2)) + PvNode;\n\n            ss->reduction = newDepth - d;\n            value         = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);\n            ss->reduction = 0;\n'''
    new_d = '''            Depth d = std::max(1, std::min(newDepth - r / 1024, newDepth + 2)) + PvNode;\n            leviathanReducedDepth = d;\n\n            ss->reduction = newDepth - d;\n            value         = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);\n            ss->reduction = 0;\n            leviathanReducedValue = value;\n'''
    changed |= replace_once(p, old_d, new_d, 'leviathanReducedDepth = d')

    old_re = '''                if (newDepth > d)\n                    value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);\n\n                // Post LMR continuation history updates\n'''
    new_re = '''                if (newDepth > d)\n                {\n                    leviathanResearched = true;\n                    value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);\n                }\n\n                // Post LMR continuation history updates\n'''
    changed |= replace_once(p, old_re, new_re, 'leviathanResearched = true')

    old_pv = '''        if (PvNode && (moveCount == 1 || value > alpha))\n        {\n            (ss + 1)->pv = &pv;\n'''
    new_pv = '''        if (PvNode && (moveCount == 1 || value > alpha))\n        {\n            if (leviathanReducedValue != VALUE_NONE)\n                leviathanResearched = true;\n            (ss + 1)->pv = &pv;\n'''
    changed |= replace_once(p, old_pv, new_pv, 'if (leviathanReducedValue != VALUE_NONE)')

    old_undo = '''        // Step 19. Undo move\n        undo_move(pos, move);\n'''
    new_undo = '''        if (leviathanReducedValue != VALUE_NONE)\n            Leviathan::Trace::record_lmr(\n              leviathanParentKey, move.raw(),\n              Leviathan::Trace::features(depth, moveCount, ss->statScore, correctionValue, PvNode,\n                                          cutNode, allNode, capture, givesCheck, ttData.depth,\n                                          ss->staticEval, alpha),\n              leviathanReducedValue, value, leviathanReducedDepth, newDepth, leviathanResearched);\n\n        // Step 19. Undo move\n        undo_move(pos, move);\n'''
    changed |= replace_once(p, old_undo, new_undo, 'Leviathan::Trace::record_lmr')
    return changed


def main() -> None:
    changed = []
    if patch_engine():
        changed.append("src/engine.cpp")
    if patch_movepick():
        changed.append("src/movepick.cpp")
    if patch_search():
        changed.append("src/search.cpp")
    print("patched=" + (",".join(changed) if changed else "none"))


if __name__ == "__main__":
    main()

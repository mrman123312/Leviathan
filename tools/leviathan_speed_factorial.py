#!/usr/bin/env python3
"""Strict patch applicator and paired speed-factorial harness for Leviathan.

This tool deliberately separates source mutation from measurement.  Any missing
or duplicated anchor is fatal, and any node-signature divergence is fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor in {path}, found {count}")
    path.write_text(text.replace(old, new, 1))


def apply_p0(root: Path) -> None:
    """Remove dormant DSL construction and expose an exact trace readiness test."""
    path = root / "src/leviathan_dsl.h"
    replace_once(
        path,
        """inline bool ready() { return state().enabled && state().authority > 0 && state().loaded; }\n\ninline int eval(const std::array<int, FeatureCount>& x) {\n    if (!ready())\n        return 0;\n\n    int out = 0;\n    for (const auto& ins : state().code)""",
        """inline bool ready() {\n    const auto& s = state();\n    return s.enabled && s.authority > 0 && s.loaded;\n}\n\ninline int eval_ready(const std::array<int, FeatureCount>& x) {\n    const auto& s = state();\n    int out = 0;\n    for (const auto& ins : s.code)""",
        "p0-dsl-head",
    )
    replace_once(
        path,
        """    if (state().authority == 1)\n        out = std::min(out, 0);\n    return std::clamp(out * state().weight / 100, -4096, 1536);\n}\n\ninline int lmr_adjustment(Depth depth,""",
        """    if (s.authority == 1)\n        out = std::min(out, 0);\n    return std::clamp(out * s.weight / 100, -4096, 1536);\n}\n\ninline int eval(const std::array<int, FeatureCount>& x) {\n    return ready() ? eval_ready(x) : 0;\n}\n\ninline int lmr_adjustment(Depth depth,""",
        "p0-dsl-tail",
    )
    replace_once(
        path,
        """                          Value staticEval,\n                          Value alpha) {\n    const std::array<int, FeatureCount> x = {""",
        """                          Value staticEval,\n                          Value alpha) {\n    if (!ready())\n        return 0;\n\n    const std::array<int, FeatureCount> x = {""",
        "p0-dsl-gate",
    )
    replace_once(path, "    return eval(x);", "    return eval_ready(x);", "p0-dsl-eval")

    path = root / "src/leviathan_trace.h"
    replace_once(
        path,
        """inline State& state() {\n    static State s;\n    return s;\n}\n\ninline void set_file(const std::string& path) {""",
        """inline State& state() {\n    static State s;\n    return s;\n}\n\ninline bool ready() {\n    const auto& s = state();\n    return s.samplePermille > 0 && !s.file.empty();\n}\n\ninline void set_file(const std::string& path) {""",
        "p0-trace-ready",
    )


def apply_p1(root: Path) -> None:
    """Cache optional-organ readiness once per node/list and skip dormant hooks."""
    path = root / "src/search.cpp"
    replace_once(
        path,
        """    int moveCount = 0;\n\n    // Step 13.""",
        """    int moveCount = 0;\n\n    const bool leviathanRiskReady  = Leviathan::Control::risk_ready();\n    const bool leviathanDslReady   = Leviathan::DSL::ready();\n    const bool leviathanTraceReady = Leviathan::Trace::ready();\n\n    // Step 13.""",
        "p1-search-readiness",
    )
    replace_once(
        path,
        "        const u64 leviathanParentKey = u64(pos.key());",
        "        const u64 leviathanParentKey = leviathanTraceReady ? u64(pos.key()) : 0;",
        "p1-search-key",
    )
    replace_once(
        path,
        """        r += Leviathan::Control::lmr_adjustment(\n          depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,\n          givesCheck, ttData.depth, ss->staticEval, alpha);\n        r += Leviathan::DSL::lmr_adjustment(\n          depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,\n          givesCheck, ttData.depth, ss->staticEval, alpha);""",
        """        if (leviathanRiskReady)\n            r += Leviathan::Control::lmr_adjustment(\n              depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,\n              givesCheck, ttData.depth, ss->staticEval, alpha);\n        if (leviathanDslReady)\n            r += Leviathan::DSL::lmr_adjustment(\n              depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,\n              givesCheck, ttData.depth, ss->staticEval, alpha);""",
        "p1-search-hooks",
    )
    replace_once(
        path,
        """        if (leviathanReducedValue != VALUE_NONE)\n            Leviathan::Trace::record_lmr(""",
        """        if (leviathanTraceReady && leviathanReducedValue != VALUE_NONE)\n            Leviathan::Trace::record_lmr(""",
        "p1-search-trace",
    )

    path = root / "src/movepick.cpp"
    replace_once(
        path,
        """    ExtMove* it = cur;\n    for (auto move : ml)""",
        """    const bool leviathanPolicyReady = Type == QUIETS && Leviathan::Policy::ready();\n    const bool leviathanAtlasReady  = Type == QUIETS && Leviathan::Atlas::ready();\n    const bool leviathanRule50Ready = Type == QUIETS && Leviathan::Fundamentals::ready()\n                                      && Leviathan::Fundamentals::state().rule50Pressure\n                                      && pos.rule50_count() >= 70;\n\n    ExtMove* it = cur;\n    for (auto move : ml)""",
        "p1-movepick-readiness",
    )
    replace_once(
        path,
        """            m.value += Leviathan::Policy::ordering_bonus(pos, m);\n            m.value += Leviathan::Atlas::ordering_bonus(pos, m);\n            m.value += Leviathan::Fundamentals::quiet_ordering_bonus(pos, m);""",
        """            if (leviathanPolicyReady)\n                m.value += Leviathan::Policy::ordering_bonus(pos, m);\n            if (leviathanAtlasReady)\n                m.value += Leviathan::Atlas::ordering_bonus(pos, m);\n            if (leviathanRule50Ready)\n                m.value += Leviathan::Fundamentals::quiet_ordering_bonus(pos, m);""",
        "p1-movepick-hooks",
    )


def apply_p2(root: Path) -> None:
    """Hoist the pawn-history entry once per quiet MovePicker list."""
    path = root / "src/movepick.cpp"
    replace_once(
        path,
        """    ExtMove* it = cur;\n    for (auto move : ml)""",
        """    const auto* pawnEntry = Type == QUIETS ? &sharedHistory->pawn_entry(pos) : nullptr;\n\n    ExtMove* it = cur;\n    for (auto move : ml)""",
        "p2-entry",
    )
    replace_once(
        path,
        "            m.value += 2 * sharedHistory->pawn_entry(pos)[pc][to];",
        "            m.value += 2 * (*pawnEntry)[pc][to];",
        "p2-use",
    )


def apply_p3(root: Path) -> None:
    """Specialize the exact one-add/two-remove NNUE capture update shape."""
    path = root / "src/nnue/nnue_accumulator.cpp"
    replace_once(
        path,
        """        apply_psq_features<-1>(j, acc, psqRemoved, featureTransformer);\n        apply_psq_features<+1>(j, acc, psqAdded, featureTransformer);""",
        """        if (psqAdded.ssize() == 1 && psqRemoved.ssize() == 2)\n        {\n            const usize tile = j * Tiling::TileHeight;\n            auto* rem0 = reinterpret_cast<const vec_t*>(&featureTransformer.weights[psqRemoved[0] * Dimensions + tile]);\n            auto* rem1 = reinterpret_cast<const vec_t*>(&featureTransformer.weights[psqRemoved[1] * Dimensions + tile]);\n            auto* add0 = reinterpret_cast<const vec_t*>(&featureTransformer.weights[psqAdded[0] * Dimensions + tile]);\n            for (IndexType k = 0; k < Tiling::NumRegs; ++k)\n                acc[k] = vec_add_16(vec_sub_16(vec_sub_16(acc[k], rem0[k]), rem1[k]), add0[k]);\n        }\n        else\n        {\n            apply_psq_features<-1>(j, acc, psqRemoved, featureTransformer);\n            apply_psq_features<+1>(j, acc, psqAdded, featureTransformer);\n        }""",
        "p3-accumulator",
    )
    replace_once(
        path,
        """        apply_psqt<-1>(j, psqt, psqRemoved, featureTransformer.psqtWeights.data());\n        apply_psqt<+1>(j, psqt, psqAdded, featureTransformer.psqtWeights.data());""",
        """        if (psqAdded.ssize() == 1 && psqRemoved.ssize() == 2)\n        {\n            const usize off = j * Tiling::PsqtTileHeight;\n            auto* rem0 = reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[psqRemoved[0] * PSQTBuckets + off]);\n            auto* rem1 = reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[psqRemoved[1] * PSQTBuckets + off]);\n            auto* add0 = reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[psqAdded[0] * PSQTBuckets + off]);\n            for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)\n                psqt[k] = vec_add_psqt_32(vec_sub_psqt_32(vec_sub_psqt_32(psqt[k], rem0[k]), rem1[k]), add0[k]);\n        }\n        else\n        {\n            apply_psqt<-1>(j, psqt, psqRemoved, featureTransformer.psqtWeights.data());\n            apply_psqt<+1>(j, psqt, psqAdded, featureTransformer.psqtWeights.data());\n        }""",
        "p3-psqt",
    )


PATCHES = {"p0": apply_p0, "p1": apply_p1, "p2": apply_p2, "p3": apply_p3}
VARIANTS = {
    "control_a": (),
    "control_b": (),
    "p0": ("p0",),
    "p01": ("p0", "p1"),
    "p02": ("p0", "p2"),
    "p03": ("p0", "p3"),
    "p012": ("p0", "p1", "p2"),
    "p013": ("p0", "p1", "p3"),
    "p0123": ("p0", "p1", "p2", "p3"),
}


UCI_OPTIONS = "\n".join(
    [
        "setoption name Threads value 1",
        "setoption name Hash value 64",
        "setoption name Leviathan Fundamentals value true",
        "setoption name Leviathan Fundamentals Authority value 2",
        "setoption name Leviathan Forcing Buyback value 384",
        "setoption name Leviathan Recapture Buyback value 256",
        "setoption name Leviathan Passer Buyback value 320",
        "setoption name Leviathan Endgame Buyback value 128",
        "setoption name Leviathan Quiet Overdrive value 160",
        "setoption name Leviathan Rule50 Pawn Bonus value 3072",
        "setoption name Leviathan Zugzwang Guard value true",
        "setoption name Leviathan Sacrifice Rescue value true",
        "setoption name Leviathan Rule50 Pressure value true",
        "setoption name Leviathan Policy value false",
        "setoption name Leviathan MetaSearch value false",
        "setoption name Leviathan Risk value false",
        "setoption name Leviathan Specialist value false",
        "setoption name Leviathan Atlas value false",
        "setoption name Leviathan Search DSL value false",
    ]
) + "\n"


def normalize_search_transcript(output: str) -> str:
    """Return a timing-independent transcript of every searched position.

    Aggregate node equality is necessary but not sufficient for a lossless
    claim: different trees can accidentally have the same total.  Preserve
    position headers, iterative scores/nodes/PVs, and best moves while removing
    only wall-clock-derived fields.
    """
    normalized = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Position:") or line.startswith("bestmove "):
            normalized.append(line)
        elif line.startswith("info ") and not line.startswith("info string "):
            line = re.sub(r"\s+nps\s+\d+", "", line)
            line = re.sub(r"\s+time\s+\d+", "", line)
            normalized.append(line)
    return "\n".join(normalized)


def run_bench(path: str, command: str = "bench") -> dict[str, int | str]:
    proc = subprocess.run(
        [path], input=UCI_OPTIONS + command + "\nquit\n", text=True, capture_output=True, check=True
    )
    output = proc.stdout + proc.stderr
    transcript = normalize_search_transcript(output)
    if not transcript:
        raise SystemExit(f"no normalized search transcript emitted by {path}")
    return {
        "ms": int(re.findall(r"Total time \(ms\)\s*:\s*(\d+)", output)[-1]),
        "nodes": int(re.findall(r"Nodes searched\s*:\s*(\d+)", output)[-1]),
        "nps": int(re.findall(r"Nodes/second\s*:\s*(\d+)", output)[-1]),
        "behavior_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
        "behavior_lines": len(transcript.splitlines()),
    }


def bootstrap_median_ci(values: list[float], seed: int, samples: int = 20000) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    boots = [statistics.median(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    boots.sort()
    return [boots[int(samples * 0.025)], boots[int(samples * 0.975)]]


def benchmark(engines: dict[str, str], rounds: int, output: Path) -> None:
    if "control_a" not in engines or "control_b" not in engines:
        raise SystemExit("control_a and control_b are required")

    signatures: dict[str, dict[str, dict[str, int | str]]] = {}
    for name, path in engines.items():
        signatures[name] = {}
        for label, command in {
            "default": "bench",
            "depth11": "bench 16 1 11 default depth",
            "nodes50k": "bench 16 1 50000 default nodes",
        }.items():
            result = run_bench(path, command)
            signatures[name][label] = {
                "nodes": result["nodes"],
                "behavior_sha256": result["behavior_sha256"],
                "behavior_lines": result["behavior_lines"],
            }
    reference_signature = signatures["control_a"]
    divergent = {name: sig for name, sig in signatures.items() if sig != reference_signature}
    if divergent:
        raise SystemExit(f"FUNCTIONAL DIVERGENCE: reference={reference_signature} divergent={divergent}")

    for path in engines.values():
        run_bench(path)

    candidates = [name for name in engines if name != "control_a"]
    observations = {name: [] for name in candidates}
    rng = random.Random(20260816)
    for round_index in range(rounds):
        order = candidates[:]
        rng.shuffle(order)
        if round_index % 2:
            order.reverse()
        for candidate in order:
            first = "control_a" if round_index % 2 == 0 else "control_b"
            second = "control_b" if round_index % 2 == 0 else "control_a"
            r1 = run_bench(engines[first])
            c = run_bench(engines[candidate])
            r2 = run_bench(engines[second])
            exact_keys = ("nodes", "behavior_sha256", "behavior_lines")
            if not all(r1[key] == c[key] == r2[key] for key in exact_keys):
                raise SystemExit(f"FUNCTIONAL DIVERGENCE during timing: {candidate} {r1} {c} {r2}")
            reference_ms = math.sqrt(r1["ms"] * r2["ms"])
            observations[candidate].append(
                {
                    "round": round_index,
                    "reference_first": first,
                    "reference_first_ms": r1["ms"],
                    "candidate_ms": c["ms"],
                    "reference_second": second,
                    "reference_second_ms": r2["ms"],
                    "sandwich_speedup": reference_ms / c["ms"],
                }
            )

    results = {}
    for index, (name, rows) in enumerate(observations.items()):
        ratios = [row["sandwich_speedup"] for row in rows]
        ci = bootstrap_median_ci(ratios, 9000 + index)
        median = statistics.median(ratios)
        faster = sum(ratio > 1.0 for ratio in ratios)
        if name == "control_b":
            status = "CALIBRATION_PASS" if abs(median - 1.0) <= 0.004 and ci[0] <= 1.0 <= ci[1] else "CALIBRATION_FAIL"
        elif ci[0] > 1.002 and faster >= math.ceil(rounds * 0.75):
            status = "PROVISIONAL_WIN"
        elif ci[1] < 0.998:
            status = "REJECT_REGRESSION"
        else:
            status = "RETEST_INCONCLUSIVE"
        results[name] = {
            "patches": VARIANTS[name],
            "rounds": rounds,
            "median_speedup": median,
            "geometric_mean_speedup": math.exp(statistics.mean(math.log(x) for x in ratios)),
            "mean_speedup": statistics.mean(ratios),
            "bootstrap_median_95pct_ci": ci,
            "faster_rounds": faster,
            "status": status,
            "observations": rows,
        }

    binary_hashes = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in engines.items()
    }
    payload = {
        "schema": "LV_SPEED_FACTORIAL_V1",
        "reference": "fbccfb6eb5cd335b1ce8fc5c5efad9e36be4e19d",
        "runner": {
            "image": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
            "arch": os.uname().machine,
        },
        "signatures": signatures,
        "binary_sha256": binary_hashes,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"signatures": signatures, "results": results}, indent=2))
    if results["control_b"]["status"] != "CALIBRATION_PASS":
        raise SystemExit("A/A calibration failed; all speed conclusions are invalid")


def parse_engines(items: list[str]) -> dict[str, str]:
    result = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or name not in VARIANTS:
            raise SystemExit(f"invalid engine mapping: {item}")
        result[name] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("variant", choices=VARIANTS)
    apply_parser.add_argument("root", type=Path)
    bench_parser = sub.add_parser("benchmark")
    bench_parser.add_argument("--engine", action="append", required=True)
    bench_parser.add_argument("--rounds", type=int, default=15)
    bench_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "apply":
        for patch_name in VARIANTS[args.variant]:
            PATCHES[patch_name](args.root)
        print(json.dumps({"variant": args.variant, "patches": VARIANTS[args.variant]}))
    else:
        benchmark(parse_engines(args.engine), args.rounds, args.output)


if __name__ == "__main__":
    main()

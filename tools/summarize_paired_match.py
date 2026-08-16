#!/usr/bin/env python3
"""Validate and summarize color-reversed match pairs without external statistics packages."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(probability * len(values)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--bootstrap", type=int, default=50000)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    games = source["games"]
    if not games or len(games) % 2:
        raise SystemExit("paired match must contain a positive even number of games")

    pair_scores = []
    for index in range(0, len(games), 2):
        first, second = games[index : index + 2]
        if first["fen"] != second["fen"]:
            raise SystemExit(f"opening mismatch in pair {index // 2}")
        if not first["a_white"] or second["a_white"]:
            raise SystemExit(f"color reversal mismatch in pair {index // 2}")
        pair_scores.append(float(first["score_a"]) + float(second["score_a"]))

    rng = random.Random(args.seed)
    boot = []
    for _ in range(args.bootstrap):
        sample = [pair_scores[rng.randrange(len(pair_scores))] for _ in pair_scores]
        boot.append(statistics.mean(sample) / 2.0)

    score = statistics.mean(pair_scores) / 2.0
    elo = None
    if 0.0 < score < 1.0:
        elo = 400.0 * math.log10(score / (1.0 - score))
    wdl = {
        "wins": sum(float(game["score_a"]) == 1.0 for game in games),
        "draws": sum(float(game["score_a"]) == 0.5 for game in games),
        "losses": sum(float(game["score_a"]) == 0.0 for game in games),
    }
    payload = {
        "schema": "LV_PAIRED_MATCH_SUMMARY_V1",
        "games": len(games),
        "pairs": len(pair_scores),
        "wdl": wdl,
        "score": score,
        "naive_elo": elo,
        "paired_bootstrap_score_95pct_ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "pair_score_histogram": dict(sorted(Counter(str(value) for value in pair_scores).items())),
        "pair_wins": sum(value > 1.0 for value in pair_scores),
        "pair_ties": sum(value == 1.0 for value in pair_scores),
        "pair_losses": sum(value < 1.0 for value in pair_scores),
        "terminations": dict(sorted(Counter(game["termination"] for game in games).items())),
        "resource": source["summary"]["resource"],
        "bootstrap": {"samples": args.bootstrap, "seed": args.seed},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

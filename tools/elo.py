"""Simple W/D/L Elo estimate for Leviathan smoke tests.

For serious acceptance use pentanomial/SPRT tooling. This script is intentionally
small and dependency-free for quick local summaries.
"""

from __future__ import annotations

import argparse
import math


def elo_from_score(score: float) -> float:
    eps = 1e-12
    score = min(1 - eps, max(eps, score))
    return 400.0 * math.log10(score / (1.0 - score))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wins", type=int)
    ap.add_argument("draws", type=int)
    ap.add_argument("losses", type=int)
    args = ap.parse_args()

    n = args.wins + args.draws + args.losses
    if n <= 0:
        raise SystemExit("no games")
    score = (args.wins + 0.5 * args.draws) / n
    print(f"games={n} W={args.wins} D={args.draws} L={args.losses}")
    print(f"score={score:.6f}")
    print(f"elo={elo_from_score(score):+.3f}")


if __name__ == "__main__":
    main()

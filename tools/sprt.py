"""Lightweight sequential screening SPRT for Leviathan W/D/L matches.

This is a screening gate, not a replacement for Stockfish/Fishtest-style
pentanomial testing. It conditions on the observed draw count and applies the
LLR to decisive games. Final strength claims still require the stronger paired
protocol documented in the project.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def p_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def llr(w: int, l: int, elo0: float, elo1: float) -> float:
    p0 = min(1 - 1e-12, max(1e-12, p_from_elo(elo0)))
    p1 = min(1 - 1e-12, max(1e-12, p_from_elo(elo1)))
    return w * math.log(p1 / p0) + l * math.log((1 - p1) / (1 - p0))


def verdict(value: float, alpha: float, beta: float) -> tuple[str, float, float]:
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)
    if value >= upper:
        return "ACCEPT_H1", lower, upper
    if value <= lower:
        return "ACCEPT_H0", lower, upper
    return "CONTINUE", lower, upper


def extract(path: Path) -> tuple[int, int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data["summary"] if "summary" in data else data
    return int(s["wins_a"]), int(s["draws"]), int(s["losses_a"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=Path)
    ap.add_argument("--wins", type=int)
    ap.add_argument("--draws", type=int, default=0)
    ap.add_argument("--losses", type=int)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    args = ap.parse_args()
    if args.match:
        w, d, l = extract(args.match)
    else:
        if args.wins is None or args.losses is None:
            raise SystemExit("provide --match or --wins/--losses")
        w, d, l = args.wins, args.draws, args.losses
    if min(w, d, l) < 0:
        raise SystemExit("negative result count")
    if not (0 < args.alpha < 1 and 0 < args.beta < 1):
        raise SystemExit("alpha/beta must be in (0,1)")

    value = llr(w, l, args.elo0, args.elo1)
    state, lower, upper = verdict(value, args.alpha, args.beta)
    n = w + d + l
    score = (w + 0.5 * d) / n if n else 0.5
    out = {
        "games": n,
        "wins": w,
        "draws": d,
        "losses": l,
        "score": score,
        "elo0": args.elo0,
        "elo1": args.elo1,
        "llr": value,
        "lower": lower,
        "upper": upper,
        "verdict": state,
        "warning": "screening WDL SPRT; final promotion requires paired pentanomial/LTC evidence",
    }
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()

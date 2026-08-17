#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", nargs=2, metavar=("HORIZON", "PATH"), required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    seen = set()
    for horizon_text, path_text in args.part:
        horizon = int(horizon_text)
        payload = json.loads(Path(path_text).read_text())
        if not isinstance(payload, list):
            raise SystemExit(f"{path_text}: expected JSON list")
        for item in payload:
            fen = item.get("final_fen") or item.get("fen")
            if not fen:
                raise SystemExit(f"{path_text}: row missing FEN")
            if fen in seen:
                continue
            seen.add(fen)
            rows.append({
                "name": item.get("name"),
                "horizon": horizon,
                "fen": fen,
                "generator_score_cp_white": item.get("stockfish_depth10_cp_white"),
            })

    Path(args.output).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"positions": len(rows), "horizons": sorted({r['horizon'] for r in rows})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

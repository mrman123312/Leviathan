#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUSE = ROOT / "tools" / "fuse_teacher_views.py"


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(FUSE), *args], text=True, capture_output=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        lc0 = d / "lc0.jsonl"
        sf = d / "sf.jsonl"
        out = d / "out.jsonl"
        fen = "8/8/8/8/8/8/4K3/6k1 w - - 0 1"
        write_jsonl(lc0, [{
            "position_id": "p1",
            "source_split_group": "game-7",
            "fen": fen,
            "lc0_native": {"policy": [0.6, 0.4], "winner_wdl": [0.0, 1.0, 0.0]},
        }])
        write_jsonl(sf, [{
            "position_id": "p1",
            "source_split_group": "game-7",
            "fen": fen,
            "stockfish_teacher": {"bestmove": "e2e3", "score": {"type": "cp", "value": 0}},
        }])
        p = run("--lc0", str(lc0), "--stockfish", str(sf), "--require-complete-overlap", "--output", str(out))
        assert p.returncode == 0, p.stderr
        records = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(records) == 1
        r = records[0]
        assert r["source"]["raw_sample_count"] == 1
        assert r["derived"]["dual_view"] is True
        assert r["views"]["lc0_native"]["policy"] == [0.6, 0.4]
        assert r["views"]["stockfish_teacher"]["bestmove"] == "e2e3"
        assert r["source_split_group"] == "game-7"

        # Same position with conflicting game/shard ancestry must be rejected.
        write_jsonl(sf, [{
            "position_id": "p1",
            "source_split_group": "game-8",
            "fen": fen,
            "stockfish_teacher": {"bestmove": "e2e3"},
        }])
        p = run("--lc0", str(lc0), "--stockfish", str(sf), "--output", str(out))
        assert p.returncode != 0
        assert "split-group ancestry mismatch" in (p.stderr + p.stdout)

        # Duplicate source positions must not silently become extra samples.
        write_jsonl(lc0, [
            {"position_id": "dup", "source_split_group": "g", "fen": fen},
            {"position_id": "dup", "source_split_group": "g", "fen": fen},
        ])
        p = run("--lc0", str(lc0), "--stockfish", str(sf), "--output", str(out))
        assert p.returncode != 0
        assert "duplicate position key" in (p.stderr + p.stdout)

    print("training tool tests OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

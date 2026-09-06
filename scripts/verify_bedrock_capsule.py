#!/usr/bin/env python3
"""Verify that the published source matches the recorded local-test source bytes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    path = ROOT / 'evidence/bedrock-v3/local-execution.json'
    if not path.exists():
        path = ROOT / 'evidence/bedrock/local-execution.json'
    manifest = json.loads(path.read_text())
    mismatches = []
    for name, expected in manifest['source_files_sha256'].items():
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'missing'
        if actual != expected:
            mismatches.append((name, expected, actual))
    if mismatches:
        for name, expected, actual in mismatches:
            print(f'MISMATCH {name}: expected={expected} actual={actual}')
        return 1
    print(f"Verified {len(manifest['source_files_sha256'])} tested source hashes.")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

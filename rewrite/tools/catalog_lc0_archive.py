#!/usr/bin/env python3
"""Snapshot the official Lc0 training-data archive into a reproducible shard catalog."""
from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_INDEX = "https://storage.lczero.org/files/training_data/"


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        d = dict(attrs)
        href = d.get("href")
        if href and href.lower().endswith(".tar"):
            self.links.append(href)


def fetch_bytes(url: str) -> tuple[bytes, dict[str, str | int | None]]:
    req = urllib.request.Request(url, headers={"User-Agent": "Leviathan-lc0-catalog/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
        return data, {
            "status": getattr(r, "status", None),
            "etag": r.headers.get("ETag"),
            "last_modified": r.headers.get("Last-Modified"),
            "content_type": r.headers.get("Content-Type"),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-url", default=DEFAULT_INDEX)
    ap.add_argument("--output", required=True)
    ap.add_argument("--include-head-sizes", action="store_true",
                    help="HEAD every shard to record Content-Length; expensive on very large indexes")
    ap.add_argument("--max-heads", type=int, default=0,
                    help="0 means all when --include-head-sizes is enabled")
    args = ap.parse_args()

    raw, metadata = fetch_bytes(args.index_url)
    parser = LinkParser()
    parser.feed(raw.decode("utf-8", errors="replace"))
    urls = sorted({urllib.parse.urljoin(args.index_url, href) for href in parser.links})
    if not urls:
        raise SystemExit("official Lc0 archive index yielded no .tar shards")

    shards = []
    total_known_bytes = 0
    head_limit = len(urls) if args.max_heads == 0 else min(len(urls), args.max_heads)
    for i, url in enumerate(urls):
        entry = {"url": url, "name": Path(urllib.parse.urlparse(url).path).name, "bytes": None}
        if args.include_head_sizes and i < head_limit:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Leviathan-lc0-catalog/1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                length = r.headers.get("Content-Length")
                if length:
                    entry["bytes"] = int(length)
                    total_known_bytes += int(length)
                entry["etag"] = r.headers.get("ETag")
                entry["last_modified"] = r.headers.get("Last-Modified")
        shards.append(entry)

    out = {
        "schema_version": 1,
        "source": args.index_url,
        "source_index_sha256": hashlib.sha256(raw).hexdigest(),
        "source_http_metadata": metadata,
        "shard_count": len(shards),
        "known_size_count": sum(1 for x in shards if x["bytes"] is not None),
        "known_total_bytes": total_known_bytes,
        "shards": shards,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("source_index_sha256", "shard_count", "known_size_count", "known_total_bytes")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

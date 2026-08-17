# Lc0 → Stockfish NNUE Training-Data Bridge

## Why this bridge exists

Stockfish's modern NNUE training lineage uses Lc0 evaluation data. Leviathan therefore needs a reproducible way to preserve **Lc0-native supervision** while also creating a **Stockfish-compatible training/teacher view** of selected positions.

The bridge must not destroy the original Lc0 policy/value/search metadata and must not count the converted record as a second independent raw sample.

## Pinned components

- Lc0 training-data parser/tooling: `LeelaChessZero/lczero-training@7c5d756ea6bb3531fb14a9b4df231577b1aa1081`
- preferred current Lc0 rescorer: `LeelaChessZero/lc0@d8ce48258c39d331c119f8c8729374ceb3df8409` (GPL-3.0-or-later)
- historical rescorer lineage: `Tilps/lc0@a69c7fb1f2aefce9a8f22e6fa867e466c1d3687e` (GPL-3.0)
- Stockfish training-data converter: `official-stockfish/Stockfish@9a4c7cf4e311f8d9526b79295b80c4d0464c07cf` (`tools` lineage, GPL-3.0)
- NNUE trainer/reader/writer: `official-stockfish/nnue-pytorch@9f72946529c4187d3679014036cd22c3be419716` (GPL-3.0)

The unlicensed `linrock/lc0-data-converter` repository is used only as a methodology reference. Its scripts are **not** imported. It documents a practical sequence built from the GPL tools above.

## Native parse path

The official `lczero-training` repository builds a `dump_chunk` executable. Leviathan uses this as the first independent proof that a downloaded archive contains valid native training records before any transformation occurs.

A real hash-locked shard was decoded successfully by the pinned official dumper. The sampled historical record uses `INPUT_CLASSICAL_112_PLANE`; the native representation contains the 1858-way policy and the value/moves-left fields supported by that data generation. Newer v6-only fields must only be used when actually present rather than synthesized into older records.

The native source view is retained rather than immediately crushed into Stockfish centipawns.

## Data path

```text
Lc0 ODbL/DbCL tar shard
        │
        ├──────────────► native Lc0 view
        │                policy / WDL / best-Q / moves-left / search metadata
        │
        ▼
pinned official Lc0 rescorer
        │
        ▼
Stockfish NNUE .plain
        │
        ├─ filter only with a versioned, recorded rule set
        │
        ▼
pinned Stockfish `convert ... validate`
        │
        ▼
.binpack
        │
        ▼
pinned nnue-pytorch loader/trainer
```

The original source identity follows the record through every transformation.

## Rescorer dependency discovered during integration

The current official Lc0 rescorer is not a generic format-only converter. Its `RunRescorer()` startup contract requires Syzygy initialization to succeed with at least 3-piece coverage before it processes files, even if NNUE `.plain` output is the desired artifact.

Leviathan did **not** bypass or patch out this upstream requirement. The minimal ten-file 3-piece WDL/DTZ dependency is pinned in `SYZYGY3_LOCK.json` with SHA-256 hashes; the successful bridge used 5 WDL + 5 DTZ files totaling 25,824 bytes.

This is separate from native Lc0 parsing, which does not require tablebases.

## Reference conversion contract

The pinned Stockfish tools documentation defines:

```text
stockfish convert from_path to_path [append] [validate]
```

and supports `.plain`, `.bin`, and `.binpack` as input/output formats. Leviathan uses `validate` on newly converted shards.

The historical public conversion methodology uses an Lc0 rescorer to export NNUE `.plain` with best-move and best-score information, optionally tablebase-rescoring/deblundering, then filters unusable records and asks Stockfish to convert/validate the result.

## Proven real-data smoke

GitHub Actions run `31989106200` exercised the complete bridge on real Lc0 data rather than synthetic fixtures.

### Source

- parent shard: `training-run3--20200713-0822.tar`
- parent bytes: `15,452,160`
- parent SHA-256: `dfaf79680b92e317a05b343f3c11c52c8ca98957a48c3fcc86b992e3762b1c7a`
- parent archive contains 948 gzipped training chunks
- bounded bridge sample used three unmodified chunk byte streams:
  - `training.686023.gz` — 15,963 bytes
  - `training.686057.gz` — 10,889 bytes
  - `training.685571.gz` — 13,043 bytes

The chunks were flattened only at the filesystem-path layer because the official rescorer enumerates files directly in its input directory. Their compressed record bytes were not rewritten before parsing.

### Official Lc0 export

Pinned Lc0 rescorer `d8ce48258c39d331c119f8c8729374ceb3df8409` reported:

- 3 games processed
- 219 positions processed
- 0 tablebase outcome rescores in this tiny sample
- 0 deblunder changes
- original outcomes: 0 losses / 1 draw / 2 wins
- post-processing outcomes: 0 losses / 1 draw / 2 wins

It produced `lc0-stockfish.plain`:

- 1,296 text lines
- 22,618 bytes
- SHA-256 `e724a48ecf560ddd71c99145476ebf52a5fbf71dc1a715b0b96a32a0e2f950f8`

### Stockfish conversion/validation

Pinned Stockfish training-tools commit `9a4c7cf4e311f8d9526b79295b80c4d0464c07cf` then ran the supported `convert ... validate` path.

Result:

- **216 validated training positions converted**
- output `lc0-stockfish.binpack`: 544 bytes
- SHA-256 `9a7990a9bf65ffd1216018c107f2fbaf1d05abc06cc60cd9c6bc51590b266ef8`

The 219 rescorer positions and 216 emitted Stockfish training entries are intentionally recorded as different stage counts rather than silently forced to match. The bridge acceptance criterion is the converter's own successful validation of the emitted training records, not a guessed one-to-one frame mapping.

Bridge artifact SHA-256: `1ed35ef9f9d3ad76650e0412ef5e7c3ba14db8cc3cab3fd287915c5f9ae36afb`.

## Leviathan modifications to the historical workflow

We should **not** blindly reproduce every old filter/deblunder choice. Those are experimental policies, not immutable truths.

Each production converted shard must record:

- source tar SHA-256;
- source archive/chunk/game or sequence identity where available;
- rescorer commit;
- Syzygy set/hash;
- deblunder configuration;
- filter implementation + commit + parameters;
- Stockfish converter commit;
- output binpack SHA-256;
- count before/after filtering;
- reason for each exclusion class.

Then run A/B tests on filtering/deblunder policy rather than treating one historical pipeline as canon.

## Information-preserving rule

`.binpack` is a **training compatibility view**, not Leviathan's canonical data representation.

Never discard the richer native Lc0 view after conversion. The dual-view corpus should retain both:

```text
source_position
├── lc0_native
│   ├── policy[1858]
│   ├── winner WDL
│   ├── best-Q WDL
│   ├── plies-left
│   └── other native search metadata when present
└── stockfish_view
    ├── best move / MultiPV
    ├── CP or mate score
    ├── WDL when requested
    ├── raw/static NNUE when requested
    └── engine/network/budget provenance
```

## Compute policy

Do not convert/relabel the full public Lc0 archive by default.

Start with bounded shards and prioritize positions where:

- Lc0 policy and Stockfish best move disagree;
- Lc0 Q/WDL and calibrated Stockfish value disagree;
- Leviathan chooses a third move;
- Stockfish's best move changes late with depth;
- tactical volatility or oracle regret is high;
- Leviathan has previously failed;
- the position is underrepresented in the existing corpus.

This makes Stockfish compute an **information amplifier** on Lc0 data rather than an indiscriminate relabeling bill.

## Acceptance gates for a converted shard

A shard is eligible for training only if:

1. source hash matches `DATA_SOURCE_LOCKS.json`;
2. extraction is deterministic;
3. native Lc0 parse succeeds;
4. required Syzygy files are pinned and verified;
5. conversion completes with Stockfish `validate`;
6. output hash/count/provenance are recorded;
7. train/validation/test split group was assigned from source ancestry **before** relabeling;
8. no source group appears in more than one split;
9. a sample decode confirms the resulting positions agree with the native source position representation;
10. license notices/ODbL attribution are preserved.

Only after those gates should the shard enter model experiments.

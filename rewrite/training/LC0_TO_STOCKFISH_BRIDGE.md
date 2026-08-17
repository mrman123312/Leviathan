# Lc0 → Stockfish NNUE Training-Data Bridge

## Why this bridge exists

Stockfish's modern NNUE training lineage uses Lc0 evaluation data. Leviathan therefore needs a reproducible way to preserve **Lc0-native supervision** while also creating a **Stockfish-compatible training/teacher view** of selected positions.

The bridge must not destroy the original Lc0 policy/value/search metadata and must not count the converted record as a second independent raw sample.

## Pinned components

- Lc0 training-data parser/tooling: `LeelaChessZero/lczero-training@7c5d756ea6bb3531fb14a9b4df231577b1aa1081`
- Lc0 tablebase rescorer: `Tilps/lc0@a69c7fb1f2aefce9a8f22e6fa867e466c1d3687e` (`rescore_tb` lineage, GPL-3.0)
- Stockfish training-data converter: `official-stockfish/Stockfish@9a4c7cf4e311f8d9526b79295b80c4d0464c07cf` (`tools` lineage, GPL-3.0)
- NNUE trainer/reader/writer: `official-stockfish/nnue-pytorch@9f72946529c4187d3679014036cd22c3be419716` (GPL-3.0)

The unlicensed `linrock/lc0-data-converter` repository is used only as a methodology reference. Its scripts are **not** imported. It documents a practical sequence built from the GPL tools above.

## Data path

```text
Lc0 ODbL/DbCL tar shard
        │
        ├──────────────► native Lc0 view
        │                policy / WDL / best-Q / moves-left / search metadata
        │
        ▼
pinned Lc0 rescorer
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

## Reference conversion contract

The pinned Stockfish tools documentation defines:

```text
stockfish convert from_path to_path [append] [validate]
```

and supports `.plain`, `.bin`, and `.binpack` as input/output formats. Leviathan should always use `validate` on new converted shards.

The public conversion methodology uses Lc0's rescorer to export NNUE `.plain` with best-move and best-score information, optionally tablebase-rescoring/deblundering, then filters unusable records and asks Stockfish to convert/validate the result.

## Leviathan modifications to the historical workflow

We should **not** blindly reproduce every old filter/deblunder choice. Those are experimental policies, not immutable truths.

Each converted shard must record:

- source tar SHA-256;
- source archive/chunk/game or sequence identity where available;
- rescorer commit;
- Syzygy set/hash or explicit `none`;
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
│   └── other native search metadata
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
4. conversion completes with Stockfish `validate`;
5. output hash/count/provenance are recorded;
6. train/validation/test split group was assigned from source ancestry **before** relabeling;
7. no source group appears in more than one split;
8. a sample decode confirms the resulting positions agree with the native source position representation;
9. license notices/ODbL attribution are preserved.

Only after those gates should the shard enter model experiments.

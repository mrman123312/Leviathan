# Leviathan Rewrite

A greenfield chess-engine core plus a controlled inheritance system for the Leviathan project.

The goal is **architectural independence without throwing away open-source chess knowledge**. Stockfish and other engines are donors/oracles, not the skeleton of the rewrite.

## Build

From the repository root:

```bash
cmake -S rewrite -B rewrite/build -DCMAKE_BUILD_TYPE=Release
cmake --build rewrite/build -j
ctest --test-dir rewrite/build --output-on-failure
```

## Smoke test

```text
uci
isready
position startpos
perft 4
go depth 4
quit
```

## Donor ecology

- `donors/DONOR_REGISTRY.json` — verified engines/assets and direct-code policy.
- `docs/DONOR_MATRIX.md` — what each donor is allowed to contribute and how it must be tested.
- `imports/` — the only approved location for committed copied/adapted donor artifacts.
- `tools/audit_donors.py` — fail-closed provenance/license audit run by CTest.
- `tools/donorctl.py` — list/show donors and scaffold provenance records.
- `donors/locks/` + `tools/materialize_source.py` — pin exact external source blobs without vendoring them prematurely.
- `models/MODEL_REGISTRY.json` + `tools/fetch_models.py` — pin/fetch pretrained networks and verify their SHA-256 identity.

Examples:

```bash
python3 rewrite/tools/donorctl.py list
python3 rewrite/tools/audit_donors.py rewrite/donors/DONOR_REGISTRY.json rewrite/imports
python3 rewrite/tools/materialize_source.py rewrite/donors/locks/fathom.json
python3 rewrite/tools/fetch_models.py stockfish18-big
```

## Current architecture

- independent FEN/rules/legal move generation;
- UCI shell and perft/self-tests;
- simple alpha-beta/quiescence control search;
- path-sensitive proof identity separate from board identity;
- TT fields reserved for evidence/debt;
- structured evaluation (`mean`, `uncertainty`, `volatility`, `provenance`);
- pluggable `Evaluator` interface so pretrained/donor evaluators can be swapped without rewriting search.

The current alpha-beta/evaluation are **controls**, not the intended final Leviathan ontology.

## Immediate inheritance strategy

1. Keep the rewrite rules/search contracts independent.
2. Use frozen pretrained models to recover evaluation strength without retraining from zero.
3. Reconstruct a strong conventional control engine from isolated donor mechanisms, not a monolithic Stockfish port.
4. A/B/A+B test every search import with fixed-node, fixed-time, tactical/deep-oracle and paired-game gates.
5. Once conventional strength is recovered, attack the shared assumptions of the donor family with candidate-set/proof-budget search, uncertainty-directed computation and native evidence semantics.

Known limitations: no integrated NNUE runtime yet, no active Syzygy adapter yet, no SMP, no advanced time manager. The donor/model infrastructure for those components is now explicit and reproducible rather than implicit.

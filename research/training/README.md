# Leviathan Bootstrap Training

Leviathan does not train from zero unless an experiment specifically requires it. The bootstrap pipeline reuses legally available public chess knowledge while preserving source identity.

## The Lc0 + Stockfish rule

Modern Stockfish NNUE training uses Lc0 evaluation data. Therefore Leviathan does **not** count an Lc0 source position and a Stockfish annotation of that same position as two independent samples.

Instead:

1. Lc0 supplies the native source record: policy, game/result value, search-Q information, moves-left and self-play ancestry where available.
2. Stockfish supplies a derived second view: CP/mate/WDL, best move/MultiPV, static NNUE or searched evaluation, exact engine/network identity and budget.
3. Leviathan learns from both views and especially from their disagreement.

See `DUAL_VIEW_SCHEMA.md`.

## Acquisition

Approved fetch targets are pinned in `DATA_SOURCE_LOCKS.json`.

```bash
python3 rewrite/tools/fetch_training_data.py --list
python3 rewrite/tools/fetch_training_data.py stockfish-nnue-pytorch-small-binpack
python3 rewrite/tools/fetch_training_data.py lc0-hourly-sample-2020-07-11-2017 --probe
```

The fetcher is bounded by a per-source byte ceiling, records SHA-256 and HTTP provenance sidecars, and verifies immutable Git-blob identity where available. Bulk archives live under `rewrite/training/materialized/` and are ignored by git.

The repository intentionally stores **manifests and reproducible acquisition**, not terabytes of training data.

## Native Lc0 signal

The official Lc0 training tuple exposes five core targets:

- board/history planes;
- 1858-way policy probabilities;
- game-result WDL;
- searched best-Q WDL;
- plies-left.

Modern v6 records carry additional search/value metadata. Preserve these targets instead of converting them all to centipawns.

## Stockfish signal

`official-stockfish/nnue-pytorch` is pinned as the Stockfish-compatible trainer/reader/writer and equivalence oracle. Its small repository binpack fixture is used only to prove that our acquisition and future loader integration work.

For production data, selected Lc0 positions should be relabeled with a pinned Stockfish binary/network and, when useful, converted into Stockfish-compatible binpack. Do **not** spend compute relabeling the full public archive before disagreement/frontier sampling demonstrates that it is useful.

## Record model

Each normalized position retains source ancestry and one or more separate views. Legacy JSONL may contain:

- `fen`: exact position
- `source`: corpus and source-batch identity
- `license`: dataset license
- `teachers`: per-engine best move, score, depth/time and engine identity
- `consensus_cp`: mean teacher score only when centipawn labels are actually comparable
- `disagreement_cp`: teacher score dispersion
- `candidate_moves`: union of teacher best moves / future MultiPV candidates
- `tags`: failure, disagreement, tactical-volatility, selfplay, frontier, etc.

The v2 dual-view schema additionally requires a `source_split_group` tied to the original game/shard so alternate labels of the same source sample cannot leak across train/validation/test.

Never collapse donor labels into one scalar before retaining the original labels.

## Bootstrap order

1. Use frozen pretrained networks as evaluation controls; do not retrain obvious chess knowledge.
2. Materialize selected Lc0 source shards with ODbL/DbCL provenance.
3. Normalize Lc0 native targets without losing archive/chunk/game identity.
4. Stockfish-label only selected positions, beginning with disagreement/frontier candidates.
5. Fuse the views without changing the raw sample count.
6. Oversample teacher disagreement, Leviathan failures, late best-move changes and high search regret.
7. Add Leviathan self-play/frontier positions as the engine becomes competitive.
8. Train/fine-tune only when expected information gain justifies compute.

The goal is eventually to shift from inherited foundation data toward Leviathan-native failure/frontier data without paying to rediscover chess from random weights.

## Multi-teacher labeler

`tools/label_corpus.py` takes newline-delimited FENs and one or more UCI engines and writes normalized JSONL. Example:

```bash
python3 rewrite/tools/label_corpus.py \
  --fen-file rewrite/training/bootstrap_seed.fen \
  --teacher stockfish=/path/to/stockfish \
  --teacher leviathan=./rewrite/build/leviathan \
  --movetime 250 \
  --output labels.jsonl
```

The labeler records individual teacher outputs first and computes disagreement only afterward. Mate scores are retained as mate labels rather than silently converted into ordinary centipawns.

## Dual-view fusion

Once an Lc0-normalized JSONL and Stockfish teacher JSONL share position IDs/FENs:

```bash
python3 rewrite/tools/fuse_teacher_views.py \
  --lc0 lc0-native.jsonl \
  --stockfish stockfish-labels.jsonl \
  --require-complete-overlap \
  --output dual-view.jsonl
```

The fusion tool rejects duplicate position keys and conflicting split ancestry. A position with two teacher views still contributes exactly one raw source sample.

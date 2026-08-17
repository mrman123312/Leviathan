# Leviathan Dual-View Training Schema

## Goal

Do **not** train as if "Lc0 data" and "Stockfish data" are two independent giant corpora when Stockfish's modern NNUE training explicitly uses Lc0 evaluation data. Leviathan keeps the original Lc0 record as the source sample and attaches Stockfish as a second teacher/view.

The unit of evidence is the **source chess position**, not the number of engines that have labeled it.

## Source view: Lc0 native

For every imported Lc0 position preserve as much of the upstream record as the selected parser exposes, including:

- source archive/shard/chunk/game identity;
- board/history planes or a lossless reconstructed position identity;
- 1858-way policy target;
- game result / WDL target;
- root/best Q and draw values when present;
- played/best action information when present;
- visits and policy divergence metadata when present;
- plies-left / moves-left target when present;
- upstream training-data format version.

The current upstream training tuple documents `planes`, `probs`, `winner`, `best_q`, and `plies_left`. Newer v6 records contain additional search/value metadata. Never silently reduce this native view to one centipawn scalar.

## Derived view: Stockfish

Selected source positions can then receive a Stockfish view containing:

- exact engine commit/version;
- NNUE network hash;
- search budget (depth/nodes/time);
- best move and optional MultiPV candidates;
- centipawn or mate score;
- optional WDL output;
- optional static/raw NNUE evaluation;
- nodes/depth reached.

This is a **derived annotation of the same source sample**, not a second independent example.

## Leviathan record

Recommended normalized shape:

```json
{
  "schema_version": 2,
  "position_id": "sha256:...",
  "source_split_group": "original-game-or-shard-id",
  "source": {
    "dataset_id": "lc0-public-selfplay-archive",
    "archive": "...tar",
    "chunk": "...",
    "game_id": "...",
    "license": "ODbL-1.0/DbCL-1.0"
  },
  "position": {
    "fen": "...",
    "history_identity": "..."
  },
  "views": {
    "lc0_native": {
      "policy": [],
      "winner_wdl": [],
      "best_q_wdl": [],
      "plies_left": null,
      "native": {}
    },
    "stockfish_teacher": {
      "engine": "Stockfish",
      "commit": "...",
      "network_sha256": "...",
      "budget": {},
      "bestmove": "...",
      "multipv": [],
      "score": {"type": "cp", "value": 0},
      "wdl": null
    },
    "leviathan_teacher": null
  },
  "derived": {
    "teacher_bestmove_disagreement": null,
    "calibrated_value_disagreement": null,
    "frontier_priority": null
  }
}
```

## Split discipline

Split **before** teacher relabeling using original game/shard identity. All views and duplicates of one source game stay in exactly one of train/validation/test.

Never randomly split individual positions from the same game across train and holdout. That leaks highly correlated states and inflates validation quality.

## Fusion rules

1. Keep Lc0 policy and value targets intact.
2. Keep Stockfish CP/mate/WDL targets intact.
3. Do not average raw Stockfish centipawns with Lc0 WDL/Q values directly.
4. If a shared scalar is needed, learn or freeze a calibration mapping first and version it.
5. Teacher disagreement is a feature/target, not noise to erase.
6. Deduplicate by canonical position/history identity and source ancestry.
7. Multiple teacher labels increase supervision on a sample; they do not increase the raw sample count.
8. Oversample high-information disagreements, Leviathan failures, late best-move changes, tactical volatility, and high oracle regret.

## Compute-saving training order

1. Use frozen Stockfish NNUE as a strong production evaluator control.
2. Use native Lc0 policy/value targets without recomputing them.
3. Stockfish-relabel only a selected high-information subset rather than all 100B+ candidate positions.
4. Train multi-head or distillation candidates on the dual view.
5. Add Leviathan self-play/failure/frontier data and increase its mixture weight as the engine improves.
6. Promote a model only on fresh source-group holdouts plus chess-strength tests.

## Suggested future heads

A Leviathan-native model can eventually expose separate heads/signals for:

- value/WDL;
- action policy or candidate prior;
- Stockfish-like tactical/static value;
- uncertainty;
- teacher disagreement;
- volatility / late-PV instability;
- moves-left / horizon character;
- provenance confidence.

This is intentionally richer than cloning either Stockfish NNUE or Lc0 alone.

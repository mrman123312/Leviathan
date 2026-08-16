# Leviathan Bootstrap Training

Leviathan does not train from zero unless an experiment specifically requires it. The bootstrap pipeline reuses legally available public chess knowledge while preserving source identity.

## Record model

Each normalized JSONL position may contain:

- `fen`: exact position
- `source`: corpus and source-batch identity
- `license`: dataset license
- `teachers`: per-engine best move, score, depth/time and engine identity
- `consensus_cp`: mean teacher score when centipawn labels are comparable
- `disagreement_cp`: teacher score dispersion
- `candidate_moves`: union of teacher best moves / future MultiPV candidates
- `tags`: failure, disagreement, tactical-volatility, selfplay, frontier, etc.

Never collapse donor labels into one scalar before retaining the original labels.

## Bootstrap order

1. Use frozen pretrained networks as evaluation controls; do not retrain obvious chess knowledge.
2. Normalize public donor positions without losing batch/license provenance.
3. Label selected positions with multiple independent teachers.
4. Oversample teacher disagreement, Leviathan failures, late best-move changes and high search regret.
5. Add Leviathan self-play/frontier positions as the engine becomes competitive.
6. Train/fine-tune only when the expected information gain justifies compute.

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

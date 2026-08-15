# Leviathan Immortal v1

Immortal v1 is the first implementation of the full RCE-derived Leviathan research organism. It does **not** claim solved chess or proven superiority to Stockfish. It makes the complete research architecture executable while preserving a parent-equivalent rollback path in the same binary.

## Trust ladder

Every organ starts with zero authority and must climb this ladder independently:

1. **Observe** — trace only; cannot alter search.
2. **Order** — may change move priority, never prune.
3. **Veto** — may make Stockfish *less* selective when predicted risk is high.
4. **Verify** — may request additional search/time/specialist work.
5. **Bidirectional control** — may also save compute when calibration and Elo justify it.
6. **Promotion** — only statistically positive descendants become lineage parents.

All new UCI options default to disabled/authority 0. A default Immortal binary must therefore retain the frozen Stockfish search signature.

## Playing organs

### P001 Policy

`src/leviathan_policy.h`

Tiny quantized 12→16→1 quiet-move prior. It changes ordering only. Format: `LVTP1`.

UCI:

- `Leviathan Policy`
- `Leviathan Policy File`
- `Leviathan Policy Weight`

### P003 MetaSearch

`src/leviathan_control.h`

A 1–4-head cheap linear ensemble predicts whether additional root computation is likely to matter. Head disagreement is retained as epistemic uncertainty. Format: `LVTM1 <features=8> <heads> <scale>`.

Authority 1 can only increase verification time. Authority 2 permits bounded time reduction/increase.

UCI:

- `Leviathan MetaSearch`
- `Leviathan Meta File`
- `Leviathan Meta Authority` (0–2)
- `Leviathan Meta Max Percent`

### P004/P005 Search-risk + calibrated LMR

`src/leviathan_control.h`

A 12-feature ensemble estimates selective-search regret. Format: `LVTR1 <features=12> <heads> <scale>`.

Authority 1 can only reduce Stockfish's reduction (search deeper). Authority 2 may add a small bounded extra reduction in exceptionally low-risk states.

UCI:

- `Leviathan Risk`
- `Leviathan Risk File`
- `Leviathan Risk Authority`
- `Leviathan Risk Threshold`
- `Leviathan Risk Veto`

### P008 Specialist router

The first live specialist is deliberately conservative: when the risk ensemble says a volatile checking/capture/PV move is unsafe to reduce, the router buys back additional reduction. This creates a real separately switchable specialist path without inserting a slow second search algorithm into every node.

`specialists/mate_prover.py` is the first independent proof-style specialist for offline/held-out experiments. It is a bounded AND/OR mate prover. It is not promoted into the live hot path until it proves marginal value.

UCI:

- `Leviathan Specialist`
- `Leviathan Specialist Veto`

### P009 Chess Atlas / PV skills

`src/leviathan_atlas.h`, `atlas/atlas_tool.py`

Atlas stores engine-native `(Position::key, Move::raw)` hints with confidence and provenance. Learned `episode`/`skill` entries and `exact` entries remain distinct. Even exact entries currently receive ordering priority only; alpha-beta still verifies them.

Format:

```text
LVTA1
<position_key_u64> <move_raw_u16> <bonus> <confidence_0_1000> <episode|skill|exact>
```

The builder writes a SHA-256 manifest so knowledge artifacts are content addressed.

UCI:

- `Leviathan Atlas`
- `Leviathan Atlas File`
- `Leviathan Atlas Weight`

### P007 Search DSL

`src/leviathan_dsl.h`, `foundry/foundry.py`

A tiny typed bytecode lets the offline Foundry propose bounded LMR changes without arbitrary runtime code generation.

Format `LVSD1`, maximum 32 instructions:

- `ADD value`
- `MULADD feature weight divisor`
- `IFGT feature threshold add`
- `IFLT feature threshold add`
- `CLAMP low high`

Authority 1 suppresses all positive/more-aggressive adjustments. Authority 2 permits bounded bidirectional changes.

UCI:

- `Leviathan Search DSL`
- `Leviathan Search DSL File`
- `Leviathan Search DSL Authority`
- `Leviathan Search DSL Weight`

## P002/P004 instrumentation

`src/leviathan_trace.h`

Optional sampled LMR JSONL records:

- native parent position key;
- raw move;
- exact 12 risk features used by the live controller;
- reduced value and depth;
- final/researched value and depth;
- regret label;
- Atlas-compatible bonus/confidence/provenance.

Tracing never changes search and is disabled by default.

UCI:

- `Leviathan Trace File`
- `Leviathan Trace Sample Permille`

## Training

### Root MetaSearch data

```bash
python trainer/generate_metasearch_dataset.py \
  --engine src/stockfish --fens data/fens.txt --out data/meta.jsonl

python trainer/train_control.py \
  --task meta --data data/meta.jsonl --out networks/meta.lvtm
```

### LMR risk data

Enable tracing during representative search/games:

```text
setoption name Leviathan Trace File value data/lmr.jsonl
setoption name Leviathan Trace Sample Permille value 10
```

Then:

```bash
python trainer/train_control.py \
  --task risk --data data/lmr.jsonl --out networks/risk.lvtr
```

Training uses independent bootstrap heads; runtime disagreement becomes an uncertainty signal.

## Search Foundry

Create a candidate:

```bash
python foundry/foundry.py new --out candidates/c001.lvsd --seed 8910
```

Mutate a parent:

```bash
python foundry/foundry.py mutate \
  --parent candidates/c001.lvsd --out candidates/c002.lvsd --seed 8911
```

Every candidate receives syntax/range checks before games. Candidate metadata stores SHA-256, parent, seed, range evidence, and provisional lifecycle state.

A candidate can never alter board state, legal moves, terminal evaluation, TT correctness, or NNUE values through the DSL.

## Chess Atlas

LMR traces can be directly harvested as episodic Atlas evidence because they use Stockfish-native keys/moves:

```bash
python atlas/atlas_tool.py build \
  --input data/lmr.jsonl --out atlas/current.lvta
```

The `.manifest.json` is part of the artifact provenance.

## Paired experiments

`tools/run_match.py` supports two different binaries or one identical binary with two option files. Openings are automatically replayed with reversed colors.

For strength tests use wall-clock compute:

```bash
python tools/run_match.py \
  --engine-a src/stockfish --engine-b src/stockfish \
  --options-a experiments/candidate.json \
  --options-b experiments/parent.json \
  --openings data/openings.fen --games 200 --movetime-ms 100 \
  --out results/candidate-v-parent.json
```

Fixed nodes are useful for search-efficiency diagnostics but are not the final equal-compute claim.

`tools/sprt.py` supplies a lightweight sequential W/D/L screening gate. It is intentionally not presented as a replacement for paired pentanomial/Fishtest-style STC/LTC testing.

## Practical lineage

```text
trace failures
   ↓
train model / mutate DSL / build Atlas candidate
   ↓
static invariants
   ↓
compile + parent bench
   ↓
held-out search-regret tests
   ↓
paired smoke games
   ↓
screening SPRT
   ↓
proper STC
   ↓
LTC / multi-hardware
   ↓
promote or retire
   ↓
new lineage parent
```

`experiments/registry.csv` remains the canonical human-readable lineage ledger. Model, Atlas, DSL, match, and trace artifacts should be content-addressed or accompanied by hashes.

## What “complete” means in v1

The architecture and code paths for P001–P010 exist in one research stack, with runtime controls, data generation, model training/export, persistent memory artifacts, a bounded operator language, candidate mutation/lifecycle tooling, specialist experimentation, paired matches, screening statistics, CI, rollback, and provenance.

It does **not** mean every organ has already earned Elo. There is deliberately no bundled non-neutral Meta/Risk/DSL/Atlas model because inventing unmeasured weights would violate the project's evidence law. The next generation is produced by running the supplied evidence pipeline, not by declaring theoretical machinery strong.

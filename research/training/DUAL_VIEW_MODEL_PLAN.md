# Leviathan Dual-View Model Campaign

## Objective

Exploit the two genuinely different supervisory signals present in the public ecosystem:

- **Lc0:** search policy, WDL/Q-style value, game outcome, moves-left/horizon information and self-play ancestry.
- **Stockfish:** fast NNUE/static value plus alpha-beta best-move/MultiPV/search labels.

Do not pretend correlated source positions are independent data. The model should learn from **agreement and disagreement between views**, not from duplicated sample counts.

## Architecture candidates

### A — Conservative: stronger value NNUE only

Fast sparse incremental value model. Train one value/WDL trunk from the dual-view corpus but expose only a search-leaf evaluation.

Purpose: cheapest falsifiable replacement for `leviathan-distilled-v1`.

### B — Architectural: shared trunk + cheap uncertainty head

One incremental sparse trunk with:

- WDL/value head;
- calibrated Stockfish-value head;
- uncertainty/disagreement head;
- optional moves-left head.

The uncertainty head becomes available to the search scheduler without computing a full policy distribution.

### C — Revolutionary: value core + conditional policy sidecar

Split inference by computational need:

```text
incremental value core  -> every required leaf
          |
          +-> uncertainty gate
                    |
                    +-> policy sidecar only at root/PV/frontier nodes
```

The expensive 1858-way Lc0-style action prior is **not** paid at every leaf. Candidate-set/proof-budget search asks for it only where value-of-information justifies the cost.

### D — Moonshot: multi-teacher evidence model

Predict not just position value but an evidence object:

```text
{ WDL, tactical/static value, action prior, uncertainty,
  teacher disagreement, volatility, moves-left, provenance confidence }
```

This directly populates Leviathan's structured `Evaluation` rather than reducing every learned fact to one centipawn scalar.

### E — Outside the lattice: learn compute allocation, not only chess value

Use teacher disagreement, late-PV changes and deep-oracle regret as labels for **where extra search is worth spending**. The learned component predicts marginal value-of-information rather than merely another evaluation score.

This challenges the shared assumption that the evaluator's only job is to estimate who is winning.

## Target preservation

Never directly average Stockfish centipawns with Lc0 WDL/Q.

Keep separate raw targets and version any calibration mapping. Recommended training targets:

- `lc0_policy`: cross-entropy / KL on native policy;
- `lc0_wdl`: categorical cross-entropy;
- `lc0_best_q`: categorical or calibrated expected-score loss;
- `stockfish_value`: Huber on a versioned expected-score calibration rather than raw unbounded CP when possible;
- `stockfish_bestmove/multipv`: candidate-ranking loss;
- `moves_left`: robust regression;
- `disagreement`: regression/classification derived only after raw views are retained;
- `frontier/regret`: search-derived target from fresh oracle experiments.

## Curriculum

### Stage 0 — production control

Run a frozen license-clean Stockfish NNUE backend in Leviathan. This separates search architecture from evaluator knowledge and gives the dual-view learner a serious baseline.

### Stage 1 — bounded donor data

Use exact-hash Lc0 shards and a bounded Stockfish-labeled subset. Verify source-group holdouts and loaders before scaling.

### Stage 2 — teacher disagreement mining

Increase sampling probability for positions where:

- Stockfish best move disagrees with Lc0 policy mode;
- calibrated Stockfish value disagrees with Lc0 Q/WDL;
- Leviathan chooses a third move;
- Stockfish's principal variation changes late;
- shallow/deep Stockfish disagree;
- tactical volatility is high;
- current Leviathan has high deep-oracle regret.

### Stage 3 — Leviathan-native frontier

Add failure positions, adversarial counterexamples, self-play and evidence-lattice frontier positions. Increase their mixture weight only after fresh-holdout gains survive.

## Required ablations

Use the **same source-group train/validation/test split** for all branches.

1. Stockfish-view only.
2. Lc0-native value only.
3. Lc0 native policy + value.
4. Dual view without disagreement target.
5. Dual view + disagreement/uncertainty.
6. Dual view + conditional policy sidecar.
7. Candidate/proof scheduler with uncertainty disabled.
8. Candidate/proof scheduler with uncertainty enabled.
9. Frozen Stockfish NNUE control.
10. Current `leviathan-distilled-v1` control.

No branch advances because its training loss looks better. It must win the relevant downstream gate.

## Acceptance ladder

### Data correctness

- exact source hashes;
- deterministic decode;
- no split ancestry leakage;
- raw target preservation;
- conversion validation when `.binpack` is used.

### Model correctness

- reproducible checkpoint/config/hash;
- independent inference cross-check;
- no NaN/overflow/quantization drift;
- calibration measured on fresh source groups.

### Chess-quality gates

- static held-out target metrics;
- best-move recall/top-k against deep oracle;
- tactical-suite regressions;
- equal-node deep-oracle regret;
- equal-time regret;
- A/B/A+B search integration;
- fixed-time games;
- SPRT only after cheaper gates survive.

### Runtime gates

For every added head measure:

- cycles/evaluation;
- accumulator update cost;
- cache footprint;
- nodes/second impact;
- completed-depth impact;
- Elo/regret gained per unit inference cost.

The policy sidecar is rejected if its extra information does not repay the depth it costs.

## First falsifiable experiment

The first model experiment after the data bridge is operational should be deliberately small:

- a bounded set of exact-hash Lc0 shards;
- grouped train/validation/test by source shard/game;
- Stockfish relabel only the selected training positions and matching validation positions;
- compare **Lc0-only vs Stockfish-only vs dual-view** under equal parameter count and equal training compute;
- keep `leviathan-distilled-v1` and frozen Stockfish NNUE as external controls;
- no self-play generation yet.

If dual-view does not beat the best single-view branch on fresh grouped holdouts **and** downstream search regret, scaling the data is not justified.

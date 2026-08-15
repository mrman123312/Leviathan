# Project Leviathan

Leviathan is an empirical Stockfish research fork. Its goal is to beat a contemporaneous Stockfish master under equal-compute conditions, not to accumulate theoretically attractive changes.

## Frozen parent

Initial parent: `5062aee519a1ba262d472d8ab139851ced56573e`.

The fork must be periodically rebased on current official Stockfish master before any public stronger-than-Stockfish claim.

## Current hypothesis: P001

The first hypothesis is that a tiny learned move prior can improve alpha-beta efficiency by ranking useful quiet moves earlier.

The first integration is intentionally narrow:

- Stockfish value NNUE remains untouched.
- TT move handling remains untouched.
- captures remain untouched.
- pruning remains untouched.
- extensions remain untouched.
- LMR remains untouched.
- only quiet-move ordering receives a bounded policy bonus.

This isolation makes the experiment causal: if P001 gains Elo, the likely mechanism is improved ordering rather than a bundled search rewrite.

## Runtime controls

P001 is controlled with environment variables so the same compiled binary can run parent-equivalent and policy-enabled trials.

- `LEVIATHAN_POLICY=1` enables the policy path.
- `LEVIATHAN_POLICY_FILE=/path/to/model.lvtp` selects the model.
- `LEVIATHAN_POLICY_WEIGHT=100` scales the bounded ordering bonus. Range is clamped to 0..400.

If no valid policy model is loaded, the policy contribution is exactly zero.

## Model

Current engine model: quantized `12 -> 16 -> 1` MLP.

The feature vector is deliberately move-local and cheap:

1. source file
2. source relative rank
3. target file
4. target relative rank
5. file displacement
6. rank displacement
7. moving piece type
8. center-improvement delta
9. gives-check flag
10. leaves an enemy-pawn-attacked square
11. enters an enemy-pawn-attacked square
12. forward-progress signal

The model uses integer weights, bounded ReLU activations, and a bounded final contribution. The design objective is not maximum offline accuracy; it is positive net Elo after inference cost.

## Model format

Text format `LVTP1`:

1. magic, feature count, hidden size
2. 16 rows of 12 signed int8 hidden weights
3. 16 signed int16 hidden biases
4. 16 signed int8 output weights
5. one signed int32 output bias

Malformed or dimensionally incompatible files fail closed and produce no policy bonus.

## Training loop

1. Build the parent engine.
2. Prepare a diverse FEN bank.
3. Run `trainer/generate_dataset.py` with a fixed teacher-node budget.
4. Split by complete `game_id` with `trainer/split_dataset.py`.
5. Train with `trainer/train.py`.
6. Export with `trainer/export.py`.
7. Run policy-weight sweeps using the same engine binary.
8. Measure NPS and cutoff/best-move ordering metrics.
9. Run paired STC games.
10. Promote only promising configurations to SPRT/LTC validation.

## Acceptance ladder

A behavior-changing experiment must pass:

1. compilation and syntax tests
2. UCI handshake
3. deterministic bench/correctness checks
4. NPS measurement
5. short paired smoke match
6. statistically serious progression/regression test
7. long-time-control confirmation when promising

No patch earns a strength claim from offline model accuracy alone.

## Next experiments

If P001 produces a credible positive result:

- P002: policy-aware LMR, bounded and tested marginally against P001.
- P003: cutoff-utility target rather than pure teacher evaluation.
- P004: policy inference optimization/shared feature reuse.

If P001 fails after reasonable model/weight experiments:

- pivot to beta-cutoff prediction,
- evaluation uncertainty,
- or shared value-policy features.

## Claim standard

Leviathan is stronger than Stockfish only after a controlled match against a contemporaneous Stockfish master with equal hardware, threads, hash, tablebase access, time control, paired openings, reversed colors, and adequate statistical evidence.

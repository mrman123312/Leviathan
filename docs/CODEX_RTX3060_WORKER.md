# Leviathan RTX 3060 Codex Worker Mission

## Role
You are Worker B in a two-worker Leviathan research lab. This branch is deliberately isolated from the protected CPU research branches.

- Remote branch: `leviathan/gpu-worker-3060`
- Protected branches: do not modify, force-push, merge into, or rewrite `leviathan/fundamentals-ultra-p01-qfrontier`, `leviathan/fundamentals-ultra-lab`, `leviathan/strength-v7-persistent-proof`, `leviathan/rewrite-v0-greenfield`, or `leviathan/rewrite-v1-donor-ecology`.
- Do not merge anything. Push only to `leviathan/gpu-worker-3060` unless the human explicitly changes this instruction.
- Do not launch large chess match campaigns. Default competitive game budget is ZERO. If every offline gate passes, a final micro-screen may use at most 40 games total, and must not automatically escalate.

## Research objective
Exploit Stockfish finite-compute weaknesses rather than chase folklore. Build a GPU-assisted system that predicts when match-budget Stockfish is likely to be wrong relative to much deeper Stockfish, then use that signal to allocate CPU search more intelligently.

The GPU is not a replacement for the CPU alpha-beta engine. The intended high-risk/high-reward architecture is:

`CPU search + GPU finite-compute risk/alternative model`

The GPU should operate at root / near-root / selected frontier batches where batching can amortize latency. Never insert a blocking GPU round-trip into every search node unless measurement proves it wins end-to-end.

## Current protected evidence
Treat these as historical facts, not targets to massage:

- P01 baseline is roughly fixed-time parity with current pinned Stockfish: 50.25% in the completed 400-game replication.
- Full P01 + ProbCut paid evidence had a 53.5% equal-node fresh 100-game result and improved deep-oracle regret versus P01, but a 40-game fixed-time micro-screen was 48.75% and inconclusive.
- Full ProbCut costs about 1.47% per-node throughput versus P01 in calibrated divergent-search timing.
- Order-only ProbCut looked promising on development counterexamples but failed prospective transfer: on 24 fresh positions it had only 2 root disagreements, 0 cp rescue / 4 cp harm, and median throughput ratio about 0.9892 versus P01. Do not revive it without a genuinely new causal reason.
- Lc0 raw policy-head imitation failed prospectively on 240 fresh positions: 104 disagreements, policy better 26 vs P01 better 65, 286 cp rescue vs 1767 cp harm. Do not build the project around raw Lc0 policy imitation.
- Caissa history retention and cutoff-aware NMP transfers were rejected.

## Mission 0 — hardware and environment preflight
Run and save exact outputs to `local_results/gpu3060/preflight/`:

1. `nvidia-smi`
2. GPU model, driver, CUDA runtime, VRAM total/free.
3. Python version.
4. PyTorch version and `torch.cuda.is_available()`.
5. `torch.cuda.get_device_name(0)`.
6. CPU model, logical/physical cores, RAM.
7. Git commit and dirty state.
8. Confirm Stockfish/P01 can build and run on CPU.

If PyTorch CUDA is unavailable, fix the local environment before model work. Do not change engine code merely to compensate for a broken CUDA setup.

## Mission 1 — build the Stockfish finite-compute error miner
Create a deterministic, restartable position miner. It should compare:

- `SF_fast`: exact intended match-style CPU budget.
- `SF_deep`: much larger CPU node budget, initially 16x-32x for development. 64x is optional only after throughput is measured.

Record a position as a candidate error when all are true:

1. `bestmove_fast != bestmove_deep`, and
2. the deep-oracle score gap between the fast move and deep move exceeds a configurable threshold (start around 20-30 cp), and
3. the result is stable enough under one verification repeat or a stronger oracle budget to avoid obvious noise.

Do NOT begin with millions of positions. Use an information-gain ladder:

- smoke: 50 positions
- development: 500 positions
- prospective holdout: 500 fresh positions
- only then consider 2k-10k if the error rate and model signal justify it

Persist every row so interrupted runs resume rather than restart.

### Required features per position
At minimum collect:

- FEN / position key
- piece count
- pawn count and pawn structure summary
- material imbalance
- side to move
- legal move count
- capture count
- checking-move count if cheap
- static NNUE eval if accessible
- fast score
- fast best move
- deep score
- deep best move
- score regret of the fast move under deep oracle
- fast completed depth / seldepth
- deep completed depth / seldepth
- fast nodes
- deep nodes
- PV change count / root best-move instability if obtainable
- eval instability across iterations if obtainable
- 8-to-7 tablebase proximity features: piece_count==8, legal captures/trades that may reach <=7 pieces, and exact Syzygy result after forced transition where feasible
- quiet/tactical classification
- rule50 count
- repetition context if available
- elapsed time

Store large raw data outside Git. Commit only small manifests, schemas, scripts, hashes, and summary JSON/CSV samples. Put local bulk paths in a manifest.

## Mission 2 — GPU risk model
Train a compact model to predict one or both of:

A. probability that `SF_fast` is materially wrong (`regret_cp >= threshold`),
B. expected regret magnitude.

Start simple and earn complexity:

1. logistic/MLP baseline on engineered search telemetry,
2. small board encoder + telemetry,
3. only then a larger transformer/CNN if the smaller model shows signal.

Use GPU for training and batched inference. Keep a CPU baseline for every model so we know whether the GPU is actually buying useful latency/throughput.

### Prospective gate
The model does not pass because training loss improves. On an untouched holdout it must show useful decision separation, for example:

- materially better-than-random ROC-AUC / PR-AUC,
- top-risk bucket captures a disproportionate share of high-regret Stockfish errors,
- calibration is usable enough to set a threshold,
- net expected rescue from spending extra CPU on flagged positions exceeds expected harm/cost.

Predeclare the exact pass rule before opening the holdout results. Save the rule in the experiment manifest.

## Mission 3 — alternative-move / verification model
If Mission 2 passes, add a second head or model that predicts which non-primary root moves deserve verification. Candidate sources may include:

- top-k root moves from the fast search,
- near-cutoff / near-miss moves,
- moves whose shallow/deep evaluation trajectories are unstable,
- forced 8-to-7 piece transitions,
- low-branching positions with unstable PV,
- quiet moves with large deep-search reversal history.

Do not copy Lc0 policy as the target. The target is deep-oracle regret reduction.

## Mission 4 — root/near-root GPU-assisted search prototype
Only if the offline model passes:

1. Keep CPU alpha-beta as the authoritative search.
2. Invoke the GPU model only at root/near-root or batched selected frontier points.
3. Let GPU output alter resource allocation, not objective truth. Examples:
   - extend time/nodes when root error risk is high,
   - verify top-k alternatives when predicted regret is high,
   - prioritize a candidate 8-to-7 exact transition,
   - spend less when risk is confidently low.
4. Measure GPU round-trip latency, batching efficiency, VRAM, CPU-GPU overlap, and total wall-clock impact.
5. Compare against CPU-only P01/Stockfish at equal node budget AND equal wall time. A GPU idea that improves per-node quality but loses wall time is not a win.

## High-risk/high-reward side probes
Run these only as bounded offline probes after the core miner exists:

1. **8→7 transition detector:** identify 8-piece positions where one or two plausible moves force entry to <=7 pieces and use Syzygy exactness as a target.
2. **Time-management blind-spot predictor:** predict positions that look root-stable but flip under deeper search. This is an allocator signal, not a new chess evaluation.
3. **Static-eval correction head:** predict shallow-to-deep evaluation shift from NNUE + search telemetry.
4. **Opponent-specific practical selector:** only after objective strength is preserved. It may rank objectively near-equal moves by measured probability of Stockfish finite-budget error, but never sacrifice large objective value just to be tricky.

## Experimental discipline
Every experiment must contain:

- hypothesis
- baseline
- mutation
- exact data split and seed
- frozen holdout
- pass/fail rule declared before holdout inspection
- CPU cost
- GPU cost
- wall time
- artifact hashes
- conclusion: SURVIVE / REJECT / INSUFFICIENT_EVIDENCE

Do not parameter-fish after seeing holdout results. A new threshold after failure is a new generation and requires a fresh holdout.

## CPU work is still required locally
The presence of a GPU does NOT exempt you from CPU tests. Run CPU-side work when it is required to establish:

- Stockfish fast/deep labels
- equal-node search quality
- fixed-time end-to-end latency
- exact behavior / correctness
- CPU baseline inference
- engine integration

The GPU lane should complement, not replace, CPU search research.

## Game budget
Competitive games are the last gate, not the first.

- default: 0 games
- if prospective offline gates fail: 0 games
- if offline gates pass but wall-time integration fails: 0 games
- if both pass: at most 40 paired games for a micro-screen
- an inconclusive 40-game screen means STOP and report inconclusive; do not silently launch 100/400/1000 more

## Required outputs
Create/update:

- `local_results/gpu3060/STATUS.md`
- `local_results/gpu3060/preflight/system.json`
- `local_results/gpu3060/experiments/<experiment_id>/manifest.json`
- `local_results/gpu3060/experiments/<experiment_id>/summary.json`
- scripts under `tools/gpu3060/`
- model code under `tools/gpu3060/models/`
- dataset schema under `tools/gpu3060/schema/`

Do not commit multi-GB datasets or checkpoints. Commit hashes and paths/manifests. Small checkpoint prototypes are allowed only if clearly useful and reasonably sized.

## Handoff format
At the end of each meaningful generation, append to `local_results/gpu3060/STATUS.md`:

- DATE/TIME
- COMMIT
- HYPOTHESIS
- RESULT
- CPU COST
- GPU COST
- WALL TIME
- WHAT FAILED
- WHAT SURVIVED
- NEXT CHEAPEST DISCRIMINATING TEST
- BETTER-THAN-STOCKFISH IMPACT: positive / negative / none / unresolved

Commit and push the scripts and compact result summaries to `leviathan/gpu-worker-3060` so Worker A can inspect them from GitHub.

## First execution now
Do Missions 0 and 1 smoke/development first. If the miner is valid and produces enough positive examples, proceed to a small Mission 2 baseline model. Do not jump directly to a large neural architecture. Do not launch chess matches. Keep iterating autonomously while information gain remains positive, but stop when the next useful evidence would require a larger resource tier and summarize that dependency instead of spending blindly.
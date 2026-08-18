# Leviathan Hybrid — CPU Search + GPU Advisor + Opponent-Clock Harvesting

## Status

P18.2 is the hardened dual-component prototype based on `leviathan/p09-static-race`. P09 remains the authoritative alpha-beta engine and safe fallback. The hybrid layer is externalized through UCI so it can be measured, disabled, or rolled back without changing the protected chess core.

## Architecture

```text
GUI
 |
 v
Hybrid UCI proxy
 |------------------------------> P09 authoritative CPU search
 |
 |  authorized opponent ponder window
 +--> shallow opponent MultiPV probe
 +--> GPU reply / risk / regret advisor
 +--> bounded CPU scout portfolio
 +--> portfolio annealing while opponent keeps thinking
 |
actual reply
 +--> matching warm scout promoted to full Threads
 +--> miss => ordinary P09 fallback
```

The GPU never decides the move and is never called synchronously at every node. It allocates speculative compute. CPU alpha-beta remains authoritative.

## P18.2 opponent-clock exploitation

Ordinary ponder bets the entire opponent-time budget on one reply. P18.2 treats that time as a speculative-compute portfolio.

When the GUI authorizes `go ponder`:

1. Reconstruct the position after Leviathan's last move.
2. Run a cheap MultiPV probe using the exact opponent binary when available.
3. GPU scores each candidate with:
   - reply probability,
   - probability finite-budget P09 is materially wrong after that reply,
   - expected centipawn regret.
4. Rank by expected verification value rather than reply probability alone.
5. Start wide across several replies while keeping total scout threads <= configured `Threads`.
6. If the opponent moves quickly, wide coverage maximizes ponder-hit probability.
7. If the opponent keeps thinking, P18.2 progressively anneals the portfolio: low-value branches are stopped and their thread budget is transferred to higher-value survivors. Surviving processes retain their TT/search history.
8. When the real move arrives, promote the matching warm branch to full threads.
9. If no branch matches, fall back immediately to cold P09.

No speculative thinking is performed unless the GUI authorizes pondering.

## GPU model

P18.2 uses three outputs:

- **reply head** — probability a shallow opponent probe candidate survives a stronger independent opponent search / observed opponent decision.
- **risk head** — probability P09's finite-budget search in that reply branch has material regret.
- **regret head** — expected centipawn regret magnitude.

The scheduler therefore distinguishes "likely but safe" replies from "less likely but dangerous" replies.

Without a trained checkpoint the proxy uses deterministic heuristics. Heuristic mode proves plumbing only and is not a strength claim.

## Training discipline

`mine_finite_compute.py` now avoids circular opponent labels:

- candidate telemetry comes from `--reply-nodes` (cheap probe),
- reply truth comes from a separate `--opponent-label-nodes` search,
- P09 risk comes from `--fast-nodes` vs `--deep-nodes`,
- the fast move is deep-verified when it differs,
- `regret_cp` is stored directly.

`train_risk_model.py` uses:

- grouped train/validation splits so replies from one position never cross folds,
- a completely separate optional prospective dataset,
- train-only feature normalization stored in the checkpoint,
- class-balanced masked BCE for reply/risk heads,
- SmoothL1 regression on log1p regret,
- early stopping,
- ROC-AUC, PR-AUC, Brier, reply top-1, regret MAE, and top-risk regret-capture metrics,
- predeclared promotion gates.

A checkpoint can be saved for research while still being rejected for champion use.

## One-shot RTX pipeline

```powershell
.\tools\hybrid\run-hybrid-one-shot.ps1 `
  -Engine C:\path\to\leviathan-p09.exe `
  -OpponentEngine C:\path\to\stockfish.exe `
  -Threads 8 `
  -Hash 128 `
  -Games 80
```

The script:

1. verifies CUDA/PyTorch,
2. generates diverse engine-distribution positions,
3. deterministically freezes 20% as a prospective holdout before labels are mined,
4. mines train labels,
5. mines the untouched holdout separately,
6. trains the 3-head CUDA advisor,
7. rejects the checkpoint if prospective gates fail,
8. prints the exact P18.2 launch command only after the model passes.

Generated bulk data stays under `local_results/` and is not committed.

## Main files

- `tools/hybrid/leviathan_hybrid_uci.py` — original executable multi-ponder controller.
- `tools/hybrid/leviathan_hybrid_uci_v2.py` — regret-aware scheduler + portfolio annealing.
- `tools/hybrid/gpu_risk_model.py` — 2-head backward-compatible / 3-head P18.2 scorer.
- `tools/hybrid/generate_training_positions.py` — deterministic diverse position generator.
- `tools/hybrid/split_positions.py` — prospective pre-label split.
- `tools/hybrid/mine_finite_compute.py` — shallow opponent probe, stronger opponent labels, P09 finite-compute risk/regret mining.
- `tools/hybrid/train_risk_model.py` — grouped/normalized/class-balanced 3-head trainer.
- `tools/hybrid/run-hybrid-one-shot.ps1` — end-to-end local pipeline.
- `tools/hybrid/test_hybrid_core.py` — deterministic UCI orchestration tests.
- `tools/hybrid/test_p18_v2_policy.py` — regret-aware scheduling and thread-budget tests.

## Promotion gates

P18.2 remains experimental until all of these pass:

1. UCI predicted-hit, alternative-hit, miss, `stop`, `ponderhit`, and game-reset correctness.
2. Scout threads never exceed configured `Threads`.
3. No per-node blocking GPU calls.
4. Prospective reply coverage measured against the target opponent.
5. Prospective risk AUC and regret-capture gates pass on positions frozen before training.
6. Warm hits beat cold P09 at equal own-clock wall time.
7. Full CPU+GPU P18.2 beats CPU-only P09 after predictor/GPU/scout overhead.
8. Only then compare P18.2 against Stockfish at fixed time.
9. Every failure falls back safely to CPU-only P09.

The primary metric is not GPU utilization. It is **useful authoritative-compatible search completed before Leviathan's own clock starts, weighted by the probability and consequence of needing that work**.

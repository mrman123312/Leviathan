# Leviathan Hybrid — CPU Search + GPU Advisor + Opponent-Clock Harvesting

## Status

P18 is the first executable dual-component prototype, based on `leviathan/p09-static-race`. It deliberately leaves P09's authoritative alpha-beta code unchanged and implements the GPU/ponder layer as a UCI proxy so the architecture can be tested and rolled back cheaply.

## Architecture

```text
GUI
 |
 v
Hybrid UCI proxy
 |---------------------------> P09 authoritative CPU search
 |
 |  opponent's authorized ponder window
 +--> opponent-model CPU MultiPV probe
 +--> GPU reply / finite-compute risk scorer
 +--> bounded CPU scout pool across likely replies
 |
actual reply
 +--> matching warm scout promoted to full Threads
 +--> miss => ordinary P09 fallback
```

The GPU never decides the move and is never called synchronously at every node. It only ranks where speculative work should be spent. CPU alpha-beta remains authoritative.

## Abusing the opponent's think time — legally through UCI ponder

Ordinary ponder bets the whole opponent-time budget on one reply. P18 turns it into a reply portfolio.

When the GUI explicitly sends `go ponder`:

1. Reconstruct the position immediately after Leviathan's last move.
2. Run a small MultiPV opponent-model search. `--opponent-engine` may point to the exact Stockfish binary being faced.
3. Send candidate telemetry to the GPU scorer.
4. Select up to `Leviathan Hybrid Scouts` likely/high-risk replies.
5. Partition the configured total `Threads` budget across those scouts. Scouts do **not** each get the full thread count.
6. Let every branch search while the opponent's clock is running, warming its own TT and search history.
7. When the real opponent move arrives, promote the matching warm process, stop the other branches, restore the full thread budget, and restart authoritative search with that warm TT/history preserved.
8. If the opponent move is outside the covered set, fall back to cold P09.

No hidden thinking is done when the GUI does not authorize pondering.

## GPU model

`tools/hybrid/gpu_risk_model.py` has two conceptual heads:

- **reply head**: probability a candidate is the opponent's actual move.
- **risk head**: probability P09's finite-budget search in that branch is materially wrong and deserves verification.

Without a trained checkpoint the proxy uses a conservative deterministic prior dominated by CPU MultiPV ordering. That mode is plumbing, not a claimed strength gain.

### Learn the actual opponent

`build_reply_dataset.py` converts hybrid session logs into reply labels. Over repeated games against one Stockfish build, the GPU can learn that engine's move-choice distribution instead of copying human policy or generic Lc0 policy.

### Learn where P09 changes its mind

`mine_finite_compute.py` compares fast and deep P09 searches after each candidate opponent reply and verifies the fast move under the deep oracle. Branches whose regret exceeds the configured threshold become positive risk examples.

`train_risk_model.py` trains both heads and stores a checkpoint loadable by the proxy.

## Files

- `tools/hybrid/leviathan_hybrid_uci.py` — multi-ponder UCI controller.
- `tools/hybrid/gpu_risk_model.py` — CUDA/CPU scorer.
- `tools/hybrid/build_reply_dataset.py` — observed opponent-reply labels.
- `tools/hybrid/mine_finite_compute.py` — deep-oracle risk mining.
- `tools/hybrid/train_risk_model.py` — two-head MLP trainer.
- `tools/hybrid/real_hybrid_smoke.py` — real-engine end-to-end smoke.
- `tools/hybrid/run-hybrid-rtx3060.ps1` — local RTX 3060 preflight/smoke/miner.
- `tools/hybrid/test_hybrid_core.py` — deterministic predicted-hit and alternative-hit tests.

## RTX 3060 smoke

```powershell
.\tools\hybrid\run-hybrid-rtx3060.ps1 `
  -Engine .\src\stockfish.exe `
  -OpponentEngine C:\path\to\stockfish.exe `
  -Threads 8 `
  -Hash 128
```

The smoke requires CUDA-enabled PyTorch and only passes when the log reaches both `ponder_pool_ready` and `warm_promote`.

## Promotion gates

P18 is experimental until all of these pass:

1. UCI predicted-hit, alternative-hit, miss, `stop`, `ponderhit`, and game-reset correctness.
2. Sum of scout CPU threads never exceeds configured `Threads`.
3. No per-node blocking GPU calls.
4. Prospective candidate coverage measured against the target opponent.
5. Warm hits beat cold P09 at equal **own-clock** wall time.
6. Whole CPU+GPU engine beats CPU-only P09 after predictor/scout overhead.
7. Trained models use fresh holdouts and predeclared gates.
8. Every failure has a safe CPU-only P09 fallback.

The key metric is not GPU utilization. It is **useful search already completed before Leviathan's clock starts**.

# Leviathan — Frozen Bedrock v3

Frozen pretrained computation, one model, no training. **ARC-Easy is now in the one-click run.**

Double-click **`RUN_ARC_EASY.bat`**. It uses the already-working v7 CUDA environment and cached Qwen3-1.7B-Base. No installation, model download, drive scan or training. `RESULTS.html` opens automatically; timestamped runs preserve earlier results.

## New executable mechanisms

Prediction-aware stopping uses the original final norm/output head and reports its overhead. DIRECT / REFINE / EXPLORE modes make task-level adaptive decisions from option scores, never answer keys. Signed causal latent directions create different bounded routes through the same frozen weights. Activation-conditioned cell analysis replaces the loose gate/up weight-norm product while accounting for dense work. The same frozen Qwen can propose typed rule expressions, receive counterexamples, validate them through host experiments, and save an executable skill without a supplied true-rule catalogue.

These implementations do not guarantee better intelligence. Malformed neural proposals, incorrect stable answers, slower routes and numerical fallback are recorded rather than hidden.

## Next benchmark

The first 50 ARC-Easy test questions compare donor, anchored refinement, two-pass transport, four-pass transport, and adaptive execution. Raw, token-normalized and character-normalized accuracy are separate. Adaptive time includes every donor/refinement/exploration call it actually makes. Reports include changed answers, counts, sample uncertainty, fallback counts and seconds per correct answer.

A 64-passage cached WikiText retention check follows, then the new mechanism probes and a neural-world demo. The old 72%/50 baseline is historical, not filled in as a new score. The 64-passage selection differs from the earlier 32-passage canary.

## Evidence

179 local CPU tests pass, including the prior 151. Native tiny-Qwen integration is a separate CI test. Full pretrained RTX accuracy must come from the next real run; no new ARC, WikiText or GPU speed claim is made by the source code alone.

Read [implementation, proof limits, and protocol](docs/24-expressive-arc.md). Earlier history is retained in [the Bedrock report](docs/22-frozen-bedrock.md), [transport repair](docs/23-transport-recurrence.md), and [pre-Bedrock README](docs/pre-bedrock-readme.md).

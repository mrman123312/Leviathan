# Frozen Bedrock v3: expressive execution with ARC-Easy

## Basis and corrections

The user's RTX 3060 FP16 transport run completed without numerical fallback on three prompts. It did not measure ARC accuracy, general reasoning improvement or a speedup. The adaptive variant used all allowed passes on those prompts. This establishes **no compute saving on that sample**, not that every possible adaptive implementation is fake or that the thresholds must be loosened until it exits.

The input L2 trust region keeps probes near an observed donor state. It does **not** establish membership in a learned manifold or guarantee semantic preservation. This version uses the term *donor-input-bounded*.

The 36-world/137-query finite-grammar experiment is not a neural-model result: it made zero neural calls. The new neural-world path separately measures actual same-Qwen proposals without supplying the correct-rule catalogue. It may fail to propose a valid rule; that is reported, never disguised by catalogue fallback.

## 1. Prediction-aware stopping

`StopPolicy` inspects the **same model's original final norm and output head**, not a random auxiliary head. It records top-token identity, probability, margin, normalized entropy, change in log probabilities, coarsened Jensen–Shannon divergence and relative hidden-state movement.

The distribution comparison uses previous top-k categories plus an 'other' bin. It is a cheap, explicitly coarsened proxy, not full-vocabulary JS. Softmax-normalized differences avoid interpreting a common additive logit shift as novel reasoning.

A confident first pass may exit. Later exits require stable predictions and small state movement for a configured patience. Thresholds are fixed before this version's ARC run. There is no fit/optimizer. A stable incorrect answer can halt, so this is **not correctness certification**.

Per-position decisions remain causal. The final decoder band is required: an arbitrary middle-layer untrained logit lens is rejected. The output head is probed in bounded chunks. All probe work is included in actual runtime. Positions can stop changing without physically removing causal attention rows; this is recorded. Full-band calls stop only when every position has halted.

## 2. DIRECT / REFINE / EXPLORE

ARC uses a separate task-level choice controller. The tested alternatives are:

- DIRECT: original donor, no extra layers.
- REFINE: two-pass anchored correction.
- EXPLORE: transported recurrence, two or four passes.
- ADAPTIVE: execute donor scores; either stop or execute refinement; either stop or execute four-pass exploration.

The adaptive decision function accepts **scores only**, not answer keys. Each stage is actually executed and its time charged to the adaptive total. It never reuses a previously benchmarked fixed-mode result and calls the selector free.

This task-level procedure is valid for multiple choice, but is not presented as a single causal language-model distribution with a WikiText loss. The fixed routes have separate WikiText retention results.

## 3. Latent branching

Signed alternatives are built from causal, inherited activation directions: preceding-token context differences, optionally orthogonalized against the band residual. Both signs can probe different bounded neighborhoods through the **same frozen layers**. No new parameter vectors or extra semantic model exist.

Different numerical trajectories are not automatically different meaningful hypotheses. Tests establish different outputs, unchanged weights and prefix causality. Useful hypothesis diversity remains an empirical question. Existing host verification / donor fallback remains the authority rule; no 'confident branch wins' claim is added.

## 4. Activation-conditioned cell relevance

Let `z = SiLU(Gx) * Ux` be the actual FFN intermediate activation. For cell i:

`C_i = D_i z_i`.

Two real-arithmetic bounds are available:

`||C_i||_2 <= ||D_i||_F ||z_i||_2`,

`||C_i||_2 <= sum_j |z_ij| ||D_ij||_2`.

The minimum is still an upper bound. Sum the unselected-cell bounds to bound the local FFN tail. This removes the original loose product of gate/input/up weight norms and accounts for actual inactive channels.

The new analysis computes dense gate/up once and a dense down-projection audit. Therefore it **does not prove a speedup**. Observe mode returns the original dense output; bounded mode falls back when the tail criterion is unmet. Floating-point rounding is not interval-certified, and a local FFN bound is not a final-logit or semantic-quality certificate.

## 5. Neural proposal -> experiments -> memory

`FrozenRuleProposer` reuses the runtime's one frozen Qwen and existing tokenizer. It sees only the declared input domain, observations, rejected hypotheses and a bounded expression syntax. It never receives the environment callback, hidden function, or transfer answers.

Generated expressions are parsed with a restricted AST-to-DSL translator; `eval`, imports, attributes, calls and arbitrary execution are forbidden. Candidate rules are tested against existing observations. Disagreement chooses further experiments; fresh validation tests a surviving candidate. Contradictions return to the same neural proposer. Validated executable rules can be stored and reused through the existing memory runtime.

A singleton neural proposal is not an exhaustive hypothesis class. Passing two fresh tests provides scoped empirical support, not a guarantee over the whole domain. Malformed and unsupported proposals are reported as failures. Mock-proposer tests are explicitly labeled and do not imply Qwen actually succeeds.

## ARC-Easy protocol and one click

Double-click `RUN_ARC_EASY.bat` or `RUN_FROZEN_BEDROCK.bat`. Both reuse:

`C:\LeviathanBenchmarkCache\.venv-v7\Scripts\python.exe`

and the existing model cache. No install, drive scan, CUDA replacement, CPU fallback or training is performed. The runner uses a packaged ARC JSON canary when available, otherwise the already-known local dataset cache. It never downloads weights.

The first 50 ARC-Easy test examples retain the earlier canary prompt:

`Question: {question}\nAnswer:` + space + choice text.

Raw summed option log likelihood is primary; token-normalized and character-normalized scores are separately named. Do not call the latter an exact official harness replication. The data source, example hash, policy hash, IDs, per-choice scores and all correctness changes are saved. Missing/error examples are excluded transparently and never treated as successes.

Modes are interleaved in a fixed pseudorandom order. CUDA is warmed. All adaptive computation is charged. Per-question progress and partial results are saved. A supervisor reports ongoing work and ends a child after 180 seconds with no output; timeout status is explicitly incomplete. Previous result directories are not deleted.

The runner then tests 64 qualifying cached WikiText passages (at most 256 tokens each), signed latent branches, prediction stopping, activation relevance and one neural-rule demo. Missing WikiText is explicitly not measured; it does not erase completed ARC results. This sample is different from the earlier 32-passage test.

## Evidence available at authoring

Local CPU PyTorch 2.10.0: **179 tests passed**, including the preceding 151 and 28 new tests. New tests cover numeric FP16 fixtures, different signed branches, actual removal of band calls in a stable fixture, causal prefix comparison, fresh observation/counterexample revision, no hidden rule fallback, rejection of arbitrary Python code, and full adaptive-cost accounting.

No full pretrained checkpoint, Windows launcher, or RTX GPU was executed in this authoring container. A separate native tiny-Qwen CI test is supplied; its results must be read before claiming it passed. The next user run supplies the real ARC/retention/hardware evidence. No new accuracy or speedup number is fabricated.

## External research, distinct from project results

- PyTorch numerical accuracy: https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html
- ARC source/card and license: https://huggingface.co/datasets/allenai/ai2_arc (CC-BY-SA-4.0).
- ARC task prompt reference: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/arc/arc_easy.yaml
- Training-Free Looped Transformers: https://arxiv.org/abs/2605.23872 (paper reports both improvements and naive-loop failures; not evidence of a Leviathan gain).
- Inner Loop Inference: https://arxiv.org/abs/2602.14759 (external frozen-computation research, not a replicated experiment here).

Logical invariants establish permitted behavior and isolation. They do not logically imply higher intelligence or better measured accuracy.

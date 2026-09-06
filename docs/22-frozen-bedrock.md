# Frozen Bedrock: executable no-training architecture

This work starts from `b168c2f6323365a3d08b9d4770ab25a7dd7a5e06`. It does not restart R0, replace the pretrained control, or delete prior DeepSeek/Qwen research. The user's newer prohibition on training supersedes the previous adapter-training recommendation. Existing training scripts are historical/optional, not called by this path.

The R3 negative result remains binding: theoretical sparsity is not wall-clock improvement. The earlier 582-parameter exercise was a microscope, not evidence for giant-scale economics. The user's measured RTX 3060 result (72%/50 ARC questions, identical 2.87624 WikiText NLL, zero tested logit difference, 25.81 vs 20.95 end-to-end tokens/s, displayed 3.22 GiB) remains an earlier preservation/overhead result, not a new Bedrock benchmark. The historical 64-passage protocol and the later 32-passage one-click protocol must not be conflated. The historical +0.057% figure is conversation history, not newly reverified evidence.

## What changed

`BedrockRuntime` extends the existing `LeviathanRuntime`. There is one semantic identity and at most one `FrozenExecutor` owning pretrained weights. Typed programs, host verifiers, memory, experiments and neural routes are operations of that system, not additional LLMs.

The new execution path contains no optimizer, fitting procedure, trainable router, random recurrent adapter, or new `nn.Parameter`. Native tiny-model tests initialize random test fixtures, freeze them, and perform no training; those fixtures are not purported pretrained models.

The repeated neural function is an actual existing decoder band. Both ordinary repeated-band and anchored-difference feedback are executable. The inherited first pass remains the anchor and fallback. Masks, rotary-position arguments and layer-specific context are captured from the original forward call. KV caches are forbidden in altered routes until a separate cache-equivalence implementation is established. This is deliberately a full-prefix correctness implementation, not a fast decoder.

Halting is per token position. Stopping based on the last token of an entire teacher-forced sequence could let future text alter earlier logits. Per-position decisions and causal attention prevent that structural leak. No convergence threshold is presented as confidence that an answer is true.

Ancestral cells compute real frozen SwiGLU slices. Their keys come from frozen gate/up weights. Bounded peer communication changes individual proposals while conserving the sum. Peer-derived queries recruit further same-layer cells. Four observable local moments are carried per token/cell across recurrent depth within one forward call. They are not trained confidence. Symmetric mutual-neighbor edges respect the neighbor cap. Splitting/merging cell views preserves channel ownership. Identically zero down slices can be identified for exact pruning. New learned capacity or general cell birth/death is not claimed.

Fast associations are directly constructed key/value activation maps, not weights learned by SGD. They are request-, revision- and layer-scoped, bounded, forkable and resettable. A symbolic pulse retains its hypothesis/observation/evidence labels and is re-encoded with the owning model's existing tokenizer and embedding. No new pulse head is introduced. Useful latent-to-symbolic reasoning has not been established by these mechanics.

Neural brainstorm branches evaluate different frozen routes through the same parameter owner. Only identical completed outputs are merged; equal token output does not establish equivalent hidden states. A host verifier can choose a successful branch. Without independent verification, the donor answer wins, even when an experimental branch is more confident.

The world learner uses a bounded typed AST and a declared finite hypothesis class. It chooses discriminating experiments, records predictions before observations, eliminates contradicted rules, performs fresh validation, stores a scoped executable program, and reuses it after a runtime restart. A counterexample deprecates the skill. The learner never sees protected transfer labels during discovery. This is genuine state/procedure acquisition without parameter training, but is not arbitrary novel-environment general intelligence.

## Proof obligations and their limits

### P0. Neutral computation preserves the donor

For a neutral policy, the executor calls the same original model directly, with no recurrent hooks, candidate FFN work or fast-state updates. Its computation graph is unchanged. Numerical equality still assumes the same precision, backend and deterministic behavior. The exact local tests compare actual output tensors. This is a valid structural guarantee; it is not a proof that a nonneutral policy is better.

### P1. No parameter update

The implementation reuses existing modules, freezes their parameters, and executes inside inference mode. It creates no neural parameters and invokes no optimizer/backpropagation. Tiny-model tests compare every state-dictionary tensor before and after. Runtime version counters are only a mutation tripwire, not a cryptographic full-checkpoint hash and not protection against malicious same-process `.data` writes.

### P2. Causal depth

Let the state at token t in round r depend only on tokens at positions <=t. Applying the original causal layers preserves that property. The stopping decision at t depends only on that position's states; freezing it does not import later-token information. Induction over rounds therefore preserves causality. The global loop stops only when all individual positions have halted, which cannot change an already halted earlier state. Prefix-versus-full-sequence tests exercise this obligation.

### P3. Bounded temporary change

For anchor h and proposed delta d, use

`d' = d * min(1, rho ||h||_2 / ||d||_2)`.

Then `||d'||_2 <= rho ||h||_2` in real arithmetic. Ordinary FP rounding remains. This bounds a local residual change, NOT the final logit change or semantic error. A downstream Lipschitz/interval proof would be needed for the latter.

### P4. Conservative communication

Let M contain cell contributions and let A be a symmetric mutual-neighbor adjacency. With D the degree diagonal, the exchange is

`M' = M + beta (A-D) M / max_degree`.

Since `1^T(A-D)=0`, `sum_i M'_i=sum_i M_i`. Each cell can receive a revised proposal without inventing aggregate signal. The implementation returns the original body sum to avoid changing its accumulation order. Communication can influence subsequent recruitment, but conservation alone cannot improve the model's answer.

### P5. FFN contribution bound

For one cell,

`C_i(x)=D_i [SiLU(G_i x+b_g) * (U_i x+b_u)]`.

Using `|SiLU(z)|<=|z|`, submultiplicativity and the Frobenius upper bound on spectral norm:

`||C_i(x)||_2 <= ||D_i||_F (||G_i||_F ||x||_2+||b_g||_2) (||U_i||_F ||x||_2+||b_u||_2)`.

The sum of these bounds upper-bounds a skipped FFN tail in real arithmetic. The bounds can be extremely loose. If the configured cell cap cannot satisfy the requested tail criterion, the executor falls back to the dense donor. This is not an interval-arithmetic certificate for downstream logits. Dense fallback and routing overhead are accounted for; no speed improvement is claimed.

### P6. Route continuity

For scores `s_j(q)=|q^T k_j|`, each score changes by at most

`epsilon=||q_new-q_old||_2 max_j ||k_j||_2`.

A selected/unselected gap greater than `2 epsilon` preserves top-k membership. The implementation adds a conservative floating-point margin, keeps the original query/score anchor, and disables reuse when state-dependent scoring changes. It does not assert that all normalized mixture coefficients are unchanged.

### P7. Same-model speculation

Greedy block verification accepts only the target model's greedy prefix and emits its first corrective token. The sampling reference uses `min(1,p/q)` acceptance and the normalized positive residual `(p-q)+` on rejection. Under correctly aligned distributions this preserves the target distribution, not necessarily the exact random seed trajectory. The full-prefix implementation intentionally avoids stale branch caches. It may be slower than direct generation.

### P8. Finite rule learning

If the true deterministic rule is in the declared hypothesis set and observations are correct, consistency elimination cannot remove it. A separating experiment removes incompatible hypotheses. Behavioral deduplication on the finite declared domain permits a unique executable rule even if multiple ASTs describe it. Fresh validation and unseen-action evaluation test persistence and transfer. Outside this hypothesis class, the learner must report misspecification rather than invent certainty.

### P9. Evidence and memory

Evidence IDs bind observations to preceding predictions. Repeating an ID is idempotent, not another success. Stored nested payloads and belief histories are copied to prevent alias mutation. A journal append succeeds before in-memory promotion. Persistent IDs cannot be overwritten by promotion. Failed dependencies block descendants. These are integrity properties, not an adversarial operating-system sandbox. Long-horizon causal credit is exact only relative to a supplied deterministic SCM; the system has not inferred a universal causal world model.

## Connected feature map

| Requested capability | Executable implementation | Remaining evidence or gap |
|---|---|---|
| Frozen-layer recurrence | `neural.FrozenExecutor.run` | Full pretrained Qwen accuracy/retention at nonzero recurrence |
| Anchored recurrence | `FrozenPolicy.feedback` | Whether it reduces distribution-shift damage |
| Adaptive depth | Per-position delta/patience | Better quality/cost, not merely halting |
| Expressive ancestral cells | Actual bodies, conservative messages, recruitment, local moments | Useful calibrated signals and large-scale sparse economics |
| Confidence/abstention | Explicit proxy labels, bound failure -> donor fallback, unknown verification outcome | Learned or empirically calibrated confidence not claimed |
| Cell lifecycle | Exact contiguous split/merge, zero-slice pruning | General growth and nonzero pruning remain research |
| Neuroplastic state | Scoped bounded associative maps | Useful within-task adaptation on real language tasks |
| Latent branching | Different frozen routes, one weight owner, host selection | Learned semantic hypothesis slots not claimed |
| Pulse-CoT | Typed scratch artifacts through original tokenizer/embedding | Useful automatic reasoning-pulse generation |
| Sparse delta recurrence | Exact row-local FFN cache only | Approximate and attention/DeltaNet delta reuse pending |
| Route reuse | Margin-bounded seed reuse | GPU speedup and realistic hit rates |
| Same-model speculation | Greedy and corrected sampling | Fast MTP kernels and transactional KV handling pending |
| Adaptive representation | Executable typed integer AST vs token path | Automatic invention of general representations pending |
| Belief revision | Finite version spaces plus existing belief store | Noisy/open-world hypothesis induction pending |
| Experiment selection | Explicit information gain per cost | Real environments beyond the supplied grammar |
| Skill compilation | Persist scoped executable rule, fresh validation, reload | General skill/code synthesis pending |
| Metacognition | Mode policy, budget meter, deduplicated competence counts | Calibrated/general learned self-model not claimed |
| Verification | Host receipts, bound output identity, unknown status | No perfect open-world truth oracle |
| Causal accountability | Dependency invalidation, supplied-SCM counterfactual | Causal structure discovery remains open |
| One model | Existing runtime extended; one neural parameter owner | No multi-LLM ensemble |
| L10 | Intentionally untouched | Later per mandate |

## Tests and experiments

Run `python -m unittest discover -s tests -v`. The full local suite has 147 passing tests, including 50 new no-training tests. The baseline's 97 tests still pass.

`benchmark_bedrock_mechanisms.py` used 12 worlds per family with a fixed seed. All 36 programs passed fresh validation. Reloaded skills answered 137 previously unqueried domain inputs correctly: 84 affine, 36 bit-permutation and 17 Boolean-circuit queries. Some circuit cases used their entire finite domain for discovery/validation and therefore contributed zero unseen queries; the report preserves each denominator.

Mean discovery queries were 2, 3 and 4.58. The fixed-order comparator used 2, 3 and 7.67 respectively. The 4.58-vs-7.67 comparison excludes the two fresh validation observations. These are finite-grammar algorithm results, not neural language scores, novel general intelligence, or proof of a new mathematical invention.

`check_bedrock_hf.py` is a native Transformers tiny-Qwen test. It requires the optional dependency and must not be described as run unless its result is available. The current local container has CPU PyTorch but no Transformers and no internet download path. No new full 1.7B, 27B, ARC, WikiText, GPU, or language-speed result was obtained locally.

## One-click continuation without reinstalling

`RUN_FROZEN_BEDROCK.bat` uses the already-working v7 environment at `C:\LeviathanBenchmarkCache\.venv-v7\Scripts\python.exe` and the existing Hugging Face cache. No drive scan, CUDA download, Python environment recreation, CPU fallback, paid API or training is launched. It runs local mechanism tests, then a small CUDA experiment using cached Qwen weights and opens results. The Windows/CUDA path cannot be exercised in this Linux CPU container; this is explicitly not a promise that an untested Windows environment cannot fail.

The original working v7 launcher and its pinned baseline remain unchanged.

## Sources and reasoning provenance

Project basis: user-supplied R0-R9 history; `Pasted markdown(3).md` parameter-cell conceptualization; existing controller, belief, memory, verification and parameter-ecology documents. Source claims about historical frontier benchmarks were not assumed independently verified.

Inspected native implementation: Hugging Face Transformers `v5.8.1`, `src/transformers/models/qwen3/modeling_qwen3.py`, including layer returns, masks, rotary arguments and cache semantics. Hybrid Qwen3.5/3.8 has additional state contracts and is not silently treated as Qwen3.

External research context: https://arxiv.org/abs/2502.05171 and https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/qwen3 . Huginn's trained recurrent model motivates recurrence; its results are not evidence that arbitrary frozen Qwen layer looping works. The bounds above are explicit local derivations, not claims copied from a paper about Leviathan.

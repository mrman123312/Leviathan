# Function-preserving weight transplantation

Leviathan should inherit pretrained capability without assuming that arbitrary architectures can directly share weights.

## Compatibility classes

### Class 1 — direct-compatible reuse

Safe when tensor semantics and dimensions match.

Examples:

- embeddings/tokenizer kept unchanged;
- copied transformer blocks with identical architecture;
- duplicated MoE experts initialized from an existing expert;
- unchanged normalization/output heads.

### Class 2 — compatible with adapter/gate

Use when the pretrained path remains valid and a new path can be added in parallel.

Canonical form:

`y = F_pretrained(x) + g * G_new(x)`

Initialize `g = 0`.

Good candidates:

- persistent memory attention;
- recurrent-state channel;
- world-model state injection;
- extra experts;
- depth retrieval / Attention-Residual-like paths;
- additional multimodal cross-attention;
- new action heads.

### Class 3 — projection bridge

Use when two pretrained components have useful representations but incompatible latent spaces.

`z_A -> P_A_to_L(z_A) -> z_Leviathan`

Train the projector first while both donor systems are frozen.

Good candidates:

- external vision encoder -> Leviathan state;
- audio encoder -> Leviathan state;
- world-model latent -> belief state;
- GUI visual representation -> native action head.

### Class 4 — distillation required

Use when computation differs fundamentally.

Examples:

- full attention -> state-space/recurrent block;
- incompatible MoE widths/routing semantics;
- dense model -> structurally different sparse model;
- one tokenizer/vocabulary -> fundamentally different representation scheme;
- unrelated residual topology.

Train the new module/student using one or more of:

- token loss;
- teacher-logit KL;
- hidden-state matching;
- contrastive representation matching;
- trajectory imitation;
- verifier-scored behavior;
- environment outcome reward.

## Mandatory transplant protocol

1. **Baseline lock** — record exact checkpoint revision, tokenizer, inference engine and evaluation suite.
2. **Base fingerprint** — save benchmark, calibration, latency, throughput and memory metrics.
3. **Insert inert modules** — new components must initially have zero or identity effect.
4. **Parity test** — before training, output drift must be explained and bounded.
5. **Freeze donor** — first train only the newly introduced parameters.
6. **Gate warm-up** — gradually increase new-path contribution.
7. **Selective unfreeze** — unfreeze only the smallest necessary donor parameter groups.
8. **Replay** — mix new-task data with retention/capability/safety replay.
9. **Independent evaluation** — the training policy may not be the sole evaluator.
10. **Shadow deployment** — compare against the locked baseline before promotion.
11. **Rollback artifact** — every promoted parametric state must retain a restorable predecessor.

## Expert expansion

If an MoE substrate has expert `E_i`, a new expert can begin as a copy:

`E_i -> {E_i_old, E_i_new}`

Initially route almost no traffic to `E_i_new`, then specialize it using domain- or function-specific objectives.

Do not assume expert identities are semantically clean. Measure expert activation and causal contribution before assigning human labels such as "math" or "planning".

## Heterogeneous expert migration

The long-term target allows non-MLP cognitive experts.

A safe migration path is:

1. keep original FFN experts;
2. add a new expert class with zero route probability;
3. train the new expert on states where an external process demonstrates value;
4. introduce small routing probability;
5. compare against matched FFN routing;
6. retain only if quality-per-compute improves;
7. never remove the old route until regression suites pass.

Candidate non-MLP expert classes:

- recurrent state update;
- episodic-memory retrieval;
- latent world prediction;
- causal counterfactual scoring;
- planner state transition;
- verifier query interface.

## Attention migration

To replace an expensive attention path with a local/recurrent/sparse path:

`y = (1-beta) * Attention_old(x) + beta * NewSequenceBlock(x)`

Start `beta = 0`.

Train the new block against:

- next-token objective;
- old-block representation target;
- long-context retrieval tasks;
- state-prediction tasks;
- downstream behavior.

Only increase `beta` when both capability and stability remain acceptable.

## MTP extension

Text MTP predicts future tokens. Leviathan's target is multi-channel future prediction:

- token head: `x[t+1:t+k]`;
- action head: `a[t+1:t+k]`;
- state head: `z[t+1:t+k]`;
- verifier/outcome head: expected success or observation class.

Training these heads does not authorize autonomous action. Execution remains behind the action-policy and risk gates.

## Multimodal transplantation

Never concatenate arbitrary latent vectors and assume interoperability.

Each donor modality requires:

- normalization contract;
- projection into shared belief space;
- time/alignment metadata;
- confidence/provenance;
- modality dropout tests;
- adversarial mismatch tests.

The shared state should retain enough information to distinguish:

`observed` vs `retrieved` vs `predicted` vs `simulated`.

## Continual-learning boundary

Weight transplantation and continual learning are separate operations.

Architectural surgery is an offline engineering procedure.

Runtime learning follows Leviathan's memory/promotion pipeline:

`experience -> verification -> memory -> skill -> plastic candidate -> regression -> optional core consolidation`

Raw deployment experiences never directly mutate the core substrate.

## What not to do

- Do not average unrelated giant-model checkpoints.
- Do not map tensors solely because their shapes match.
- Do not destroy the donor tokenizer unless there is a measured migration path.
- Do not let a new module silently bypass verifier/governance controls.
- Do not use teacher self-confidence as ground truth.
- Do not accept a transplant that raises one benchmark while causing unknown regressions.
- Do not commit model weights into the Leviathan Git repository.

The engineering goal is **capability-preserving architectural evolution**, not a one-shot rewrite.

# Consumer NRDF: implementation boundary

## Why a graft first

The donor computes its original FFN, then adds `tanh(g) * G(x)` with `g=0` initially. The default changes only the final decoder FFN; additional layers require explicit selection. This preserves causal masks, positions, attention and DeltaNet state. It is not full-backbone looped pretraining, nor reduced dense donor FLOPs.

Each causal token state seeds parallel latent slots. One shared attention/MLP transition repeats with input injection. Slot soft selection provides merge mechanics; learned semantic branches and pruning are still pending. Optional step conditioning is an ablation, not a universal improvement.

## Cells and plastic state

Ancestral cells access actual gate/up rows and down columns. All cells reconstruct the dense FFN; arbitrary subsets do not automatically reconstruct it. Sparse bodies currently influence a new residual, not replace the dense path. Cells communicate, recruit within hard budgets, revise and only then commit a proposal.

State is explicitly passed across depth and recreated for independent forward calls. It is not averaged across batch members. This avoids the historical global-cell-buffer leakage problem. Long-lived task state requires a separate causal cache protocol, which is not implemented.

Low-rank fast state is norm-bounded and produced by a learned updater. It is not raw experience modifying pretrained weights. Useful adaptation must be demonstrated separately.

Matching hidden dimensions is insufficient for cross-layer recruitment: layers have different learned representations. Cross-layer routes stay disabled until trained compatibility bridges and retention evidence exist.

## Quantization

The production-oriented baseline uses the original bitsandbytes NF4 modules. A float recurrent adapter can surround them. Sparse NF4 slicing is rejected rather than silently dequantizing or misindexing packed tensors.

A separate symmetric group-INT4 correctness format supports nibble unpacking, local scale lookup and actual tile slices. It is not NF4, AWQ or GPTQ, and no speed kernel is claimed. Quantization error versus a float donor and cellization error versus the same quantized donor are reported separately.

## Gates and gradients

Training executes the candidate graph at zero outer gate. Only the gate necessarily receives task gradient at that point; inner modules need auxiliary alignment losses or subsequent gate opening. Evaluation can bypass a zero gate exactly. A nonfinite candidate is rejected because zero times NaN does not preserve function.

## Pulse bridge

The optional bridge decodes through the owning model's LM head and re-encodes through its existing embedding. Weak references avoid a second registered copy. Argmax is discrete; an alignment loss trains the readout. Random bridge tokens have no assured reasoning meaning. No private assistant scratchpad is collected.

## Halting

Training samples fixed depths. Hard adaptive halting is evaluation-only and compacts continuing rows. The head is disabled by default until calibrated. Tests of forced halting prove mechanics, not learned allocation. Report quality/cost at several depths and include a depth not used during training. More loops can harm quality.

## Efficiency experiments and falsifiers

Exact-input SDR only reuses deterministic row-local pure operators when all dependencies are captured by input and scope epochs. It is invalid for arbitrary causal attention or DeltaNet. ByteLRU bounds retained storage, not transient peak allocations.

Three additional falsifiable mechanisms are implemented as correctness primitives:

1. **Route-margin certificate:** bound every linear routing-score perturbation; reuse a top-k set only if the selected/unselected gap exceeds twice that bound. Recompute coefficients. Reject if certification costs more than rerouting.
2. **Stable-body coefficient delta:** update an aggregate from coefficient changes only when nonlinear body outputs are unchanged. Measure floating-point drift and refresh; reject stale-body reuse.
3. **Greedy-logit margin certificate:** preserve argmax only with a justified bound on every logit change and a sufficiently large margin. This does not certify sampling or derive bounds from latent similarity.

These are project-specific combinations of existing ideas, not claims of mathematical novelty. Their production GPU usefulness is unmeasured.

## Acceptance

Compare a pinned donor and zero-gated candidate under the same precision, backend, prompts and cache settings. Then train one mechanism at a time. Require heldout language, ARC-Easy, ARC-Challenge, knowledge, math, coding, calibration, safety and real cost tests. Never tune repeatedly on a protected test suite. CPU tiny tensors and an eight-item ARC smoke are not full-Qwen or 3060 evidence.

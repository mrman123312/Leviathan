# Strength-first evidence-conditioned cognition

## Scope and source of truth

This continues frozen Bedrock v3 (`e1780e47ed5d74a8c8f0e0adbae2d45be7862aed`). The unchanged donor, previous ARC-Easy runner, historical DeepSeek experiments, and L0-L10 research are retained. The user changed the immediate objective from speed/efficiency to **reasoning strength on ARC-AGI-1**, without authorizing neural training or multiple runtime models.

ARC-AGI-1 requires exact grid transformations inferred from visible demonstrations. ARC-Easy is a separate science multiple-choice dataset. Neither a 72% ARC-Easy result nor fitting ARC demonstrations means ARC-AGI-1 is solved. Public training tasks are development data; public evaluation is not private leaderboard verification. The original ARC repository describes three trials per query; this implementation reports the stricter pass@1 and pass@2, with all queries required for task success.

## Integrated implementation

`StrengthRuntime` extends the existing `BedrockRuntime`. It reuses one semantic identity, the existing executed cognitive graph, prediction records, memory store and frozen Qwen owner. Search, the interpreter, verification and memory are algorithms, not other LLMs. Two languages express hypotheses: a typed grid-operation AST and a bounded Python-shaped grid language. Qwen supplies programs and structural hints; a generic enumerator retains an unguided reserve. Search-only results are never called neural capability.

### 1. Counterexample-compiled recurrence

Each failed complete program receives a witness: the failing demonstration, output shape, first disagreeing cell, or interpreter error. These witnesses are re-encoded for neural revision. Duplicate complete programs are not re-evaluated. A wrong complete program may still be a useful prefix of a correct longer program; it is not globally forbidden as a subexpression.

Model proposals guide child/parent operation transitions, supply subexpressions, and seed bounded literal-hole, color-map and compositional repair. Failing to propose an operation never removes it from the generic grammar. Numeric/color fitting here is finite symbolic inference on permitted support examples, not gradient training. No table of known task IDs/solutions is imported.

### 2. Evidence-derived task slots and behavioral alternatives

Contrast activations are captured from correct source demonstrations versus explicit controls. This is a **proposed task-state intervention**, not a replication of Function Vectors and not a proof of semantic content. Up to four source slots are constructed without using the two reserved validation demonstration outputs. Every slot is tested separately.

Different programs are grouped by their actual predictions over visible examples and query inputs. Query outputs are unavailable. Diversity is therefore predictive rather than merely a hidden-vector distance. Agreement on a finite input set is not proof of global function equivalence. Distinct test-grid hypotheses are retained for up to two submissions.

### 3. Conditional cell interventions

The same donor FFN can be temporarily ablated at a contiguous intermediate-channel group. Tests establish real effect, restoration and no donor tensor mutation. Current group choices are coarse research probes, not discovered universal semantic brain regions.

Candidate interventions must improve output NLL on **both reserved demonstrations** before they can guide a proposal. NLL is a selection proxy, not proof of better exact-grid accuracy. The direct donor proposal is retained. Conditional cell effects and their support context can be stored with a skill, retrieved only under compatible revision/current demonstration fit, and retested before reuse.

### 4. Dual-form skill memory and exact macros

A skill candidate stores an executable program, activation-state metadata/vectors, intervention evidence, support hash, revision and dependencies. Demonstration fit creates an EPISODIC candidate, not an independently verified semantic fact. Promotion requires a host receipt bound to the exact payload. Counterevidence invalidates dependent records transitively.

Repeated subprograms across at least two distinct support hashes become macro candidates. Expansion is exact syntactic substitution. Both operands of a binary program are traversed. This is basic library extraction, not a full Stitch implementation or proof of lifelong learning. Current demonstrations must validate a retrieved executable procedure before it affects an answer. Evaluation tasks do not write reusable cross-task answer memory.

### 5. Evidence/no-progress scheduling

The controller records best demonstration error, new predictive behaviors and support-consistent candidates. An identical method/state is not requested repeatedly. It changes among raw-grid, object and difference representations, proposal repair, inverse-goal joins, and generic composition. These are transparent decision rules, not a calibrated expected-value learner. The immediate objective is accuracy; finite token, candidate and wall-time guards only bound runaway work.

## Further mechanisms introduced in this pass

**Inverse-goal junctions.** For a bijection B (rotation, reflection, transpose), compute B^-1 of visible output examples. When a prefix P matches those inverse goals, B(P(x)) matches the visible examples by construction. This is a task-conditioned forward/backward junction; it does not consult query labels and does not prove unseen generalization.

**Binary representation joins.** Competing transformed grids can become two operands: mask/pattern Kronecker products, object/canvas overlay, intersection, xor, or paint-mask. This permits compositions that a chain of unary transformations cannot express. Candidate dimensions, symbols and program sizes remain checked.

**Witness-constrained literal holes.** A plausible neural algorithm with a wrong literal is treated as a bounded sketch. Alternative observed colors/small integer parameters are tested. The original program is unchanged, every repaired program still faces all demonstrations, and repair provenance is explicit.

**Two hypothesis languages.** Familiar Python-shaped loops and comprehensions let Qwen express algorithms outside the finite named grid operations. The implementation does NOT call Python `exec`, `eval`, or `compile`. Its interpreter explicitly supports bounded assignments, loops, conditionals, list construction/comprehension, pure builtins, and a small interpreted list-method whitelist. No imports, arbitrary attributes, functions/recursion, files, network, shell, or unrestricted while loops are available. Unsupported programs are rejected, not silently translated into known solutions.

Nested comparisons, sorting, sequence growth and interpreter steps have explicit limits. This is a bounded language implementation, **not an adversarial OS sandbox or a formal proof covering every possible resource attack**. Model-generated code never receives the evaluator or an environment handle.

## Neural recurrence is not the default answer

Optional recurrence now uses damped residual integration in a middle-layer band:

`z0 = e + (F(e)-e)/K`

`z_next = z + (F(z)-z)/K`

`candidate = beta*F(e) + (1-beta)*z_final`

A final bounded update and nonfinite fallback preserve a safe donor alternative. This follows a published frozen-looping control rather than renaming arbitrary perturbations. No claim is made that a better numerical integrator necessarily makes better answers. It must earn use on visible support and then face untouched task queries.

The old transported routes remain historical experiments. Their projection saturation could make nominally different first probes coincide. The new method does not count different configuration names as evidence of different useful computations.

## Proof obligations

1. **Task boundary:** `ArcTask.from_public` rejects test entries containing outputs. The solver has no label argument. The trusted evaluator saves a sealed batch before opening answer files. This is code/data-flow separation, not cryptographic secrecy from the machine owner.
2. **Exact scoring:** dimensions and every cell must match; malformed grids and abstentions count wrong. Only the first two attempts count. All queries must succeed for whole-task success.
3. **Program consistency:** a selected synthesized program has executed against every visible demonstration. That is necessary, not sufficient, for generalization.
4. **Compositional repair:** rejecting P does not reject Q(P(x)). This preserves valid longer hypotheses that use imperfect prefixes.
5. **Inverse junction:** B(P(x))=y follows from P(x)=B^-1(y) on observed examples when B is a checked bijection.
6. **Frozen neural owner:** no new nn.Parameter or optimizer; all inference under frozen weights. Tiny tests compare state dictionaries. Full-model runtime version counters are mutation tripwires, not cryptographic weight hashes.
7. **Causal interventions:** fixed support-derived vectors and fixed cell interventions do not import future query-token labels. Prefix equivalence tests cover native tiny-Qwen paths. Recurrent altered routes never reuse stale per-depth KV state.
8. **Bounded activation:** a local relative-L2 projection bounds the local change in real arithmetic with rounding. It does not prove global logit accuracy or on-manifold state.
9. **Memory:** support-only evidence cannot promote itself to verified skill status; macros preserve exact AST expansion, not universal task applicability.

## Controls and attribution

The one-click run evaluates:

- **donor_grid:** the plain frozen Qwen predicts digit-row grids directly, two attempts, no search/interventions;
- **symbolic:** the same grid grammar/search, zero neural calls;
- **hybrid:** Qwen proposals plus the evidence-conditioned search and gated internal interventions;
- **raw_neural_programs:** first two parseable generated programs without demonstration-fit selection. These reuse already-paid proposals and are not an independent timing control.

The hybrid must beat the symbolic control to demonstrate useful neural contribution. Internal intervention gains require an additional ablation with the identical proposer/search and activation trials disabled. A combined-system gain is not automatically a new neural primitive. Synthetic tests and public development tasks do not demonstrate general intelligence.

## Execution

`RUN_ARC_AGI_1.bat` selects 24 fixed hash-ordered public evaluation tasks; `RUN_ARC_AGI_1_FULL.bat` selects all 400. Both use the existing v7 Windows CUDA environment and bundled pinned data, without installs, drive scanning, new model downloads or neural training. Strength settings allow depth-four search and up to 60,000 candidate evaluations per task. Numerical/CPU/native integration evidence is separate from the user's full pretrained GPU run.

`python scripts/run_strength_arc.py --mode symbolic --split training --limit 24 --depth 2 --beam 8 --max-candidates 6000 --task-seconds 20` is the smaller CPU development control. These are public TRAINING tasks, not hidden evaluation. Do not tune to individual evaluation outputs. Pretraining exposure to public ARC cannot be ruled out for the donor.

## Research sources (primary)

- ARC-AGI-1 task definition: https://arcprize.org/arc-agi/1
- Official data/README and license: https://github.com/fchollet/ARC-AGI at 399030444e0ab0cc8b4e199870fb20b863846f34
- Training-Free Looped Transformers: https://arxiv.org/abs/2605.23872 and official implementation https://github.com/L-z-Chen/Training-Free-Looped-Transformer
- Function Vectors in Large Language Models: https://arxiv.org/abs/2310.15213
- Narcissus (program-structured neural guidance): https://arxiv.org/abs/2608.25657
- Top-Down Synthesis for Library Learning (Stitch): https://arxiv.org/abs/2211.16605

The combinations here are engineering/research proposals, not claims of worldwide novelty. Published results for these papers are not Leviathan scores.

## Remaining gaps

Unrestricted representation invention, robust unfamiliar-world hypothesis generation, calibrated causal competence, autonomous long-term consolidation, expressive giant-scale MoP, verified GPU speedups, hybrid 27B support and L10 remain unfinished. The current system is stronger in executable hypothesis expressiveness, not yet proven stronger in broad neural intelligence. ARC-AGI-1 has not been beaten by this implementation.

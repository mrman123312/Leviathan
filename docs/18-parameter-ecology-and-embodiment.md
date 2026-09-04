# 18 — Parameter ecology and architecture embodiment

Leviathan now treats the 128-channel MoP tile as an **ancestral cell body**, not as the final sparse-compute primitive.

The goal is to evolve:

```text
DeepSeek routed expert
      ↓
exact 128-channel tiles
      ↓
parameterized cells
      ↓
confidence + abstention
      ↓
proposal messages
      ↓
sparse communication
      ↓
disagreement-controlled recruitment
      ↓
ephemeral local state
      ↓
verified coalitions
      ↓
transactional local plasticity
      ↓
grow / split / merge / prune
```

without discarding the inherited DeepSeek function.

## One model, not a civilization of models

A parameterized cell is **not** an agent and is not a miniature language model.

All cells remain inside one Leviathan neural system with:

- one global cognitive state;
- one parameter ownership system;
- one training objective;
- one output distribution/action stream;
- one governance and promotion boundary.

The analogy is a richer neural unit, not a committee of independent models.

## The insertion invariant

For inherited tile `i`:

```text
Cell_i(h) = Tile_i(h) + alpha_i * Refine_i(h, control_i)
```

with:

```text
alpha_i = 0
```

at insertion.

Therefore the new cell path is initially inert and the inherited tile output remains authoritative. The original V4 expert router and shared expert are also retained.

`src/leviathan/parameter_cells.py` implements this reference architecture for the packed Hugging Face DeepSeek V4 expert-bank representation. `install_parameter_cell_reference()` replaces routed expert banks with the cellized wrapper while leaving the parent router and shared expert path untouched; `restore_parameter_cell_reference()` reverses the insertion.

## The expressive membrane

A cell is the inherited 128-channel SwiGLU slice plus a cheap shared control membrane.

The membrane uses:

- a small per-cell embedding;
- low-dimensional state projection;
- four-dimensional confidence output;
- abstention output;
- proposal message output;
- associative recruitment query;
- low-rank residual refinement path;
- a scalar influence gate initialized to exactly zero.

The reference defaults are deliberately small:

```text
control      32
message      32
recruitment  32
confidence    4
local state  64 (reserved for the later ephemeral-state stage)
refine rank  16
```

This avoids the failure mode where every tile becomes a full mini-transformer.

## Confidence is observational first

The first learned cell addition is not permission to suppress computation.

Cells may estimate:

```text
epistemic confidence
aleatoric confidence
domain fit
evidence strength
abstention probability
```

but those values are logged before they are trusted for routing. Calibration must be demonstrated against real outcomes before abstention can remove inherited computation.

## Sparse communication

`SparseCellCommunication` implements a bounded one-round message exchange. Communication is restricted by group (normally token/state assignment) and a hard neighbor count. There is no global all-to-all exchange over the entire parameter reservoir.

This is the first executable primitive corresponding to the original idea that parameter groups should be able to "discuss" rather than merely sum independent outputs.

At insertion, communication still has **zero authority over the inherited residual path**.

## Disagreement controls additional compute

The reference controller has three actions:

```text
low disagreement    -> commit
medium disagreement -> communicate
high disagreement   -> recruit
```

The thresholds are explicitly experimental and must be calibrated. The important architectural property is the hard budget:

```text
seed cells                64
recruits per round        32
max active cells         256
max rounds                 2
max communication neighbors 8
```

These are reference values, not claims that 64/256 is optimal for V4.

Cell disagreement is exposed upward in `MetaState` as observational telemetry so L8 metacognition can eventually learn to trade capability gain against cell count, rounds, latency and hardware cost.

## Associative recruitment

`AssociativeCellRecruiter` gives cells a learned key/query address space. It can nominate relevant cells without making the initial seed router disappear.

The seed router remains the safe first activation mechanism. Cell-to-cell recruitment must remain inside a global budget and initially stays observational.

## Coalition compilation

`CoalitionRegistry` records groups of cells that repeatedly participate in **verified** success. A coalition cannot become a compilation candidate from one successful trajectory.

The default research threshold is:

```text
>= 8 verified trials
>= 90% verified success rate
```

Even then it is only a candidate. Kernel fusion requires an additional efficiency test because a stable coalition that is slower than the inherited V4 route is not a win.

This links L1.5 parameter ecology to L8 cognitive compilation:

```text
dynamic discovery -> repeated verified coalition -> shortcut route -> candidate fused kernel
```

## Local state and plasticity are deliberately later

Persistent cell self-modification is not enabled at insertion.

The order is:

1. ephemeral per-sequence/per-task state;
2. prove it helps and resets correctly;
3. transactional plastic overlay;
4. verify/replay/rollback;
5. only then consider promotion.

The ancestral pretrained tile is never rewritten directly from raw experience.

## MoP-0 through MoP-9

The canonical roadmap is machine-readable in `spec/parameter-cells.toml`:

| Stage | Meaning | Current architectural status |
|---|---|---|
| MoP-0 | exact inherited tiles | executable |
| MoP-1 | independent cross-expert tile routing | specified / next learned gate |
| MoP-2 | confidence + abstention | membrane executable, authority observational |
| MoP-3 | proposal messages | executable auxiliary signal |
| MoP-4 | one sparse communication round | executable primitive, zero authority at insertion |
| MoP-5 | disagreement-triggered recruitment | controller/recruiter primitive executable, not integrated into donor routing |
| MoP-6 | ephemeral local state | contract only |
| MoP-7 | learned coalitions | verified registry executable, neural shortcut not trained |
| MoP-8 | transactional local plasticity | governance contract only |
| MoP-9 | grow/split/merge/prune | contract only |

That distinction is intentional. We do not label a stage complete because the data structure exists.

## L0-L10 embodiment ledger

The architecture discussion also exposed a recurring problem: a layer can be beautifully specified while still being absent from the running model.

`spec/architecture-maturity.toml` and `src/leviathan/architecture_maturity.py` therefore enforce five separate gates for every layer:

```text
Specification
    ↓
Executable
    ↓
Integrated
    ↓
Learned
    ↓
Demonstrated
```

A layer is not called achieved until all five pass.

The current development order is deliberately **not** strictly bottom-to-top:

```text
L1 / L1.5
    ↓
L2
    ↓
L5
    ↓
L8
    ↓
L6
    ↓
L7
    ↓
L9
    ↓
L4 / L3
    ↓
L0
    ↓
L10
```

The reason is leverage. Once the parameter substrate can become adaptive, richer transformation, persistent state, metacognitive control, world modeling, verified learning and governed action are more likely to produce an intelligence discontinuity than prematurely replacing the tokenizer or building a custom hardware ISA.

L10 remains last because running cognition primarily in a new canonical latent is where the project begins to cross the pretrained-function-preservation wall.

## Acceptance remains brutal

Nothing in the parameter ecology weakens the R4 gates.

Before independent tile routing:

- full V4 checkpoint identity must be verified;
- donor vs MoP-0 logits/hidden states must match to the agreed precision envelope;
- ARC-Easy remains a canary;
- the 64 untouched WikiText passages remain protected;
- benchmark regression is not hidden in aggregates;
- the original V4 route remains a rollback path.

For later sparse cell stages:

- fewer active cells is not success by itself;
- theoretical FLOP reduction is not success by itself;
- wall-clock latency/throughput/HBM movement must improve;
- capability, retention, calibration and safety must not regress.

The intended end state is not merely "smaller experts." It is a **self-organizing sparse neural tissue** whose units are richer than ordinary parameter slices while still inheriting a proven pretrained semantic machine as its ancestor.

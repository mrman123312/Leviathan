# Leviathan Plyless Search Architecture

Status: experimental design contract. No strength claim is implied by this document.

## Mission

Stockfish uses `Depth` primarily as a ply-denominated computation budget, while its physical search stack is separately capped by `MAX_PLY = 246`.

Leviathan Plyless Search separates three concepts that ordinary depth conflates:

1. **Physical ply** — how many legal half-moves have actually occurred from the root. This remains exact and is still used for mate distance, repetition, rule-50 semantics, stack safety, and the hard 246-ply ceiling.
2. **Decision depth** — how much unresolved adversarial branching has been crossed. This is the scarce main-search budget.
3. **Proof depth** — how far a certified or tightly bounded continuation has been followed. Proof depth may continue after ordinary decision budget would have reached zero, but never beyond physical `MAX_PLY`.

The target is not fake displayed depth. The target is **more verified physical horizon per node without losing root-decision quality**.

## Core claim to test

A ply is not a unit of uncertainty.

A node with one legal move has zero move-choice entropy. A node with thirty plausible replies has much more. Charging both nodes exactly one full unit of logical depth wastes horizon on the first and can underfund the second.

Stockfish already partially compensates with extensions, reductions, pruning, TT cutoffs, qsearch, singular search, and move ordering. Plyless Search makes the underlying distinction explicit and persistent.

## Three-clock search

### Physical clock

- increments on every real move;
- never compressed;
- hard capped by `MAX_PLY`;
- controls mate-distance arithmetic and all path-dependent chess rules.

### Decision clock

The decision clock falls only when meaningful unresolved alternatives survive.

Initial experimental charge model:

- 1 surviving choice: near-zero charge;
- 2 surviving choices: fractional charge candidate;
- 3–4 surviving choices: intermediate charge candidate;
- broad unresolved node: full or greater charge.

Raw legal-move count is only a v0 proxy. The intended quantity is **surviving competitive width after cheap proof/bound filters**.

### Proof clock

A proof corridor is a sequence where alternatives are absent or have already received sufficient bounds to establish that they cannot affect the current alpha-beta decision.

Proof corridors may include:

- unique legal moves;
- unique check evasions;
- tablebase-exact transitions;
- exact TT continuations with path-safe context;
- forced mate/threat corridors;
- promotion races whose alternatives are fully bounded;
- future verified rival-elimination corridors.

The proof clock records how far the engine actually looked. It is not allowed to masquerade as full-width depth.

## Decision edges, not move edges

The long-term representation is a graph of **decision nodes** joined by macro-edges.

A macro-edge contains:

- start position identity plus required path context;
- exact move sequence;
- end decision-node identity;
- physical-ply length;
- rule-50/repetition effects;
- proof/bound type;
- validity/confidence conditions.

A corridor can then be traversed as one decision edge while still preserving every real chess move for legality and path semantics.

## Search information Stockfish currently discards

Candidates for persistent sidecar metadata:

- number of recursively searched children;
- number of alpha-raising rivals;
- reduced-search boundary distance;
- failed-null pressure;
- PV instability;
- history-context/repetition sensitivity;
- corridor length to next decision;
- proof type and bound slack.

Stockfish's ordinary TT preserves the best move, value/bound, depth and static evaluation, but not the **shape of uncertainty** that produced them. Plyless Search intends to reuse that shape on later iterative-deepening visits.

## Frontier behavior

Ordinary qsearch uses a stand-pat abstraction in quiet positions. Plyless Frontier experiments test whether a unique legal quiet move should pierce that horizon because passing is not a legal chess option.

Future frontier resolver:

1. reach decision budget zero;
2. classify frontier uncertainty;
3. traverse zero-width proof corridors;
4. run bounded threat/proof search where uncertainty remains small;
5. stop only when the frontier is both quiet and genuinely optional, or when a proof/node/physical-ply cap fires.

## Why this can approach 246 physical plies without pretending to brute-force chess

Exhaustively searching all chess branches to 246 plies is computationally infeasible. Plyless Search makes no such claim.

Instead, it tries to make 246 the **reachable certified physical horizon** along low-entropy/proven corridors while concentrating expensive branching work at actual decision points.

A future position might therefore have, for example:

- physical horizon: 120 plies;
- decision depth consumed: 28 units;
- proof corridor contribution: 92 plies.

That is meaningful only if game results show the extra horizon improves decisions.

## v0 experiments

- unique-legal-move decision-depth compression;
- unique quiet qsearch frontier traversal;
- branching distribution profiler;
- effective searched-width profiler;
- same-node strength tests and fixed-time tests.

## v1: Proven Decision Width

Replace raw legal count with surviving width after search evidence:

- moves actually searched recursively;
- alpha-raising rivals;
- exact/deep TT bounds;
- tablebase/proof specialists;
- failed-null pressure and boundary uncertainty.

Persist a compact uncertainty summary in a sidecar decision cache. On the next iterative-deepening visit, use it to allocate depth before paying to rediscover the same search shape.

## v2: Fractional decision budget

Move from integer ply depth to fixed-point decision units. Candidate conceptual charge:

`charge = f(surviving_width, bound_slack, volatility, path_context)`

The function must be learned/tuned only after deterministic versions establish causality. It must remain bounded and fail closed.

## v3: Decision DAG / corridor macros

Cache verified macro-edges from one decision node to the next. Reuse corridors across transpositions when all required path context matches. Re-expand immediately when a bound, repetition context, rule-50 state, or board identity no longer matches.

## v4: Dual search

Keep alpha-beta as the trusted verifier while a best-first uncertainty resolver spends computation on the frontier most likely to change the root choice. Alpha-beta supplies correctness pressure; the frontier resolver supplies non-uniform horizon.

## Scientific promotion gates

A Plyless component is promoted only if it survives all applicable gates:

1. **Containment:** disabled mode reproduces frozen Stockfish exactly.
2. **Equal-node strength:** >= 50% is the absolute screen floor; promotion requires a repeatable positive signal, not merely crossing 50% once.
3. **Fixed-time strength:** the mechanism must not buy quality with an impractical node/time explosion.
4. **Horizon evidence:** claimed deeper reach must be measured as physical/selective/proof reach, not marketing depth.
5. **Regression diversity:** opening, tactical, quiet, endgame, repetition and rule-50 families.
6. **Larger gate:** surviving screen candidates receive 100+ paired games, then longer controls.

## Speed track remains orthogonal

Lossless speed work is not allowed to change the search tree. Candidate rewrites must preserve exact node signatures before their speedup is counted. Search-selectivity changes are judged as strength changes, not speed optimizations.

## North-star metric

The eventual objective is not raw NPS or displayed depth.

**Useful verified horizon per unit time, converted into repeatable game strength.**

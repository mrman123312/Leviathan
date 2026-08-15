# Leviathan Immortality Architecture

## Mission

Project Leviathan is not only trying to produce a stronger static Stockfish fork. The long-range target is a chess research organism that can continue discovering, validating, remembering, and integrating stronger ways to allocate computation.

"Immortal" is an engineering metaphor, not a solved-chess claim. Chess is not currently solved; no finite unsolved engine can honestly guarantee that it will never lose. Leviathan's practical immortality target is therefore:

1. stay competitive with the strongest contemporary engine baseline;
2. automatically discover new search-control hypotheses;
3. preserve validated improvements and exact knowledge without silent regression;
4. detect when its current search/evaluation model is uncertain;
5. change *how it thinks* by position rather than using one immutable global search recipe;
6. accumulate verified knowledge across development generations;
7. remain able to roll back any descendant that fails empirical tests.

## RCE-derived design law

The key hidden assumption in the original Leviathan plan was: **one engine, one search algorithm, increasingly better parameters.**

The stronger architecture treats the search algorithm itself as a mutable object. The playing core remains extremely conservative and fast; a metacognitive control plane learns which computations deserve time; an offline Search Foundry invents and validates new operators; a persistent Chess Atlas stores transferable and exact knowledge.

The system is divided so experimental intelligence cannot silently corrupt the trusted playing core.

```text
                           LEVIATHAN
                               |
              +----------------+----------------+
              |                                 |
        PLAYING CORE                       IMMORTAL LAB
              |                                 |
      Stockfish alpha-beta              Search Foundry
      Stockfish value NNUE                   |
      policy prior                            +-- operator invention
              |                               +-- parameter/program search
              v                               +-- blind parent testing
      MetaSearch controller                   +-- promote/contest/retire
              |                               |
      +-------+---------+                     v
      |       |         |                validated descendants
  normal AB  proof   specialist                |
            probes     solvers                 |
      |       |         |                     v
      +-------+---------+--------------> Chess Atlas
              |                            |
              v                            +-- exact/proven islands
          best move                        +-- search-regret memory
                                           +-- reusable PV skills
                                           +-- position-family knowledge
                                           +-- experiment lineage
```

## Finalist A — MetaSearch: value of computation

### Primitive replacement

Conventional search asks: **Which move should be searched next?**

MetaSearch asks a higher-order question: **Which computation should be performed next because it is most likely to improve the final move decision per unit cost?**

A computation can be:

- deepen the current PV;
- search a rival root move;
- run a narrow verification window;
- re-search a reduced move at greater depth;
- invoke a tactical/proof specialist;
- obtain an expensive secondary evaluation;
- widen a bound;
- stop because further computation is unlikely to change the decision.

### Value-of-computation target

For a candidate computation c in search state z:

`VoC(c|z) = expected decision loss avoided - compute cost`

The initial implementation should not attempt a perfect theoretical VoC. It should learn proxies from deep teacher searches:

- probability the current best move changes with more search;
- expected centipawn/mate loss if search stops now;
- probability a reduced/pruned move later becomes best;
- expected narrowing of the best-vs-runner-up uncertainty gap;
- expected improvement in root decision confidence per node.

### Foundry transplant

Cognitive Foundry's metacognitive controller and `distinguishing_action` concept translate directly. In chess the external state is fully known; the uncertainty lies in **our computational knowledge of minimax value**. The distinguishing action becomes a search operation chosen because it separates competing hypotheses about the best move.

## Finalist B — Calibrated Search-Risk Budget

Aggressive selective search wins speed by sometimes being wrong. Today this risk is mostly encoded indirectly in tuned rules.

Leviathan should make the risk explicit.

For each contemplated reduction/prune, estimate:

`P(search decision is wrong | node, move, depth, history, bounds, policy, eval instability)`

Then maintain a bounded risk budget. A move may be heavily reduced when the predicted regret is tiny; if models disagree or uncertainty is high, Leviathan **abstains from pruning** and falls back to normal/full search.

This is the chess equivalent of Cognitive Foundry's high-confidence abstention: the engine still must move, but it never has to make an unsafe selective-search assumption.

Useful labels:

- shallow/reduced score versus deep score;
- whether a pruned candidate becomes PV under a deeper teacher;
- score regret caused by stopping at a lower budget;
- PV instability across iterative-deepening depths;
- disagreement among independent evaluators/search modes.

Hard rule: the learned layer may veto an aggressive reduction before it is ever allowed to invent an aggressive one.

## Finalist C — Search Foundry: self-evolving search operators

Cognitive Foundry's most valuable transplant is operator invention with held-out verification and lifecycle management.

Instead of hand-editing every search formula forever, define a restricted typed Search DSL whose programs can alter bounded search-control quantities.

Example inputs:

- depth;
- move count;
- improving flag;
- PV/cut/all-node type;
- history values;
- policy score;
- TT depth/bound;
- eval-beta margin;
- correction magnitude;
- check/capture status;
- uncertainty/regret features.

Example safe outputs:

- bounded LMR adjustment;
- bounded ordering bonus;
- verification threshold;
- specialist-router score;
- extra-search request.

Allowed primitives should initially be simple integer arithmetic, comparisons, min/max, clamps, piecewise conditionals, and small lookup tables. Programs are never trusted because they are elegant.

Lifecycle:

```text
repeated search-regret pattern
        |
        v
candidate operator/program
        |
        v
static safety + range checks
        |
        v
bench/NPS gate
        |
        v
held-out tactical/search-regret set
        |
        v
paired STC match
        |
        v
SPRT / LTC
        |
   +----+----+
   |         |
promote    reject
   |
observe future regressions
   |
active -> contested -> retired
```

No operator may directly modify legal move generation, position state, terminal evaluation, or verifier invariants.

## Finalist D — Chess Atlas: persistent verified memory

A normal engine forgets almost everything once a search ends. Leviathan should accumulate knowledge in several distinct stores.

### Episodic search memory

Store difficult searches and regret events:

- a move initially looked bad but became best after deep search;
- a prune/reduction caused a miss;
- an evaluation stayed unstable for many depths;
- a specialist solver resolved a position cheaply.

### Semantic position-family memory

Compress repeated episodes into transferable motifs/features rather than exact FEN lookup only.

Examples:

- fortress-like low-progress structures;
- opposite-colored bishop drawing mechanisms;
- king-net tactical volatility;
- zugzwang-sensitive endgames;
- exchange-sacrifice structures where static evaluation is systematically unstable.

These are hypotheses, not hardcoded chess truths. They survive only if held-out data demonstrates transfer.

### Procedural memory / PV skills

Learn reusable verified line proposals or "skills": short PV chunks that repeatedly compress search in structurally related positions.

A skill proposes; alpha-beta verifies.

### Exact knowledge islands

When a reachable subgraph is actually proven, retain it as exact knowledge with provenance and hash identity. Over generations the Atlas can grow tablebase-like islands around positions the engine repeatedly encounters or deliberately solves offline.

The Atlas is retrieval assistance, not permission to bypass verification when its match confidence is low.

## Finalist E — Explicit PV-chunk proposal

Strong neural chess systems can internally represent future-line information. Leviathan should test making this explicit and cheap: predict a 2–6 ply candidate sequence rather than only one move.

The sequence is never accepted directly. It is a move-ordering / aspiration hypothesis that alpha-beta verifies. Success metric is not sequence imitation accuracy but saved nodes and Elo.

This also implements the hierarchical-planning idea from Cognitive Foundry in a chess-native form: a short line or tactical objective is a reusable search skill, while exact alpha-beta remains the action verifier.

## Finalist F — Specialist search ecology

One algorithm need not dominate every topology. Candidate specialists include:

- normal Stockfish alpha-beta;
- tactical proof-number / AND-OR style search;
- local exact solver for small reversible endgames;
- DAG-oriented repeated-position search;
- forward/backward meet-in-the-middle tactical goal search;
- expensive secondary neural evaluator for rare uncertain nodes.

The novel part is not merely having several algorithms. A learned router selects them from evidence and all specialists exchange bounded evidence/bounds through a controlled interface.

Specialists must be ablatable. If a router cannot demonstrate marginal Elo after its inference cost, remove it.

## Finalist G — Goal-attractor / reverse tactical search

Cognitive Foundry's reverse-goal indexing suggests a chess-specific experiment.

For narrow tactical objectives such as forced mate, promotion, or specific material-winning endpoints, generate or retrieve predecessor constraints from the target and search forward from the current position simultaneously. Attempt meet-in-the-middle only when a cheap detector predicts a sufficiently constrained target topology.

This is a specialist, not the default search.

## Finalist H — Multi-hypothesis value model

Do not collapse all evaluation into one scalar confidence story.

Maintain cheap competing views, for example:

- standard NNUE value;
- short-horizon tactical value;
- deeper distilled value;
- bound/uncertainty head;
- optional specialist evaluator.

Disagreement is information. It can request more search even when every individual model looks confident.

Because chess is deterministic, the key uncertainty is epistemic/search uncertainty rather than aleatoric environment randomness.

## Outside-the-lattice result — the engine is not the unit of evolution

The deepest RCE escape is to stop treating a released binary as the enduring object.

The enduring object is a **lineage**:

```text
Leviathan generation N
   |
   +-- search traces
   +-- failures/regrets
   +-- exact proofs
   +-- trained models
   +-- operator ancestry
   +-- benchmark evidence
   |
   v
Search Foundry generates independent descendants
   |
origin-blind tests
   |
validated survivors/hybrids
   |
Leviathan generation N+1
```

This is the practical meaning of immortality: the lineage learns how to improve its own search machinery while never silently sacrificing a demonstrated capability.

## Transplants from Cognitive Foundry v109

### Keep

1. **Competing hypotheses** -> rival PV/value/search-mode hypotheses.
2. **Distinguishing action / information gain** -> choose the next computation that best resolves root uncertainty.
3. **Metacognitive controller** -> decide deepen / verify / specialist / stop.
4. **High-confidence abstention** -> do not prune/reduce when uncertainty is unresolved.
5. **Hierarchical verified planning** -> short PV skills/subgoals proposed cheaply and verified by alpha-beta.
6. **Operator invention** -> synthesize bounded search-control programs, then held-out + game verification.
7. **Active / contested / retired lifecycle** -> search-operator registry.
8. **Episodic / semantic / procedural memory** -> search episodes, transferable motifs, PV/operator skills.
9. **Counterexample-triggered refit + full replay** -> retrain a candidate after a discovered failure and rerun all protected regressions.
10. **Bounded indexes / reverse-goal indexes** -> fast Atlas retrieval and tactical goal specialists.
11. **Merkle/content-addressed internal artifacts** -> reproducible datasets, models, proof islands, experiment provenance.
12. **Causal ablation discipline** -> every claimed organ must lose its claimed capability when removed.

### Do not transplant directly

- general symbolic world-state learning: chess rules and board state are already exact;
- broad language/multimodal modules: irrelevant to playing strength;
- exhaustive general BFS: inferior to chess-specialized search at scale;
- aleatoric world uncertainty: standard chess is deterministic;
- unrestricted runtime DSL invention: too risky and too slow in the hot path;
- generic safety/permission machinery: keep research guardrails outside the node loop.

## Research order

### P002 — Search-regret dataset

Create low-budget and high-budget teacher traces and measure how often early computation would have selected the wrong move or wrong confidence.

### P003 — Stop/deepen predictor

At root only, predict whether additional compute is likely to change the selected move enough to matter. This is the cheapest MetaSearch proof-of-concept.

### P004 — Selective-search regret model

Instrument reduced/researched moves and learn a veto signal for dangerous LMR/pruning decisions.

### P005 — Risk-budget LMR

Use calibrated regret only to make reductions *less aggressive* first. After positive evidence, permit bounded extra reductions for very low-risk cases.

### P006 — PV-skill proposals

Train explicit short-line proposals; use them only for move ordering/aspiration until positive Elo is demonstrated.

### P007 — Search DSL / operator foundry

Allow offline synthesis of bounded search formulas. Every proposal receives full parent-relative empirical testing.

### P008 — Specialist router

Add one specialist at a time, beginning with a tactical proof search or expensive evaluator for high-uncertainty roots.

### P009 — Chess Atlas

Persist search-regret episodes and exact solved islands; test retrieval on held-out related positions before allowing live influence.

### P010 — Lineage automation

Automate candidate creation, protected regression, NPS gates, paired matches, promotion, contest, retirement, and rollback.

## Non-negotiable evidence laws

1. Equal-compute match claims remain the ultimate playing-strength criterion.
2. A clever architecture receives zero Elo credit until games show it.
3. Each experimental organ must be switchable and causally ablatable.
4. Learned control first receives veto/ordering authority before pruning authority.
5. Uncertainty must be calibrated against deeper-search outcomes.
6. Search Foundry candidates are tested on unseen openings/position sets and multiple time controls.
7. No descendant silently deletes the parent baseline or experiment record.
8. Exact knowledge and learned guesses are stored separately.
9. A failing operator is retired, not rationalized.
10. The live engine remains simpler than the research organism around it.

## Long-range victory condition

Leviathan is stronger than Stockfish only when the playing core wins a statistically valid, equal-resource comparison against a contemporaneous Stockfish master.

Leviathan is **immortal** only in the stronger engineering sense when its research lineage can repeatedly generate and validate descendants, preserve exact knowledge and proven capabilities, identify its own computational uncertainty, and improve without depending on one fixed human-designed search recipe.

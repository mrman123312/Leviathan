# Leviathan Donor Matrix v1

The donor pool is not a shopping list. Each donor earns a niche. Agreement across independent engines is evidence that an idea is mature; disagreement exposes design space; an outside-lattice donor exists to challenge the shared assumptions of the alpha-beta family.

## Donor roles

| Donor | Primary reason it exists in the pool | Direct source policy |
|---|---|---|
| Stockfish | strongest control/oracle; mature search/NNUE/TT/SMP implementation | GPLv3 allowed |
| PlentyChess | threat-input NNUE, fractional-depth experiments, modern top-tier independent choices | GPLv3 allowed |
| Obsidian | independent top-tier C++ search/NNUE/PGO implementation | GPLv3 allowed |
| Berserk | compact mechanism microscope for pruning, ordering, LMR, extensions, NNUE | GPLv3 allowed |
| Alexandria | modern OpenBench/Bullet lineage and alternate bitboard/search implementation | GPLv3 allowed |
| Koivisto | historically influential independent NNUE/search lineage | GPLv3 allowed |
| Ethereal | readable mature search and NNUE accumulator implementation | GPLv3-or-later allowed |
| Seer | modular worker/orchestrator/move-orderer/TT architecture | GPLv3 allowed |
| RubiChess | long-running independent engine and CPU-optimization comparator | GPLv3 allowed |
| Stormphrax | independent NNUE/data lineage and modern Chess960-aware engine | GPLv3 allowed |
| Caissa | clean-sheet MIT engine, custom neural runtime/trainer/cache/NUMA design | MIT allowed |
| Arasan | decades-old independent architecture, portability, tests, protocols | MIT allowed |
| Patricia | objective/style shaping and specialized training/evaluation | MIT allowed |
| Lc0 | outside-the-lattice MCTS/DAG + policy/value + GPU search ontology | GPLv3 allowed |
| Reckless | current elite Rust engine and independent search/layout reference | AGPL reference-only |
| Viridithas current | original-data Rust lineage and independent NNUE/search ideas | AGPL reference-only |
| Viridithas pre-20 | permissive historical Rust lineage | MIT, version-gated |
| Pawnocchio | Zig implementation perspective and alternate low-level layout | GPLv3 code only |

## Subsystem harvest map

| Leviathan subsystem | Primary donors | What to extract | Required comparison |
|---|---|---|---|
| Rules / move generation | Stockfish, Caissa, Arasan, RubiChess, Pawnocchio | board representation, attack generation, make/unmake, castling/FRC handling | legal-move differential + perft corpus; speed after correctness |
| Repetition / draw truth | Stockfish, Caissa, Arasan, Viridithas | repetition representation, cuckoo/path techniques, rule-50 handling | adversarial repeated-path corpus; identical-board/different-history tests |
| Position identity | Stockfish + Leviathan native | Zobrist/canonical identity mechanics | prove board identity separately from search-proof identity |
| Transposition table | Stockfish, Berserk, Seer, Caissa, Reckless(reference) | packing, replacement, prefetch, concurrency, bound semantics | fixed-hash A/B; collision stress; path-sensitivity ablation |
| Evidence-aware TT | Leviathan native, informed by all TT donors | evidence/debt/provenance fields and trust update | TT-on/off, provenance-on/off, equal-node regret tests |
| Static evaluation runtime | Stockfish, Ethereal, Seer, PlentyChess, Caissa, Obsidian | accumulator layout, cache behavior, SIMD, feature refresh | exact output validation + evals/sec + search Elo at fixed evaluator |
| Evaluation architecture | PlentyChess, Caissa, Lc0, Patricia, Viridithas | threat inputs, custom NNUE, policy/value split, style-specialized objectives | same-search evaluator swap; calibration and tactical regret |
| Frozen strong evaluator | Stockfish networks CC0 | immediate strong chess knowledge without retraining | frozen-net compatibility and search isolation |
| Network training | nnue-pytorch, Caissa, PlentyChess, Viridithas, Alexandria | sparse loaders, self-play, self-distillation, progressive data, Bullet workflows | tiny reproducible training runs before large compute |
| Move ordering | Stockfish, Berserk, Seer, PlentyChess, Reckless(reference) | history families, continuation/countermove, TT priority, threat-aware ordering | ordering-only A/B at fixed node budget |
| Search reductions | Stockfish, Berserk, PlentyChess, Obsidian | LMR and fractional-depth/reduction policies | equal-node + fixed-time + tactical holdout; interaction tests |
| Forward pruning | Stockfish, Berserk, Obsidian, Alexandria | NMP, RFP, futility, LMP, ProbCut, SEE pruning | one-mechanism-at-a-time A/B; deep-oracle false-prune regret |
| Extensions | Stockfish, Berserk, Obsidian | singular/check/recapture-style extension concepts | extension-only A/B; search explosion guard |
| Search topology | Stockfish baseline, Lc0 outside-lattice, Leviathan native | alpha-beta control vs MCTS/DAG vs proof-budget/candidate-set search | same evaluator + same hardware + fixed-time/fixed-node batteries |
| Candidate-set scheduler | Leviathan native, informed by Lc0 batching/policy and alpha-beta move ordering | allocate effort by unresolved decision value rather than serial rank | compare against conventional ordering under equal total work |
| Uncertainty / volatility | Leviathan native, Lc0 policy/value separation, Patricia objective shaping | calibrated uncertainty and tactical volatility as scheduler inputs | calibration curves; remove-head ablations; regret reduction |
| SMP / threading / NUMA | Stockfish, Caissa, PlentyChess, Reckless(reference) | thread ownership, TT sharing, NUMA pinning, scaling | 1/2/4/8/16-thread scaling; deterministic single-thread baseline |
| Time management | Stockfish, Berserk, Reckless(reference), Arasan | reserve, instability response, fail-high/low adaptation | cyclical/no-increment/adversarial time controls |
| Tablebases | Fathom, Stockfish, Ethereal, Arasan | standalone Syzygy probing interface | tablebase conformance; no contamination of core board representation |
| UCI / harness | Arasan, Stockfish, fastchess | protocol robustness, options, compliance | fastchess UCI compliance + command transcript tests |
| Tournament / SPRT | fastchess, OpenBench, Fishtest methodology | paired openings, SPRT, fixed-game matches, telemetry | A/A calibration before every serious A/B campaign |
| Behavioral/style objective | Patricia | non-Elo objective, filtered training data, sacrifice/aggression metrics | strength/style Pareto frontier, never style-only selection |
| GPU neural search | Lc0 | batching, cache, backend abstraction, policy/value inference | CPU control vs GPU path; backend-independent search contract |
| Architecture modularity | Seer, Caissa, Lc0 | worker/orchestrator/backend boundaries | replace one subsystem without touching unrelated search code |

## Harvest order

### Phase A — correctness donors

1. Fathom for standalone tablebase boundary.
2. Arasan/Caissa/Stockfish as differential rules references.
3. Build a generated-position legal-move and perft corpus.
4. Freeze deterministic transcript fingerprints.

No strength mechanism may mask a rules bug.

### Phase B — strong evaluation without a server room

1. Add an evaluator plug-in boundary.
2. Integrate a frozen CC0 Stockfish network through a compatible adapter.
3. Preserve the native baseline evaluator as the null condition.
4. Add at least one independent evaluator later (Viridithas CC0 net if format/runtime can be cleanly isolated, or a donor-specific adapter).

Search experiments should initially keep the evaluator fixed.

### Phase C — conventional-search reconstruction

Rebuild the strongest compact alpha-beta control from individually tested mechanisms. Do not port Stockfish search.cpp. Candidate mechanisms enter one at a time from Berserk/Ethereal/Seer/PlentyChess/Stockfish/Obsidian.

Order:

1. fixed-capacity TT;
2. iterative deepening/PVS/aspiration;
3. move-order histories;
4. SEE;
5. NMP;
6. RFP/futility/LMP;
7. LMR;
8. ProbCut;
9. singular/other extensions;
10. SMP/time management.

At every step keep a rollbackable baseline and run interaction tests where mechanisms are known to couple.

### Phase D — Leviathan-native divergence

Once conventional strength is recovered:

- Evidence-aware TT;
- path-sensitive proof identity;
- structured value/uncertainty/volatility/provenance evaluation;
- proof debt;
- candidate-set search;
- information-value scheduler;
- persistent game-level proof memory;
- alternative search representations.

### Phase E — outside-the-lattice

Use Lc0 as a counterfactual architecture, not a donor for cosmetic features. Compare:

- tree vs DAG;
- scalar move ordering vs policy allocation;
- depth vs visit/proof budget;
- leaf evaluation vs batched neural inference;
- exact alpha-beta bounds vs probabilistic/value evidence;
- transient node search vs persistent search graph.

The question is not "how do we add MCTS to Stockfish?" It is "which assumptions are common to every alpha-beta donor, and can Leviathan remove the need for them?"

## Promotion gate for an imported mechanism

A donor mechanism becomes active Leviathan architecture only after:

1. exact provenance and immutable source revision are recorded;
2. the imported/adapted code passes unit/correctness tests;
3. a null or simpler alternative is tested;
4. isolated A/B shows useful effect;
5. fixed-node and fixed-time results agree enough to understand the mechanism;
6. tactical/deep-oracle regret does not materially worsen;
7. interaction with existing mechanisms is measured where plausible;
8. profiling shows the cost is understood;
9. fresh holdout/paired-game evidence survives;
10. rollback remains trivial.

A copied mechanism has no prestige advantage over a Leviathan-native one.

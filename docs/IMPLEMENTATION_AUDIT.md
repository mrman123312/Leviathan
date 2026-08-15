# Leviathan Immortal v1 — Implementation Audit

This audit exists to prevent architecture-by-description. An item counts as implemented only when it has an executable code path or artifact pipeline, an authority boundary, and a test/evidence route.

## Trusted parent

- Frozen parent: `5062aee519a1ba262d472d8ab139851ced56573e`
- Frozen AVX2 bench: `2884956` nodes.
- Default Leviathan organs are disabled / authority zero.
- CI verifies default parent signature.
- CI also loads neutral Policy + MetaSearch + Risk + Atlas + Search DSL simultaneously and requires the same node-count signature.

## P001 — policy ordering

**Runtime:** `src/leviathan_policy.h`, `src/movepick.cpp`

**Data/training:** `trainer/generate_dataset.py`, `trainer/model.py`, `trainer/train.py`, `trainer/export.py`, `trainer/split_dataset.py`

**Authority:** quiet-move ordering only.

**Format:** `LVTP1`.

**Status:** implemented, untrained/unproven for strength.

## P002 — search regret

**Root counterfactual data:** `trainer/generate_metasearch_dataset.py`.

**Native selective-search traces:** `src/leviathan_trace.h`, hooked from LMR in `src/search.cpp`.

**Authority:** observation only; tracing never alters search.

**Status:** implemented.

## P003 — MetaSearch / value of computation

**Runtime:** `src/leviathan_control.h`, root-time hook in `src/search.cpp`.

**Training:** `trainer/train_control.py --task meta`.

**Authority 1:** may buy additional root time only.

**Authority 2:** bounded bidirectional time adjustment.

**Format:** `LVTM1`.

**Status:** implemented, model quality unproven.

## P004/P005 — selective-search regret and risk-aware LMR

**Runtime:** `src/leviathan_control.h`, LMR hook in `src/search.cpp`.

**Training:** `trainer/train_control.py --task risk` using the same 12 live features emitted by tracing.

**Authority 1:** veto/reduce aggressive reduction only.

**Authority 2:** may also add a small bounded reduction at exceptionally low predicted risk.

**Format:** `LVTR1`.

**Status:** implemented, calibration/Elo unproven.

## P006 — explicit short PV skills

**Generator/evaluator:** `trainer/pv_skills.py`.

Capabilities:

- generate 2–6-ply teacher PV proposals;
- preserve multiple ranked proposals;
- compare proposed first move, matching-prefix length and score regret against deeper teacher search;
- pack the complete line into content-addressed `LVPS1` artifacts.

**Live authority:** deliberately not expanded into sequence-following. Verified first-move evidence may be converted into Atlas/order evidence while alpha-beta remains authoritative. A future sequence-aware hot-loop consumer must first prove saved nodes/Elo exceed its cost.

**Status:** complete as a proposal/data/artifact organ; live multi-ply authority intentionally withheld by the evidence law.

## P007 — Search Foundry and bounded Search DSL

**Runtime:** `src/leviathan_dsl.h`, LMR hook in `src/search.cpp`.

**Foundry:** `foundry/foundry.py`.

Capabilities:

- generate independent programs;
- mutate parent programs;
- preserve candidate ancestry and SHA-256 identity;
- syntax/type/range test candidates;
- bound all runtime output;
- screen evidence without runtime code generation.

**Authority 1:** positive/more-aggressive reductions are suppressed.

**Authority 2:** bounded bidirectional LMR adjustment.

**Format:** `LVSD1`.

**Status:** implemented.

## P008 — specialist ecology

**Live router:** `src/leviathan_control.h` exposes an independently switchable tactical verification specialist path that buys depth back for high-risk volatile checks/captures/PV moves.

**Independent proof specialist:** `specialists/mate_prover.py` implements bounded memoized AND/OR forced-mate proof search.

**Authority:** no specialist may replace legal move generation or terminal correctness. Offline specialists must demonstrate marginal value before hot-loop promotion.

**Status:** implemented with one live conservative route and one independent proof specialist.

## P009 — Chess Atlas

**Runtime:** `src/leviathan_atlas.h`, ordering hook in `src/movepick.cpp`.

**Builder/verifier:** `atlas/atlas_tool.py`.

Stores:

- engine-native position key;
- engine-native raw move;
- confidence;
- ordering evidence;
- provenance class (`episode`, `skill`, `exact`).

Artifacts receive SHA-256 manifests. Learned hints and exact/proven knowledge remain distinct. Even exact entries are verified by alpha-beta in v1.

**Format:** `LVTA1`.

**Status:** implemented.

## P010 — automated lineage

**Candidate generation:** `foundry/foundry.py`.

**Mechanical lifecycle:** `foundry/lineage.py`, format `LVLINE1`.

Lifecycle states/actions include:

- PROVISIONAL;
- SCREENED;
- ACTIVE;
- CONTESTED;
- REJECTED;
- RETIRED;
- explicit rollback to a registered parent.

Final ACTIVE promotion requires supplied evidence for compile, parent signature, positive Elo, STC, LTC and multi-hardware gates. Screening evidence alone cannot promote a candidate.

**Experiment workflow:** `.github/workflows/immortal-lineage.yml` creates a candidate, builds one binary, runs candidate/parent as the same executable with different UCI settings, reverses colors, performs wall-clock screening matches, records SPRT output and archives artifacts.

**Status:** implemented. Automatic code merging is intentionally excluded; evidence promotes lineage artifacts, not unreviewed source.

## P011 — reverse goal-attractor specialist

**Runtime/research specialist:** `specialists/goal_attractor.py`.

Instead of unsafe unrestricted retrograde chess move generation, it:

1. consumes already solved/proposed tactical lines;
2. indexes every intermediate position by its suffix toward the goal (`LVGA1`);
3. performs bounded forward DAG/BFS search from a new position;
4. returns a bridge when forward search intersects the reverse goal index;
5. marks every returned line `verification_required`.

This implements reverse-goal indexing / meet-in-the-middle proposal search without mislabeling an indexed PV suffix as a proof.

**Status:** implemented as a proposal specialist.

## P012 — multi-hypothesis uncertainty

`src/leviathan_control.h` supports independently bootstrapped 1–4-head linear ensembles. MetaSearch and risk control preserve the head spread rather than averaging disagreement away. Spread is added to search uncertainty and may request more computation.

**Status:** implemented mechanically. It remains epistemic search uncertainty, not a substitute chess evaluation.

## Match and statistics infrastructure

- `tools/run_match.py`: paired UCI matches, reversed colors, same-binary A/B option files, wall-clock or fixed-node resources.
- `tools/sprt.py`: lightweight W/D/L screening SPRT only.
- Final claims still require proper paired/pentanomial STC, LTC and multi-hardware validation.

## CI / regression protection

`.github/workflows/immortal-bootstrap.yml` performs:

1. Python syntax checks for all major organs;
2. guarded/idempotent Stockfish integration check;
3. NNUE download;
4. AVX2 build;
5. UCI option verification;
6. exact frozen parent bench check;
7. neutral artifacts for every live organ;
8. simultaneous neutral-all-organs node-equivalence test;
9. Foundry, Atlas, PV-skill, lineage and statistics smoke tests.

`tools/bootstrap_immortal.py` contains exact source anchors and fails closed if future Stockfish source moves beneath the integration rather than guessing a patch location.

## What is intentionally not claimed

The code architecture is complete for Immortal v1. The following empirical products are **not** fabricated:

- a strong trained non-neutral policy;
- a calibrated non-neutral MetaSearch model;
- a calibrated non-neutral risk model;
- a Search-DSL candidate with positive Elo;
- a populated production Chess Atlas;
- a demonstrated specialist Elo gain;
- a statistical win over contemporary Stockfish master.

Those are outputs of the implemented research organism, not source files that can honestly be invented without running the required datasets, training and games.

## Release criterion for “implementation complete”

Immortal v1 implementation is releasable when the final branch CI is green with the integrated Stockfish source already committed (bootstrap reports no source patch required), the frozen parent bench is unchanged, neutral all-organ equivalence passes, and all listed tooling smoke tests pass.

That release criterion is separate from the much harder **strength criterion**: statistically beat contemporary Stockfish master under equal compute.

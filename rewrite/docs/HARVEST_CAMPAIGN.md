# Leviathan Donor Harvest Campaign

## Goal

Recover mature open-source chess-engine knowledge inside the greenfield rewrite **without re-creating Stockfish's architecture** and without requiring full retraining from zero.

Every donor mechanism must enter through an explicit interface, preserve provenance, and survive Leviathan's own tests.

## Gate 0 — supply chain / legality

- [x] donor registry with code-vs-model/data licenses
- [x] current AGPL donors reference-only under present policy
- [x] pre-20 Viridithas version-gated MIT path
- [x] provenance sidecars for committed imports
- [x] source locks with immutable revisions and Git blob hashes
- [x] pretrained-model registry with SHA-256 filename verification
- [x] CI donor/model/source-lock audit

## Gate 1 — correctness donors

- [x] independent Leviathan rules/FEN/legal move engine
- [x] startpos perft 1-4
- [x] castling/en-passant/promotion/FEN round-trip tests
- [x] pin Fathom as standalone MIT Syzygy donor
- [x] optional Fathom tablebase interface + build path
- [ ] generated-position differential legal-move corpus vs Stockfish/Caissa/Arasan
- [ ] adversarial repetition and identical-board/different-history corpus
- [ ] Chess960/FRC castling corpus
- [ ] deterministic transcript fingerprints

## Gate 2 — strong evaluator without a server room

- [x] evaluator plug-in interface
- [x] pin Stockfish 18 big/small CC0 networks
- [x] pin post-SF18 research control network
- [ ] implement isolated Stockfish-NNUE-compatible runtime adapter or select cleaner compatible donor runtime
- [ ] exact-eval cross-check against matching Stockfish build
- [ ] evals/sec benchmark
- [ ] second independent evaluator adapter for A/B calibration

## Gate 3 — compact conventional control search

Import/reimplement one mechanism at a time; no monolithic `search.cpp` transplant.

- [ ] fixed-capacity TT (Stockfish/Berserk/Seer/Caissa comparison)
- [ ] PVS + aspiration
- [ ] SEE
- [ ] history/continuation/countermove ordering
- [ ] null-move pruning
- [ ] reverse futility / futility / late-move pruning
- [ ] LMR and PlentyChess fractional-depth alternative
- [ ] ProbCut
- [ ] singular/other extensions
- [ ] SMP/NUMA
- [ ] mature time manager

For each: null branch, isolated A/B, A+B interaction where coupled, fixed-node, fixed-time, deep-oracle/tactical regret, profiler, paired games, rollback.

## Gate 4 — Leviathan-native search

- [ ] evidence-aware TT
- [ ] calibrated proof debt
- [ ] structured value/uncertainty/volatility/provenance heads
- [ ] candidate-set search
- [ ] information-value computation scheduler
- [ ] persistent game-level proof memory
- [ ] equal-work comparison against conventional alpha-beta control

## Gate 5 — outside-the-lattice

Use Lc0 as a counterfactual architecture:

- [ ] tree vs DAG
- [ ] serial move ordering vs policy allocation
- [ ] depth vs visit/proof budget
- [ ] scalar leaf evaluation vs policy/value evidence
- [ ] transient tree vs persistent graph
- [ ] batched inference vs node-local inference

Question: **what assumption do all alpha-beta donors share, why does it exist, and can Leviathan remove the need for it?**

## Donor priority

**Critical:** Stockfish, PlentyChess, Berserk, Caissa, Lc0, Reckless (reference-only).

**High:** Obsidian, Alexandria, Ethereal, Seer, Stormphrax, Patricia, Viridithas.

**Supporting:** Koivisto, RubiChess, Arasan, Pawnocchio, Fathom, OpenBench, fastchess, nnue-pytorch.

## Promotion rule

No imported mechanism becomes Leviathan architecture because its donor is strong. It becomes Leviathan architecture only after it demonstrates decision-relevant value under the rewrite's own evidence gates.

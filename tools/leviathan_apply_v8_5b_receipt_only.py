#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one anchor, found {n}")
    return text.replace(old, new, 1)

hp = Path('src/search.h')
h = hp.read_text()
h = replace_once(h,
'''    struct LeviathanProofMemoryEntry {
        Key          key     = 0;
        unsigned int debt    = 0;
        Move         witness = Move::none();
    };
''',
'''    struct LeviathanProofMemoryEntry {
        Key          key             = 0;
        unsigned int debt            = 0;
        Move         witness         = Move::none();
        bool         witnessWasLmr   = false;
        unsigned int witnessDepth    = 0;
    };
''','entry')

h = replace_once(h,
'''    Move leviathan_proof_memory_witness(Key key) const {
        const auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        return e.key == key ? e.witness : Move::none();
    }

    void leviathan_proof_memory_store(Key key, int debt, Move witness = Move::none()) {
        if (debt < 3 && !witness)
            return;
        auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        const unsigned boundedDebt = unsigned(std::clamp(debt, 0, 5));
        if (e.key != key)
            e = {key, boundedDebt, witness};
        else
        {
            e.debt = std::max(e.debt, boundedDebt);
            if (witness)
                e.witness = witness;
        }
    }
''',
'''    Move leviathan_proof_memory_witness(Key key) const {
        const auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        return e.key == key ? e.witness : Move::none();
    }

    bool leviathan_proof_memory_witness_was_lmr(Key key) const {
        const auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        return e.key == key && e.witnessWasLmr;
    }

    unsigned int leviathan_proof_memory_witness_depth(Key key) const {
        const auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        return e.key == key ? e.witnessDepth : 0;
    }

    void leviathan_proof_memory_store(Key key,
                                      int debt,
                                      Move witness = Move::none(),
                                      bool witnessWasLmr = false,
                                      int witnessDepth = 0) {
        if (debt < 3 && !witness)
            return;
        auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        const unsigned boundedDebt = unsigned(std::clamp(debt, 0, 5));
        const unsigned boundedWitnessDepth = unsigned(std::max(0, witnessDepth));
        if (e.key != key)
            e = {key, boundedDebt, witness, witness && witnessWasLmr,
                 witness && witnessWasLmr ? boundedWitnessDepth : 0U};
        else
        {
            e.debt = std::max(e.debt, boundedDebt);
            if (witness)
            {
                if (e.witness != witness)
                {
                    e.witness       = witness;
                    e.witnessWasLmr = witnessWasLmr;
                    e.witnessDepth  = witnessWasLmr ? boundedWitnessDepth : 0U;
                }
                else if (witnessWasLmr)
                {
                    e.witnessWasLmr = true;
                    e.witnessDepth  = std::max(e.witnessDepth, boundedWitnessDepth);
                }
            }
        }
    }
''','helpers')
hp.write_text(h)

sp = Path('src/search.cpp')
s = sp.read_text()
# IMPORTANT: preserve V7.4 decay block exactly. No stale-witness hygiene here.

s = replace_once(s,
'''    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    PVMoves   pv;
''',
'''    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));
    const Depth leviathanEntryDepth = depth;

    PVMoves   pv;
''','entry depth')

s = replace_once(s,
'''    bool  capture, ttCapture, leviathanNullFragile;
    int   priorReduction, leviathanProofDebt;
''',
'''    bool  capture, ttCapture, leviathanNullFragile, leviathanWitnessWasLmr;
    int   priorReduction, leviathanProofDebt, leviathanWitnessCertifiedDepth;
''','locals')

s = replace_once(s,
'''    leviathanProofDebt   = 0;
    leviathanWitness     = Move::none();
    priorReduction        = (ss - 1)->reduction;
''',
'''    leviathanProofDebt   = 0;
    leviathanWitness     = Move::none();
    leviathanWitnessWasLmr = false;
    leviathanWitnessCertifiedDepth = 0;
    priorReduction        = (ss - 1)->reduction;
''','init')

s = replace_once(s,
'''    leviathanWitness = leviathan_proof_memory_witness(posKey);
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);
''',
'''    leviathanWitness = leviathan_proof_memory_witness(posKey);
    leviathanWitnessWasLmr = leviathan_proof_memory_witness_was_lmr(posKey);
    leviathanWitnessCertifiedDepth = int(leviathan_proof_memory_witness_depth(posKey));
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);
''','load')

s = replace_once(s,
'''    if (probCutNearMiss)
    {
        leviathanProofDebt = std::min(5, leviathanProofDebt + 1);
        leviathanWitness   = probCutNearMiss;
    }
''',
'''    if (probCutNearMiss)
    {
        leviathanProofDebt = std::min(5, leviathanProofDebt + 1);
        if (leviathanWitness != probCutNearMiss)
        {
            leviathanWitnessWasLmr = false;
            leviathanWitnessCertifiedDepth = 0;
        }
        leviathanWitness = probCutNearMiss;
    }
''','probcut identity')

anchor = '''        // Leviathan strength v6.4 - Proof Regime. Once several independent
        // warnings agree, incremental buybacks are not enough: accumulated LMR
'''
code = '''        // V8.5B receipt-only ablation. Preserve V7.4 memory decay semantics;
        // the only added authority is a depth-monotonic re-proof for an exact move
        // that previously demonstrated a serious LMR error.
        const bool leviathanWitnessNeedsFullProof =
          move == persistentWitness && leviathanWitnessWasLmr && (PvNode || ss->ttPv)
          && leviathanEntryDepth >= 7
          && (leviathanWitnessCertifiedDepth == 0
              || leviathanEntryDepth >= leviathanWitnessCertifiedDepth + 2);
        if (leviathanWitnessNeedsFullProof)
        {
            newDepth = std::max(newDepth, leviathanEntryDepth - 1);
            r = std::min(r, 0);
        }

'''
if s.count(anchor) != 1:
    raise SystemExit(f'proof anchor count {s.count(anchor)}')
s = s.replace(anchor, code + anchor, 1)

s = replace_once(s,
'''                if (leviathanReducedValue <= alpha && value > alpha)
                    leviathanWitness = move;
''',
'''                if (leviathanReducedValue <= alpha && value > alpha)
                {
                    if (leviathanWitness != move)
                        leviathanWitnessCertifiedDepth = 0;
                    leviathanWitness       = move;
                    leviathanWitnessWasLmr = true;
                }
''','margin skeptic receipt')

s = replace_once(s,
'''            if (boundaryFlip || lmrError >= 160)
                leviathanWitness = move;
''',
'''            if (boundaryFlip || lmrError >= 160)
            {
                if (leviathanWitness != move)
                    leviathanWitnessCertifiedDepth = 0;
                leviathanWitness       = move;
                leviathanWitnessWasLmr = true;
            }
''','counterexample receipt')

s = replace_once(s,
'''        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        if (rootNode)
''',
'''        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        if (leviathanWitnessNeedsFullProof && leviathanWitness == move
            && leviathanWitnessWasLmr)
            leviathanWitnessCertifiedDepth =
              std::max(leviathanWitnessCertifiedDepth, int(leviathanEntryDepth));

        if (rootNode)
''','certify')

s = replace_once(s,
'''                {
                    leviathanWitness = leviathanDisplacedMove;
                    leviathanProofDebt = std::min(5, leviathanProofDebt + 1);
                }
''',
'''                {
                    if (leviathanWitness != leviathanDisplacedMove)
                    {
                        leviathanWitnessWasLmr = false;
                        leviathanWitnessCertifiedDepth = 0;
                    }
                    leviathanWitness = leviathanDisplacedMove;
                    leviathanProofDebt = std::min(5, leviathanProofDebt + 1);
                }
''','rival identity')

s = replace_once(s,
'''    if (!excludedMove && (leviathanProofDebt >= 3 || leviathanWitness))
        leviathan_proof_memory_store(posKey, leviathanProofDebt, leviathanWitness);
''',
'''    if (!excludedMove && (leviathanProofDebt >= 3 || leviathanWitness))
        leviathan_proof_memory_store(posKey, leviathanProofDebt, leviathanWitness,
                                     leviathanWitnessWasLmr,
                                     leviathanWitnessCertifiedDepth);
''','store')
sp.write_text(s)

final = sp.read_text()
if '''                else
                    e.debt -= decay;
''' not in final:
    raise SystemExit('V7 decay semantics changed unexpectedly')
if 'if (e.debt < 3)' in final:
    raise SystemExit('receipt-only branch leaked witness hygiene')
for needle in ['leviathanWitnessWasLmr','leviathanWitnessCertifiedDepth','leviathanWitnessNeedsFullProof']:
    if needle not in final:
        raise SystemExit(f'missing receipt invariant: {needle}')
print('V8.5B applied: LMR receipt only; V7.4 witness decay preserved exactly')

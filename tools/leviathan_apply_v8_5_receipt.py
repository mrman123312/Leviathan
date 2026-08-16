#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


# ---------------- search.h: minimal persistent causal receipt ----------------
hp = Path("src/search.h")
h = hp.read_text()

h = replace_once(
    h,
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
''',
    "receipt entry",
)

h = replace_once(
    h,
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
''',
    "receipt helpers",
)

hp.write_text(h)


# ---------------- search.cpp: one cause -> one bounded action ----------------
sp = Path("src/search.cpp")
s = sp.read_text()

# Expire witness identity whenever scalar evidence decays below actionable level.
s = replace_once(
    s,
    '''                if (e.debt <= decay)
                    e = {};
                else
                    e.debt -= decay;
''',
    '''                if (e.debt <= decay)
                    e = {};
                else
                {
                    e.debt -= decay;
                    if (e.debt < 3)
                    {
                        e.witness       = Move::none();
                        e.witnessWasLmr = false;
                        e.witnessDepth  = 0;
                    }
                }
''',
    "receipt decay",
)

# Immutable entry depth; Stockfish mutates depth for later siblings.
s = replace_once(
    s,
    '''    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    PVMoves   pv;
''',
    '''    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));
    const Depth leviathanEntryDepth = depth;

    PVMoves   pv;
''',
    "entry depth",
)

s = replace_once(
    s,
    '''    bool  capture, ttCapture, leviathanNullFragile;
    int   priorReduction, leviathanProofDebt;
''',
    '''    bool  capture, ttCapture, leviathanNullFragile, leviathanWitnessWasLmr;
    int   priorReduction, leviathanProofDebt, leviathanWitnessCertifiedDepth;
''',
    "receipt locals",
)

s = replace_once(
    s,
    '''    leviathanProofDebt   = 0;
    leviathanWitness     = Move::none();
    priorReduction        = (ss - 1)->reduction;
''',
    '''    leviathanProofDebt   = 0;
    leviathanWitness     = Move::none();
    leviathanWitnessWasLmr = false;
    leviathanWitnessCertifiedDepth = 0;
    priorReduction        = (ss - 1)->reduction;
''',
    "receipt initialization",
)

s = replace_once(
    s,
    '''    leviathanWitness = leviathan_proof_memory_witness(posKey);
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);
''',
    '''    leviathanWitness = leviathan_proof_memory_witness(posKey);
    leviathanWitnessWasLmr = leviathan_proof_memory_witness_was_lmr(posKey);
    leviathanWitnessCertifiedDepth = int(leviathan_proof_memory_witness_depth(posKey));
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);
''',
    "receipt load",
)

# ProbCut/rival witnesses are still useful V7 ordering evidence, but they do not
# inherit LMR certification if they replace a different witness.
s = replace_once(
    s,
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
''',
    "ProbCut receipt identity",
)

# Insert the sole new V8.5 authority immediately before V7 Proof Regime.
proof_anchor = '''        // Leviathan strength v6.4 - Proof Regime. Once several independent
        // warnings agree, incremental buybacks are not enough: accumulated LMR
'''
proof_code = '''        // Leviathan strength v8.5 - Causal LMR Receipt. This is the only new
        // production search authority beyond V7.4. Reopen one exact move only if
        // that move previously demonstrated a serious LMR counterexample, only on
        // PV-relevant nodes, and only after at least two more entry-depth plies.
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
if s.count(proof_anchor) != 1:
    raise SystemExit(f"proof anchor count {s.count(proof_anchor)}")
s = s.replace(proof_anchor, proof_code + proof_anchor, 1)

# Margin Skeptic boundary rescue is itself a direct LMR counterexample.
s = replace_once(
    s,
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
''',
    "margin-skeptic LMR receipt",
)

# Counterexample Feedback can independently mint the same receipt.
s = replace_once(
    s,
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
''',
    "counterexample LMR receipt",
)

# Certify only after the child search completed and was not aborted.
s = replace_once(
    s,
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
''',
    "receipt certification",
)

# Rival witness may replace the identity; if so it is not an LMR receipt.
s = replace_once(
    s,
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
''',
    "rival receipt identity",
)

s = replace_once(
    s,
    '''    if (!excludedMove && (leviathanProofDebt >= 3 || leviathanWitness))
        leviathan_proof_memory_store(posKey, leviathanProofDebt, leviathanWitness);
''',
    '''    if (!excludedMove && (leviathanProofDebt >= 3 || leviathanWitness))
        leviathan_proof_memory_store(posKey, leviathanProofDebt, leviathanWitness,
                                     leviathanWitnessWasLmr,
                                     leviathanWitnessCertifiedDepth);
''',
    "receipt persistence",
)

sp.write_text(s)

# Structural fail-closed checks. V8.5 must stay V7-like: no Evidence hot-path state.
final_h = hp.read_text()
final_s = sp.read_text()
required = [
    ("receipt flag", "witnessWasLmr", final_h),
    ("receipt depth", "witnessDepth", final_h),
    ("entry depth", "const Depth leviathanEntryDepth", final_s),
    ("minimal proof gate", "move == persistentWitness && leviathanWitnessWasLmr && (PvNode || ss->ttPv)", final_s),
    ("two-ply reopen", "leviathanEntryDepth >= leviathanWitnessCertifiedDepth + 2", final_s),
    ("receipt store", "leviathanWitnessCertifiedDepth);", final_s),
]
for label, needle, haystack in required:
    if needle not in haystack:
        raise SystemExit(f"missing V8.5 invariant: {label}")

for forbidden in [
    "Leviathan::Evidence::State",
    "leviathanEvidence",
    "witnessEvidence",
    "PROVENANCE_DOMAIN",
    "SEARCH_DOMAIN",
    "RIVAL_DOMAIN",
]:
    if forbidden in final_s or forbidden in final_h:
        raise SystemExit(f"V8.5 hot-path taxonomy leaked in: {forbidden}")

print("V8.5 causal receipt applied: V7.4 scalar search + exact LMR witness/depth only")

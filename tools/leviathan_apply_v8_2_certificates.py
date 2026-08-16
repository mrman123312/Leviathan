#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)

# Persistent memory grows by one bounded depth watermark. No allocation, no locks.
hp = Path("src/search.h")
h = hp.read_text()
h = replace_once(
    h,
    '''    struct LeviathanProofMemoryEntry {
        Key                       key      = 0;
        unsigned int              debt     = 0;
        Leviathan::Evidence::Mask evidence = 0;
        Move                      witness  = Move::none();
    };
''',
    '''    struct LeviathanProofMemoryEntry {
        Key                       key          = 0;
        unsigned int              debt         = 0;
        Leviathan::Evidence::Mask evidence     = 0;
        Move                      witness      = Move::none();
        unsigned int              witnessDepth = 0;
    };
''',
    "proof-memory certificate field",
)

h = replace_once(
    h,
    '''    Move leviathan_proof_memory_witness(Key key) const {
        const auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        return e.key == key ? e.witness : Move::none();
    }

    void leviathan_proof_memory_store(Key key,
                                      int debt,
                                      Leviathan::Evidence::Mask evidence,
                                      Move witness = Move::none()) {
        if (debt < 3 && !witness)
            return;
        auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        const unsigned boundedDebt = unsigned(std::clamp(debt, 0, 5));
        if (e.key != key)
            e = {key, boundedDebt, evidence, witness};
        else
        {
            e.debt = std::max(e.debt, boundedDebt);
            e.evidence |= evidence;
            if (witness)
                e.witness = witness;
        }
    }
''',
    '''    Move leviathan_proof_memory_witness(Key key) const {
        const auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        return e.key == key ? e.witness : Move::none();
    }

    unsigned int leviathan_proof_memory_witness_depth(Key key) const {
        const auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        return e.key == key ? e.witnessDepth : 0;
    }

    void leviathan_proof_memory_store(Key key,
                                      int debt,
                                      Leviathan::Evidence::Mask evidence,
                                      Move witness = Move::none(),
                                      int witnessDepth = 0) {
        if (debt < 3 && !witness)
            return;
        auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        const unsigned boundedDebt = unsigned(std::clamp(debt, 0, 5));
        const unsigned boundedWitnessDepth = unsigned(std::max(0, witnessDepth));
        if (e.key != key)
            e = {key, boundedDebt, evidence, witness, boundedWitnessDepth};
        else
        {
            e.debt = std::max(e.debt, boundedDebt);
            e.evidence |= evidence;
            if (witness)
            {
                if (e.witness != witness)
                {
                    e.witness      = witness;
                    e.witnessDepth = boundedWitnessDepth;
                }
                else
                    e.witnessDepth = std::max(e.witnessDepth, boundedWitnessDepth);
            }
        }
    }
''',
    "proof-memory certificate helpers",
)
hp.write_text(h)

sp = Path("src/search.cpp")
s = sp.read_text()

# Entry depth is immutable even though Stockfish intentionally mutates depth for
# later siblings after alpha improvements.
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
    "immutable entry depth",
)

s = replace_once(
    s,
    '''    int   priorReduction;
    Leviathan::Evidence::State leviathanEvidence;
''',
    '''    int   priorReduction, leviathanWitnessCertifiedDepth;
    Leviathan::Evidence::State leviathanEvidence;
''',
    "certificate local declaration",
)

s = replace_once(
    s,
    '''    leviathanEvidence    = {};
    leviathanWitness     = Move::none();
    priorReduction        = (ss - 1)->reduction;
''',
    '''    leviathanEvidence    = {};
    leviathanWitness     = Move::none();
    leviathanWitnessCertifiedDepth = 0;
    priorReduction        = (ss - 1)->reduction;
''',
    "certificate local initialization",
)

s = replace_once(
    s,
    '''    leviathanWitness = leviathan_proof_memory_witness(posKey);
    if (leviathanWitness)
''',
    '''    leviathanWitness = leviathan_proof_memory_witness(posKey);
    leviathanWitnessCertifiedDepth = int(leviathan_proof_memory_witness_depth(posKey));
    if (leviathanWitness)
''',
    "certificate load",
)

# New witnesses invalidate the old certificate.
s = replace_once(
    s,
    '''    if (probCutNearMiss)
    {
        leviathanEvidence.add(Leviathan::Evidence::Kind::PROBCUT_NEAR_PROOF, 1);
        leviathanWitness = probCutNearMiss;
    }
''',
    '''    if (probCutNearMiss)
    {
        leviathanEvidence.add(Leviathan::Evidence::Kind::PROBCUT_NEAR_PROOF, 1);
        if (leviathanWitness != probCutNearMiss)
            leviathanWitnessCertifiedDepth = 0;
        leviathanWitness = probCutNearMiss;
    }
''',
    "ProbCut certificate invalidation",
)

# Upgrade the witness proof contract: it now reopens only when requested depth
# exceeds the watermark, and restores entry depth that sibling penalties may have cut.
s = replace_once(
    s,
    '''        const bool leviathanWitnessNeedsFullProof =
          move == persistentWitness && depth >= 6
          && (leviathanEvidence.contains(Leviathan::Evidence::Kind::LMR_COUNTEREXAMPLE)
              || leviathanEvidence.contains(Leviathan::Evidence::Kind::PROBCUT_NEAR_PROOF)
              || leviathanEvidence.contains(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY));
        if (leviathanWitnessNeedsFullProof)
            r = std::min(r, 0);
''',
    '''        const bool leviathanWitnessNeedsFullProof =
          move == persistentWitness && leviathanEntryDepth >= 6
          && leviathanEntryDepth > leviathanWitnessCertifiedDepth
          && (leviathanEvidence.contains(Leviathan::Evidence::Kind::LMR_COUNTEREXAMPLE)
              || leviathanEvidence.contains(Leviathan::Evidence::Kind::PROBCUT_NEAR_PROOF)
              || leviathanEvidence.contains(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY));
        if (leviathanWitnessNeedsFullProof)
        {
            newDepth = std::max(newDepth, leviathanEntryDepth - 1);
            r = std::min(r, 0);
        }
''',
    "depth-monotonic witness proof contract",
)

# Any newly discovered LMR witness replaces the certificate identity.
s = replace_once(
    s,
    '''            if (boundaryFlip || lmrError >= 160)
                leviathanWitness = move;
''',
    '''            if (boundaryFlip || lmrError >= 160)
            {
                if (leviathanWitness != move)
                    leviathanWitnessCertifiedDepth = 0;
                leviathanWitness = move;
            }
''',
    "LMR certificate invalidation",
)

# Certify only after the move finished and the search was not aborted.
s = replace_once(
    s,
    '''        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        if (rootNode)
''',
    '''        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        if (leviathanWitnessNeedsFullProof && leviathanWitness == move)
            leviathanWitnessCertifiedDepth =
              std::max(leviathanWitnessCertifiedDepth, int(leviathanEntryDepth));

        if (rootNode)
''',
    "successful witness certification",
)

# A newly displaced rival is a new proof identity unless it happens to be the same witness.
s = replace_once(
    s,
    '''                {
                    leviathanWitness = leviathanDisplacedMove;
                    leviathanEvidence.add(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY, 1);
                }
''',
    '''                {
                    if (leviathanWitness != leviathanDisplacedMove)
                        leviathanWitnessCertifiedDepth = 0;
                    leviathanWitness = leviathanDisplacedMove;
                    leviathanEvidence.add(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY, 1);
                }
''',
    "rival certificate invalidation",
)

s = replace_once(
    s,
    '''        leviathan_proof_memory_store(posKey, leviathanProofDebt,
                                     leviathanEvidence.mask, leviathanWitness);
''',
    '''        leviathan_proof_memory_store(posKey, leviathanProofDebt,
                                     leviathanEvidence.mask, leviathanWitness,
                                     leviathanWitnessCertifiedDepth);
''',
    "certificate persistence",
)

# Decay hygiene must erase the watermark with the witness identity.
s = replace_once(
    s,
    '''                        e.evidence = 0;
                        e.witness  = Move::none();
''',
    '''                        e.evidence     = 0;
                        e.witness      = Move::none();
                        e.witnessDepth = 0;
''',
    "certificate decay hygiene",
)

sp.write_text(s)

final_h = hp.read_text()
final_s = sp.read_text()
for label, needle, haystack in [
    ("certificate field", "witnessDepth", final_h),
    ("certificate load", "leviathan_proof_memory_witness_depth", final_s),
    ("immutable entry depth", "leviathanEntryDepth", final_s),
    ("certificate gating", "leviathanEntryDepth > leviathanWitnessCertifiedDepth", final_s),
    ("certificate persistence", "leviathanWitnessCertifiedDepth);", final_s),
]:
    if needle not in haystack:
        raise SystemExit(f"missing V8.2 invariant: {label}")

print("V8.2 proof certificates applied with depth-monotonic re-verification")

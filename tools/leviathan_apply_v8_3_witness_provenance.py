#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)

hp = Path("src/search.h")
h = hp.read_text()

h = replace_once(
    h,
    '''    struct LeviathanProofMemoryEntry {
        Key                       key          = 0;
        unsigned int              debt         = 0;
        Leviathan::Evidence::Mask evidence     = 0;
        Move                      witness      = Move::none();
        unsigned int              witnessDepth = 0;
    };
''',
    '''    struct LeviathanProofMemoryEntry {
        Key                       key             = 0;
        unsigned int              debt            = 0;
        Leviathan::Evidence::Mask evidence        = 0;
        Move                      witness         = Move::none();
        Leviathan::Evidence::Mask witnessEvidence = 0;
        unsigned int              witnessDepth    = 0;
    };
''',
    "witness-cause field",
)

h = replace_once(
    h,
    '''    unsigned int leviathan_proof_memory_witness_depth(Key key) const {
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
    '''    Leviathan::Evidence::Mask leviathan_proof_memory_witness_evidence(Key key) const {
        const auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        return e.key == key ? e.witnessEvidence : 0;
    }

    unsigned int leviathan_proof_memory_witness_depth(Key key) const {
        const auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        return e.key == key ? e.witnessDepth : 0;
    }

    void leviathan_proof_memory_store(Key key,
                                      int debt,
                                      Leviathan::Evidence::Mask evidence,
                                      Move witness = Move::none(),
                                      Leviathan::Evidence::Mask witnessEvidence = 0,
                                      int witnessDepth = 0) {
        if (debt < 3 && !witness)
            return;
        auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        const unsigned boundedDebt = unsigned(std::clamp(debt, 0, 5));
        const auto boundedWitnessEvidence = witnessEvidence & Leviathan::Evidence::KNOWN_MASK;
        const unsigned boundedWitnessDepth = unsigned(std::max(0, witnessDepth));
        if (e.key != key)
            e = {key, boundedDebt, evidence, witness, boundedWitnessEvidence, boundedWitnessDepth};
        else
        {
            e.debt = std::max(e.debt, boundedDebt);
            e.evidence |= evidence;
            if (witness)
            {
                if (e.witness != witness)
                {
                    e.witness         = witness;
                    e.witnessEvidence = boundedWitnessEvidence;
                    e.witnessDepth    = boundedWitnessDepth;
                }
                else
                {
                    e.witnessEvidence |= boundedWitnessEvidence;
                    e.witnessDepth = std::max(e.witnessDepth, boundedWitnessDepth);
                }
            }
        }
    }
''',
    "witness-cause helpers",
)

hp.write_text(h)

sp = Path("src/search.cpp")
s = sp.read_text()

s = replace_once(
    s,
    '''    int   priorReduction, leviathanWitnessCertifiedDepth;
    Leviathan::Evidence::State leviathanEvidence;
    int&  leviathanProofDebt = leviathanEvidence.debt;
''',
    '''    int   priorReduction, leviathanWitnessCertifiedDepth;
    Leviathan::Evidence::State leviathanEvidence;
    Leviathan::Evidence::Mask  leviathanWitnessEvidence = 0;
    int&  leviathanProofDebt = leviathanEvidence.debt;
''',
    "local witness evidence",
)

s = replace_once(
    s,
    '''    leviathanWitness = leviathan_proof_memory_witness(posKey);
    leviathanWitnessCertifiedDepth = int(leviathan_proof_memory_witness_depth(posKey));
    if (leviathanWitness)
''',
    '''    leviathanWitness = leviathan_proof_memory_witness(posKey);
    leviathanWitnessEvidence = leviathan_proof_memory_witness_evidence(posKey);
    leviathanWitnessCertifiedDepth = int(leviathan_proof_memory_witness_depth(posKey));
    if (leviathanWitness)
''',
    "witness evidence load",
)

s = replace_once(
    s,
    '''        if (leviathanWitness != probCutNearMiss)
            leviathanWitnessCertifiedDepth = 0;
        leviathanWitness = probCutNearMiss;
''',
    '''        const auto probCutWitnessEvidence =
          Leviathan::Evidence::bit(Leviathan::Evidence::Kind::PROBCUT_NEAR_PROOF);
        if (leviathanWitness != probCutNearMiss)
        {
            leviathanWitnessCertifiedDepth = 0;
            leviathanWitnessEvidence = probCutWitnessEvidence;
        }
        else
            leviathanWitnessEvidence |= probCutWitnessEvidence;
        leviathanWitness = probCutNearMiss;
''',
    "ProbCut witness provenance",
)

s = replace_once(
    s,
    '''        const bool leviathanWitnessNeedsFullProof =
          move == persistentWitness && leviathanEntryDepth >= 6
          && leviathanEntryDepth > leviathanWitnessCertifiedDepth
          && (leviathanEvidence.contains(Leviathan::Evidence::Kind::LMR_COUNTEREXAMPLE)
              || leviathanEvidence.contains(Leviathan::Evidence::Kind::PROBCUT_NEAR_PROOF)
              || leviathanEvidence.contains(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY));
''',
    '''        constexpr auto leviathanProofWitnessCauses =
          Leviathan::Evidence::bit(Leviathan::Evidence::Kind::LMR_COUNTEREXAMPLE)
          | Leviathan::Evidence::bit(Leviathan::Evidence::Kind::PROBCUT_NEAR_PROOF)
          | Leviathan::Evidence::bit(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY);
        const bool leviathanWitnessNeedsFullProof =
          move == persistentWitness && leviathanEntryDepth >= 6
          && leviathanEntryDepth > leviathanWitnessCertifiedDepth
          && (leviathanWitnessEvidence & leviathanProofWitnessCauses);
''',
    "cause-bound witness proof contract",
)

s = replace_once(
    s,
    '''                if (leviathanWitness != move)
                    leviathanWitnessCertifiedDepth = 0;
                leviathanWitness = move;
''',
    '''                const auto lmrWitnessEvidence =
                  Leviathan::Evidence::bit(Leviathan::Evidence::Kind::LMR_COUNTEREXAMPLE);
                if (leviathanWitness != move)
                {
                    leviathanWitnessCertifiedDepth = 0;
                    leviathanWitnessEvidence = lmrWitnessEvidence;
                }
                else
                    leviathanWitnessEvidence |= lmrWitnessEvidence;
                leviathanWitness = move;
''',
    "LMR witness provenance",
)

s = replace_once(
    s,
    '''                    if (leviathanWitness != leviathanDisplacedMove)
                        leviathanWitnessCertifiedDepth = 0;
                    leviathanWitness = leviathanDisplacedMove;
                    leviathanEvidence.add(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY, 1);
''',
    '''                    const auto rivalWitnessEvidence =
                      Leviathan::Evidence::bit(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY);
                    if (leviathanWitness != leviathanDisplacedMove)
                    {
                        leviathanWitnessCertifiedDepth = 0;
                        leviathanWitnessEvidence = rivalWitnessEvidence;
                    }
                    else
                        leviathanWitnessEvidence |= rivalWitnessEvidence;
                    leviathanWitness = leviathanDisplacedMove;
                    leviathanEvidence.add(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY, 1);
''',
    "rival witness provenance",
)

s = replace_once(
    s,
    '''                        e.evidence     = 0;
                        e.witness      = Move::none();
                        e.witnessDepth = 0;
''',
    '''                        e.evidence        = 0;
                        e.witness         = Move::none();
                        e.witnessEvidence = 0;
                        e.witnessDepth    = 0;
''',
    "witness-cause decay hygiene",
)

s = replace_once(
    s,
    '''        leviathan_proof_memory_store(posKey, leviathanProofDebt,
                                     leviathanEvidence.mask, leviathanWitness,
                                     leviathanWitnessCertifiedDepth);
''',
    '''        leviathan_proof_memory_store(posKey, leviathanProofDebt,
                                     leviathanEvidence.mask, leviathanWitness,
                                     leviathanWitnessEvidence,
                                     leviathanWitnessCertifiedDepth);
''',
    "witness provenance persistence",
)

sp.write_text(s)

final_h = hp.read_text()
final_s = sp.read_text()
for label, needle, haystack in [
    ("witness cause field", "witnessEvidence", final_h),
    ("witness cause loader", "leviathan_proof_memory_witness_evidence", final_s),
    ("cause-bound proof", "leviathanWitnessEvidence & leviathanProofWitnessCauses", final_s),
    ("probcut witness cause", "probCutWitnessEvidence", final_s),
    ("lmr witness cause", "lmrWitnessEvidence", final_s),
    ("rival witness cause", "rivalWitnessEvidence", final_s),
]:
    if needle not in haystack:
        raise SystemExit(f"missing V8.3 invariant: {label}")

print("V8.3 witness provenance bound cause, identity, and certificate depth atomically")

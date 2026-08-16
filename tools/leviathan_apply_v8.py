#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


# --- search.h: persistent evidence provenance ---
hp = Path("src/search.h")
h = hp.read_text()
h = replace_once(
    h,
    '#include "history.h"\n#include "misc.h"\n',
    '#include "history.h"\n#include "leviathan_evidence.h"\n#include "misc.h"\n',
    "search.h include",
)

h = replace_once(
    h,
    '''    struct LeviathanProofMemoryEntry {
        Key          key     = 0;
        unsigned int debt    = 0;
        Move         witness = Move::none();
    };
''',
    '''    struct LeviathanProofMemoryEntry {
        Key                       key      = 0;
        unsigned int              debt     = 0;
        Leviathan::Evidence::Mask evidence = 0;
        Move                      witness  = Move::none();
    };
''',
    "proof memory entry",
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
    '''    Leviathan::Evidence::Mask leviathan_proof_memory_evidence(Key key) const {
        const auto& e = leviathanProofMemory[usize(key) & (LEVIATHAN_PROOF_MEMORY_SIZE - 1)];
        return e.key == key ? e.evidence : 0;
    }

    Move leviathan_proof_memory_witness(Key key) const {
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
    "proof memory helpers",
)
hp.write_text(h)


# --- search.cpp: typed evidence state + specialist routing ---
sp = Path("src/search.cpp")
s = sp.read_text()

s = replace_once(
    s,
    '''    bool  capture, ttCapture, leviathanNullFragile;
    int   priorReduction, leviathanProofDebt;
    Piece movedPiece;
''',
    '''    bool  capture, ttCapture, leviathanNullFragile;
    int   priorReduction;
    Leviathan::Evidence::State leviathanEvidence;
    int&  leviathanProofDebt = leviathanEvidence.debt;
    Piece movedPiece;
''',
    "local evidence state",
)

s = replace_once(
    s,
    '''    leviathanNullFragile = false;
    leviathanProofDebt   = 0;
    leviathanWitness     = Move::none();
''',
    '''    leviathanNullFragile = false;
    leviathanEvidence    = {};
    leviathanWitness     = Move::none();
''',
    "evidence initialization",
)

s = replace_once(
    s,
    '''    // If this worker already discovered that heuristics disagree here, begin
    // the revisit with that proof debt instead of paying to rediscover it.
    leviathanProofDebt = std::max(leviathanProofDebt,
                                  int(leviathan_proof_memory_load(posKey)));
    leviathanWitness = leviathan_proof_memory_witness(posKey);
''',
    '''    // If this worker already discovered that heuristics disagree here, begin
    // the revisit with both its bounded debt and typed causal evidence. The score
    // itself is never inherited: persistent memory carries skepticism, not truth.
    const unsigned leviathanPersistentDebt = leviathan_proof_memory_load(posKey);
    leviathanEvidence.merge(leviathan_proof_memory_evidence(posKey),
                             int(leviathanPersistentDebt));
    leviathanWitness = leviathan_proof_memory_witness(posKey);
    if (leviathanWitness)
        leviathanEvidence.mask |= Leviathan::Evidence::bit(
          Leviathan::Evidence::Kind::PERSISTENT_WITNESS);
''',
    "persistent evidence load",
)

s = replace_once(
    s,
    '''    // Leviathan strength v6 - Proof Obligation. Search shortcuts are evidence,
    // not truth. Accumulate independent reasons that the node's current story
    // may be unreliable, then spend verification only when warnings agree.
    if (pos.has_repeated())
        leviathanProofDebt += 2;
    if (pos.rule50_count() >= 80)
        leviathanProofDebt += 1;
    if (is_valid(ttData.value) && !is_decisive(ttData.value))
    {
        const int ttEvalGap = std::abs(int(ttData.value - ss->staticEval));
        leviathanProofDebt += ttEvalGap >= 256 ? 2 : int(ttEvalGap >= 96);
    }
    const int correctionGap = std::abs(correctionValue) / 131072;
    if (correctionGap >= 96)
        leviathanProofDebt += 1;
    leviathanProofDebt = std::min(leviathanProofDebt, 5);
''',
    '''    // Leviathan strength v8 - Evidence Lattice. Keep debt as the bounded
    // aggregate budget, but preserve the causal class of every warning so each
    // search subsystem can react only to evidence relevant to its own failure mode.
    if (pos.has_repeated())
        leviathanEvidence.add(Leviathan::Evidence::Kind::HISTORY, 2);
    if (pos.rule50_count() >= 80)
        leviathanEvidence.add(Leviathan::Evidence::Kind::RULE50, 1);
    if (is_valid(ttData.value) && !is_decisive(ttData.value))
    {
        const int ttEvalGap = std::abs(int(ttData.value - ss->staticEval));
        if (ttEvalGap >= 256)
            leviathanEvidence.add(Leviathan::Evidence::Kind::EVAL_DISAGREEMENT, 2);
        else if (ttEvalGap >= 96)
            leviathanEvidence.add(Leviathan::Evidence::Kind::EVAL_DISAGREEMENT, 1);
    }
    const int correctionGap = std::abs(correctionValue) / 131072;
    if (correctionGap >= 96)
        leviathanEvidence.add(Leviathan::Evidence::Kind::CORRECTION_STRESS, 1);
''',
    "base evidence classification",
)

s = replace_once(
    s,
    '''    if (!PvNode && !excludedMove && leviathanProofDebt < 2
        && ttData.depth > depth - (ttData.value <= beta)
''',
    '''    if (!PvNode && !excludedMove && leviathanProofDebt < 2
        && !Leviathan::Evidence::tt_sensitive(leviathanEvidence)
        && ttData.depth > depth - (ttData.value <= beta)
''',
    "TT specialist gate",
)

s = replace_once(
    s,
    '''            leviathanNullFragile = true;
            leviathanProofDebt = std::min(5, leviathanProofDebt + 2);
''',
    '''            leviathanNullFragile = true;
            leviathanEvidence.add(Leviathan::Evidence::Kind::NULL_FRAGILITY, 2);
''',
    "null evidence",
)

s = replace_once(
    s,
    '''            const bool leviathanVerifyNull = leviathanProofDebt >= 2 && depth >= 10;
''',
    '''            const bool leviathanVerifyNull = depth >= 10
              && (leviathanProofDebt >= 2
                  || leviathanEvidence.contains(Leviathan::Evidence::Kind::NULL_FRAGILITY)
                  || leviathanEvidence.multi_source());
''',
    "null verification routing",
)

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
        leviathanEvidence.add(Leviathan::Evidence::Kind::PROBCUT_NEAR_PROOF, 1);
        leviathanWitness = probCutNearMiss;
    }
''',
    "ProbCut evidence",
)

s = replace_once(
    s,
    '''            if (leviathanProofDebt < 2
                && moveCount >= (3 + depth * depth) / (2 - improving))
''',
    '''            if (leviathanProofDebt < 2
                && !Leviathan::Evidence::pruning_sensitive(leviathanEvidence)
                && !Leviathan::Evidence::rival_sensitive(leviathanEvidence)
                && moveCount >= (3 + depth * depth) / (2 - improving))
''',
    "quiet skip specialist gate",
)

s = replace_once(
    s,
    '''            if (capture || givesCheck)
            {
                Piece capturedPiece = pos.piece_on(move.to_sq());
                int   captHist = captureHistory[movedPiece][move.to_sq()][type_of(capturedPiece)];

                // Futility pruning for captures
''',
    '''            if (capture || givesCheck)
            {
                Piece capturedPiece = pos.piece_on(move.to_sq());
                int   captHist = captureHistory[movedPiece][move.to_sq()][type_of(capturedPiece)];
                const bool leviathanForcingProtected =
                  move == persistentWitness
                  || (leviathanProofDebt >= 3 && moveCount <= 6)
                  || (Leviathan::Evidence::rival_sensitive(leviathanEvidence) && moveCount <= 4);

                // Futility pruning for captures
''',
    "forcing witness protection declaration",
)

s = replace_once(
    s,
    '''                    if (futilityValue <= alpha
                        && !(leviathanProofDebt >= 3 && moveCount <= 6))
                        continue;
''',
    '''                    if (futilityValue <= alpha && !leviathanForcingProtected)
                        continue;
''',
    "capture futility witness protection",
)

s = replace_once(
    s,
    '''                if ((alpha >= VALUE_DRAW || pos.non_pawn_material(us) != PieceValue[movedPiece])
                    && !(leviathanProofDebt >= 3 && moveCount <= 6)
                    && !pos.see_ge(move, -margin)
''',
    '''                if ((alpha >= VALUE_DRAW || pos.non_pawn_material(us) != PieceValue[movedPiece])
                    && !leviathanForcingProtected
                    && !pos.see_ge(move, -margin)
''',
    "capture SEE witness protection",
)

s = replace_once(
    s,
    '''                  || move == persistentWitness
                  || (leviathanProofDebt >= 2
                      && moveCount <= 6 + std::min(leviathanProofDebt, 4));
''',
    '''                  || move == persistentWitness
                  || (Leviathan::Evidence::rival_sensitive(leviathanEvidence) && moveCount <= 6)
                  || (Leviathan::Evidence::pruning_sensitive(leviathanEvidence) && moveCount <= 6)
                  || (leviathanProofDebt >= 2
                      && moveCount <= 6 + std::min(leviathanProofDebt, 4));
''',
    "quiet specialist protection",
)

s = replace_once(
    s,
    '''            else if (value >= beta && !is_decisive(value)
                     && leviathanProofDebt < 3)
''',
    '''            else if (value >= beta && !is_decisive(value)
                     && leviathanProofDebt < 3
                     && !Leviathan::Evidence::pruning_sensitive(leviathanEvidence))
''',
    "multicut specialist gate",
)

s = replace_once(
    s,
    '''            else if ((ttData.value >= beta || cutNode) && leviathanProofDebt < 3)
                extension = -3;
''',
    '''            else if ((ttData.value >= beta || cutNode) && leviathanProofDebt < 3
                     && !Leviathan::Evidence::pruning_sensitive(leviathanEvidence))
                extension = -3;
''',
    "negative extension specialist gate",
)

s = replace_once(
    s,
    '''        if (move == probCutNearMiss)
            r -= 768;
        else if (move == persistentWitness)
            r -= 640;

        // Proof debt converts contradictory evidence into bounded search authority.
''',
    '''        if (move == probCutNearMiss)
            r -= 768;
        else if (move == persistentWitness)
            r -= 640;

        if (Leviathan::Evidence::pruning_sensitive(leviathanEvidence) && moveCount <= 4)
            r -= 256;
        if (Leviathan::Evidence::rival_sensitive(leviathanEvidence) && moveCount <= 4)
            r -= 192;

        // Proof debt converts contradictory evidence into bounded search authority.
''',
    "typed LMR buyback",
)

s = replace_once(
    s,
    '''        if (depth >= 7 && leviathanProofDebt >= 4 && moveCount <= 2)
            r = std::min(r, 0);      // full depth (negative extension remains allowed)
        else if (depth >= 6 && leviathanProofDebt >= 3 && moveCount <= 4)
            r = std::min(r, 1024);   // at most about one ply of positive reduction
''',
    '''        if (depth >= 7 && leviathanProofDebt >= 4 && moveCount <= 2)
            r = std::min(r, 0);      // full depth (negative extension remains allowed)
        else if (depth >= 7 && leviathanEvidence.severe() && moveCount <= 3)
            r = std::min(r, 0);      // diverse severe evidence certifies one extra rival
        else if (depth >= 6 && leviathanProofDebt >= 3 && moveCount <= 4)
            r = std::min(r, 1024);   // at most about one ply of positive reduction
''',
    "diverse proof regime",
)

s = replace_once(
    s,
    '''            const int  evidenceBoost = boundaryFlip ? 2 : lmrError >= 160 ? 2 : int(lmrError >= 64);
            leviathanProofDebt = std::min(5, leviathanProofDebt + evidenceBoost);
            if (boundaryFlip || lmrError >= 160)
                leviathanWitness = move;
''',
    '''            const int  evidenceBoost = boundaryFlip ? 2 : lmrError >= 160 ? 2 : int(lmrError >= 64);
            leviathanEvidence.add(Leviathan::Evidence::Kind::LMR_COUNTEREXAMPLE,
                                  evidenceBoost);
            if (boundaryFlip || lmrError >= 160)
                leviathanWitness = move;
''',
    "LMR counterexample evidence",
)

s = replace_once(
    s,
    '''        const unsigned leviathanChildDebt =
          leviathan_proof_memory_load(leviathan_tt_key(pos));
        if ((PvNode && leviathanChildDebt >= 3) || leviathanChildDebt >= 4)
            leviathanProofDebt = std::min(5, leviathanProofDebt + 1);
''',
    '''        const unsigned leviathanChildDebt =
          leviathan_proof_memory_load(leviathan_tt_key(pos));
        if ((PvNode && leviathanChildDebt >= 3) || leviathanChildDebt >= 4)
            leviathanEvidence.add(Leviathan::Evidence::Kind::CHILD_UNCERTAINTY, 1);
''',
    "child uncertainty evidence",
)

s = replace_once(
    s,
    '''                    leviathanWitness = leviathanDisplacedMove;
                    leviathanProofDebt = std::min(5, leviathanProofDebt + 1);
''',
    '''                    leviathanWitness = leviathanDisplacedMove;
                    leviathanEvidence.add(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY, 1);
''',
    "displaced rival evidence",
)

s = replace_once(
    s,
    '''                if (moveCount > 1 && value < beta && !is_decisive(value))
                    leviathanProofDebt = std::min(5, leviathanProofDebt + 1);
''',
    '''                if (moveCount > 1 && value < beta && !is_decisive(value))
                    leviathanEvidence.add(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY, 1);
''',
    "multi-alpha rival evidence",
)

s = replace_once(
    s,
    '''    if (!excludedMove && (leviathanProofDebt >= 3 || leviathanWitness))
        leviathan_proof_memory_store(posKey, leviathanProofDebt, leviathanWitness);
''',
    '''    if (!excludedMove && (leviathanProofDebt >= 3 || leviathanWitness))
        leviathan_proof_memory_store(posKey, leviathanProofDebt,
                                     leviathanEvidence.mask, leviathanWitness);
''',
    "persistent evidence store",
)

# Replace the two multiplicative root uncertainty blocks with one capped budget.
old_root = '''            // Leviathan strength v7.2 - Uncertainty Budget Coupling. Local proof
            // failures can now propagate to and persist at the root. Convert that
            // epistemic state into more verification time, while the normal hard
            // maximum-time clamp below remains authoritative.
            const unsigned leviathanRootDebt =
              leviathan_proof_memory_load(leviathan_tt_key(rootPos));
            if (!is_decisive(bestValue) && leviathanRootDebt >= 3)
                totalTime *= leviathanRootDebt >= 5 ? 1.18
                           : leviathanRootDebt >= 4 ? 1.12
                                                    : 1.06;

            // Leviathan strength v6.6 - Root Challenger Certification. A stable
            // leader is not the same as a certified leader when another root move
            // remains close. Use the rolling scores already maintained for every
            // root move to spend more of the existing clock budget on ambiguous
            // decisions. The normal maximum-time clamp below remains authoritative.
            if (rootDepth >= 6 && rootMoves.size() > 1 && !is_decisive(bestValue)
                && rootMoves[0].averageScore != -VALUE_INFINITE)
            {
                Value rivalAverage = -VALUE_INFINITE;
                const usize rivalLimit = std::min(rootMoves.size(), usize(6));
                for (usize i = 1; i < rivalLimit; ++i)
                    if (rootMoves[i].averageScore != -VALUE_INFINITE)
                        rivalAverage = std::max(rivalAverage, rootMoves[i].averageScore);

                if (rivalAverage != -VALUE_INFINITE)
                {
                    const int rivalGap = std::max(0, int(rootMoves[0].averageScore - rivalAverage));
                    const double certification = rivalGap <= 12 ? 1.22
                                               : rivalGap <= 32 ? 1.14
                                               : rivalGap <= 64 ? 1.07
                                                                : 1.0;
                    totalTime *= certification;
                }
            }
'''
new_root = '''            // Leviathan strength v8 - Root Evidence Budget. Typed proof evidence
            // and close-rival ambiguity share one capped multiplier instead of
            // multiplying independently. This keeps the response powerful while
            // preventing correlated warnings from creating runaway time use.
            double leviathanEvidenceFactor = 1.0;
            if (!is_decisive(bestValue))
            {
                Leviathan::Evidence::State rootEvidence;
                const Key rootEvidenceKey = leviathan_tt_key(rootPos);
                rootEvidence.merge(leviathan_proof_memory_evidence(rootEvidenceKey),
                                   int(leviathan_proof_memory_load(rootEvidenceKey)));
                leviathanEvidenceFactor = Leviathan::Evidence::root_time_factor(rootEvidence);
            }

            double leviathanRivalFactor = 1.0;
            if (rootDepth >= 6 && rootMoves.size() > 1 && !is_decisive(bestValue)
                && rootMoves[0].averageScore != -VALUE_INFINITE)
            {
                Value rivalAverage = -VALUE_INFINITE;
                const usize rivalLimit = std::min(rootMoves.size(), usize(6));
                for (usize i = 1; i < rivalLimit; ++i)
                    if (rootMoves[i].averageScore != -VALUE_INFINITE)
                        rivalAverage = std::max(rivalAverage, rootMoves[i].averageScore);

                if (rivalAverage != -VALUE_INFINITE)
                {
                    const int rivalGap = std::max(0, int(rootMoves[0].averageScore - rivalAverage));
                    leviathanRivalFactor = rivalGap <= 12 ? 1.22
                                           : rivalGap <= 32 ? 1.14
                                           : rivalGap <= 64 ? 1.07
                                                            : 1.0;
                }
            }
            totalTime *= std::min(1.28, leviathanEvidenceFactor * leviathanRivalFactor);
'''
s = replace_once(s, old_root, new_root, "root evidence budget")

sp.write_text(s)

# Cheap structural assertions before a compiler sees the patch.
final_h = hp.read_text()
final_s = sp.read_text()
required = {
    "typed memory field": "Leviathan::Evidence::Mask evidence",
    "typed state": "Leviathan::Evidence::State leviathanEvidence",
    "history evidence": "Kind::HISTORY",
    "null evidence": "Kind::NULL_FRAGILITY",
    "probcut evidence": "Kind::PROBCUT_NEAR_PROOF",
    "lmr evidence": "Kind::LMR_COUNTEREXAMPLE",
    "child evidence": "Kind::CHILD_UNCERTAINTY",
    "rival evidence": "Kind::RIVAL_AMBIGUITY",
    "capped root budget": "std::min(1.28, leviathanEvidenceFactor * leviathanRivalFactor)",
}
for label, needle in required.items():
    haystack = final_h if label == "typed memory field" else final_s
    if needle not in haystack:
        raise SystemExit(f"missing V8 invariant: {label}")

print("V8 evidence lattice patch applied with all structural invariants present")

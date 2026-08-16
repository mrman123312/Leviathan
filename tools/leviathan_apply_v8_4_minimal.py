#!/usr/bin/env python3
from pathlib import Path


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one anchor, found {n}")
    return text.replace(old, new, 1)

hp = Path('src/search.h')
h = hp.read_text()
h = replace_once(
    h,
    '''        const auto boundedWitnessEvidence = witnessEvidence & Leviathan::Evidence::KNOWN_MASK;
''',
    '''        const auto boundedWitnessEvidence = Leviathan::Evidence::Mask(
          witnessEvidence & Leviathan::Evidence::KNOWN_MASK);
''',
    'explicit witness-mask narrowing',
)
hp.write_text(h)

sp = Path('src/search.cpp')
s = sp.read_text()

# Restore V7.4 search authority everywhere except a tightly scoped LMR witness proof.
s = replace_once(
    s,
    '''        if (!pos.has_repeated()
            && !leviathanEvidence.contains(Leviathan::Evidence::Kind::EVAL_DISAGREEMENT)
            && is_valid(ttData.value)
''',
    '''        if (!pos.has_repeated() && is_valid(ttData.value)
''',
    'remove TT-eval quarantine authority',
)

s = replace_once(
    s,
    '''    if (!PvNode && !excludedMove && leviathanProofDebt < 2
        && !Leviathan::Evidence::tt_sensitive(leviathanEvidence)
        && ttData.depth > depth - (ttData.value <= beta)
''',
    '''    if (!PvNode && !excludedMove && leviathanProofDebt < 2
        && ttData.depth > depth - (ttData.value <= beta)
''',
    'restore V7 TT cutoff authority',
)

s = replace_once(
    s,
    '''            const bool leviathanVerifyNull = depth >= 10
              && (leviathanProofDebt >= 2
                  || leviathanEvidence.contains(Leviathan::Evidence::Kind::NULL_FRAGILITY)
                  || leviathanEvidence.multi_source());
''',
    '''            const bool leviathanVerifyNull = leviathanProofDebt >= 2 && depth >= 10;
''',
    'restore V7 null verification',
)

s = replace_once(
    s,
    '''            if (leviathanProofDebt < 2
                && !Leviathan::Evidence::pruning_sensitive(leviathanEvidence)
                && !Leviathan::Evidence::rival_sensitive(leviathanEvidence)
                && moveCount >= (3 + depth * depth) / (2 - improving))
''',
    '''            if (leviathanProofDebt < 2
                && moveCount >= (3 + depth * depth) / (2 - improving))
''',
    'restore V7 quiet skip',
)

s = replace_once(
    s,
    '''                const bool leviathanForcingProtected =
                  move == persistentWitness
                  || (leviathanProofDebt >= 3 && moveCount <= 6)
                  || (Leviathan::Evidence::rival_sensitive(leviathanEvidence) && moveCount <= 4);

                // Futility pruning for captures
''',
    '''                // Futility pruning for captures
''',
    'remove broad forcing protection declaration',
)

s = replace_once(
    s,
    '''                    if (futilityValue <= alpha && !leviathanForcingProtected)
                        continue;
''',
    '''                    if (futilityValue <= alpha
                        && !(leviathanProofDebt >= 3 && moveCount <= 6))
                        continue;
''',
    'restore V7 capture futility',
)

s = replace_once(
    s,
    '''                if ((alpha >= VALUE_DRAW || pos.non_pawn_material(us) != PieceValue[movedPiece])
                    && !leviathanForcingProtected
                    && !pos.see_ge(move, -margin)
''',
    '''                if ((alpha >= VALUE_DRAW || pos.non_pawn_material(us) != PieceValue[movedPiece])
                    && !(leviathanProofDebt >= 3 && moveCount <= 6)
                    && !pos.see_ge(move, -margin)
''',
    'restore V7 capture SEE',
)

s = replace_once(
    s,
    '''                  || move == persistentWitness
                  || (Leviathan::Evidence::rival_sensitive(leviathanEvidence) && moveCount <= 6)
                  || (Leviathan::Evidence::pruning_sensitive(leviathanEvidence) && moveCount <= 6)
                  || (leviathanProofDebt >= 2
''',
    '''                  || move == persistentWitness
                  || (leviathanProofDebt >= 2
''',
    'restore V7 quiet protection',
)

s = replace_once(
    s,
    '''            else if (value >= beta && !is_decisive(value)
                     && leviathanProofDebt < 3
                     && !Leviathan::Evidence::pruning_sensitive(leviathanEvidence))
''',
    '''            else if (value >= beta && !is_decisive(value)
                     && leviathanProofDebt < 3)
''',
    'restore V7 multicut',
)

s = replace_once(
    s,
    '''            else if ((ttData.value >= beta || cutNode) && leviathanProofDebt < 3
                     && !Leviathan::Evidence::pruning_sensitive(leviathanEvidence))
''',
    '''            else if ((ttData.value >= beta || cutNode) && leviathanProofDebt < 3)
''',
    'restore V7 negative extension',
)

s = replace_once(
    s,
    '''        if (Leviathan::Evidence::pruning_sensitive(leviathanEvidence) && moveCount <= 4)
            r -= 256;
        if (Leviathan::Evidence::rival_sensitive(leviathanEvidence) && moveCount <= 4)
            r -= 192;

''',
    '',
    'remove broad typed LMR refunds',
)

s = replace_once(
    s,
    '''        // Leviathan strength v8.1 - Witness Proof Contract. A remembered move
        // that concretely exposed search error is not just a hint. Revisit that one
        // move at full depth when its causal evidence is still live. This is tightly
        // scoped: one legal witness, bounded depth threshold, no global extension.
        constexpr auto leviathanProofWitnessCauses =
          Leviathan::Evidence::bit(Leviathan::Evidence::Kind::LMR_COUNTEREXAMPLE)
          | Leviathan::Evidence::bit(Leviathan::Evidence::Kind::PROBCUT_NEAR_PROOF)
          | Leviathan::Evidence::bit(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY);
        const bool leviathanWitnessNeedsFullProof =
          move == persistentWitness && leviathanEntryDepth >= 6
          && leviathanEntryDepth > leviathanWitnessCertifiedDepth
          && (leviathanWitnessEvidence & leviathanProofWitnessCauses);
        if (leviathanWitnessNeedsFullProof)
        {
            newDepth = std::max(newDepth, leviathanEntryDepth - 1);
            r = std::min(r, 0);
        }
''',
    '''        // Leviathan strength v8.4 - Causal-Minimal LMR Certificate.
        // Only a concrete move that previously demonstrated a large LMR error may
        // buy extra proof. Restrict to PV-relevant nodes and reopen at most every
        // two depth levels so the certificate amortizes work instead of causing
        // iterative-deepening re-search inflation.
        const bool leviathanLmrWitness =
          leviathanWitnessEvidence
          & Leviathan::Evidence::bit(Leviathan::Evidence::Kind::LMR_COUNTEREXAMPLE);
        const bool leviathanWitnessNeedsFullProof =
          move == persistentWitness && leviathanLmrWitness && (PvNode || ss->ttPv)
          && leviathanEntryDepth >= 7
          && (leviathanWitnessCertifiedDepth == 0
              || leviathanEntryDepth >= leviathanWitnessCertifiedDepth + 2);
        if (leviathanWitnessNeedsFullProof)
        {
            newDepth = std::max(newDepth, leviathanEntryDepth - 1);
            r = std::min(r, 0);
        }
''',
    'replace broad witness proof with LMR-only certificate',
)

s = replace_once(
    s,
    '''        if (depth >= 7 && leviathanProofDebt >= 4 && moveCount <= 2)
            r = std::min(r, 0);      // full depth (negative extension remains allowed)
        else if (depth >= 7 && leviathanEvidence.severe() && moveCount <= 3)
            r = std::min(r, 0);      // diverse severe evidence certifies one extra rival
        else if (depth >= 6 && leviathanProofDebt >= 3 && moveCount <= 4)
''',
    '''        if (depth >= 7 && leviathanProofDebt >= 4 && moveCount <= 2)
            r = std::min(r, 0);      // full depth (negative extension remains allowed)
        else if (depth >= 6 && leviathanProofDebt >= 3 && moveCount <= 4)
''',
    'restore V7 proof regime width',
)

sp.write_text(s)

# Guard against accidental authority leakage.
final = sp.read_text()
for forbidden in [
    '!Leviathan::Evidence::tt_sensitive(leviathanEvidence)',
    'Leviathan::Evidence::pruning_sensitive(leviathanEvidence) && moveCount <= 4',
    'Leviathan::Evidence::rival_sensitive(leviathanEvidence) && moveCount <= 4',
    'leviathanEvidence.severe() && moveCount <= 3',
]:
    if forbidden in final:
        raise SystemExit(f'forbidden broad V8 authority remains: {forbidden}')
for required in [
    'leviathanLmrWitness',
    'leviathanEntryDepth >= leviathanWitnessCertifiedDepth + 2',
    'Leviathan::Evidence::State leviathanEvidence',
    'leviathanWitnessEvidence',
]:
    if required not in final:
        raise SystemExit(f'missing V8.4 invariant: {required}')
print('V8.4 causal-minimal authority patch applied')

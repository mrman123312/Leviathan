#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


sp = Path("src/search.cpp")
s = sp.read_text()

# Typed evidence must age with its confidence. A stale bitmask must never combine
# with fresh evidence later and impersonate independent confirmation.
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
                        e.evidence = 0;
                        e.witness  = Move::none();
                    }
                }
''',
    "typed evidence decay hygiene",
)

# A position that previously produced a large TT/static-eval disagreement should
# not immediately let the same TT bound replace static eval on the revisit.
s = replace_once(
    s,
    '''        if (!pos.has_repeated() && is_valid(ttData.value)
            && (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttData.value;
''',
    '''        if (!pos.has_repeated()
            && !leviathanEvidence.contains(Leviathan::Evidence::Kind::EVAL_DISAGREEMENT)
            && is_valid(ttData.value)
            && (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttData.value;
''',
    "TT eval quarantine",
)

# A concrete witness is stronger than generic uncertainty. If that move previously
# exposed LMR, ProbCut, or rival-selection failure, give exactly that move a full-depth
# proof on revisit rather than globally inflating the node.
anchor = '''        // Leviathan strength v6.4 - Proof Regime. Once several independent
        // warnings agree, incremental buybacks are not enough: accumulated LMR
'''
insert = '''        // Leviathan strength v8.1 - Witness Proof Contract. A remembered move
        // that concretely exposed search error is not just a hint. Revisit that one
        // move at full depth when its causal evidence is still live. This is tightly
        // scoped: one legal witness, bounded depth threshold, no global extension.
        const bool leviathanWitnessNeedsFullProof =
          move == persistentWitness && depth >= 6
          && (leviathanEvidence.contains(Leviathan::Evidence::Kind::LMR_COUNTEREXAMPLE)
              || leviathanEvidence.contains(Leviathan::Evidence::Kind::PROBCUT_NEAR_PROOF)
              || leviathanEvidence.contains(Leviathan::Evidence::Kind::RIVAL_AMBIGUITY));
        if (leviathanWitnessNeedsFullProof)
            r = std::min(r, 0);

'''
if s.count(anchor) != 1:
    raise SystemExit(f"witness proof anchor: expected exactly one, found {s.count(anchor)}")
s = s.replace(anchor, insert + anchor, 1)

sp.write_text(s)

final = sp.read_text()
for label, needle in {
    "evidence decay": "e.evidence = 0;",
    "TT quarantine": "Kind::EVAL_DISAGREEMENT",
    "witness contract": "leviathanWitnessNeedsFullProof",
}.items():
    if needle not in final:
        raise SystemExit(f"missing V8.1 invariant: {label}")

print("V8.1 evidence hygiene and witness proof contract applied")

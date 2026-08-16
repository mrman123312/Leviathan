#!/usr/bin/env python3
from pathlib import Path

p = Path('src/search.cpp')
s = p.read_text()
old = '''                if (e.debt <= decay)
                    e = {};
                else
                    e.debt -= decay;
'''
new = '''                if (e.debt <= decay)
                    e = {};
                else
                {
                    e.debt -= decay;
                    // V8.5A causal ablation: once the stored warning is below the
                    // actionable Proof-Debt threshold, its witness must not retain
                    // ordering/protection authority by itself. Keep the residual
                    // debt for normal V7 behavior, but retire the stale move identity.
                    if (e.debt < 3)
                        e.witness = Move::none();
                }
'''
if s.count(old) != 1:
    raise SystemExit(f'hygiene anchor count {s.count(old)}')
s = s.replace(old, new, 1)
p.write_text(s)

final = p.read_text()
if 'if (e.debt < 3)\n                        e.witness = Move::none();' not in final:
    raise SystemExit('hygiene invariant missing')
for forbidden in ['witnessWasLmr', 'witnessDepth', 'leviathanEntryDepth', 'leviathanWitnessNeedsFullProof']:
    if forbidden in final:
        raise SystemExit(f'hygiene-only branch leaked receipt behavior: {forbidden}')
print('V8.5A applied: stale witness cleanup only')

#!/usr/bin/env python3
from pathlib import Path
p=Path('src/search.cpp')
s=p.read_text()
anchor='''                // Post LMR continuation history updates
                update_continuation_histories(ss, movedPiece, move.to_sq(), 1334);
            }
        }

        // Step 18. Full-depth search when LMR is skipped
'''
repl='''                // Post LMR continuation history updates
                update_continuation_histories(ss, movedPiece, move.to_sq(), 1334);
            }
            // Leviathan Margin Skeptic: a reduced quiet move that fails low only
            // narrowly may be reduction-uncertain. Re-open only bounded, early-ish
            // near-alpha cases instead of weakening LMR globally.
            else if (d < newDepth && depth >= 6 && moveCount <= 10 && !capture && !givesCheck
                     && value >= alpha - (18 + 3 * std::min(int(depth), 12))
                     && (PvNode || ss->ttPv
                         || Leviathan::Fundamentals::quiet_tactical_tension(
                           pos, move, capture, givesCheck)))
            {
                leviathanResearched = true;
                value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
                if (value > alpha)
                    update_continuation_histories(ss, movedPiece, move.to_sq(), 1334);
            }
        }

        // Step 18. Full-depth search when LMR is skipped
'''
if s.count(anchor)!=1:
    raise SystemExit(f'anchor count={s.count(anchor)}')
p.write_text(s.replace(anchor,repl,1))
print('Margin Skeptic applied')

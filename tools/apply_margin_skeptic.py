#!/usr/bin/env python3
from pathlib import Path

# Phase-fix already has moving_piece(). Add only the factual post-move tension
# predicate needed by Margin Skeptic; do not import the cumulative v5 header.
h=Path('src/leviathan_fundamentals.h');s=h.read_text()
anchor='''inline bool zeroing_quiet(const Position& pos, Move move, bool capture) {
    return capture || pawn_move(pos, move);
}

inline bool protected_scope_move'''
repl='''inline bool zeroing_quiet(const Position& pos, Move move, bool capture) {
    return capture || pawn_move(pos, move);
}

inline bool quiet_tactical_tension(const Position& pos,
                                   Move move,
                                   bool capture,
                                   bool givesCheck) {
    if (capture || givesCheck)
        return false;
    Piece pc = moving_piece(pos, move);
    if (pc == NO_PIECE || type_of(pc) == PAWN || type_of(pc) == KING)
        return false;
    Color us = color_of(pc);
    return pos.attackers_to_exist(move.to_sq(), pos.pieces(), ~us);
}

inline bool protected_scope_move'''
if s.count(anchor)!=1:
    raise SystemExit(f'header anchor count={s.count(anchor)}')
h.write_text(s.replace(anchor,repl,1))

p=Path('src/search.cpp');s=p.read_text()
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
    raise SystemExit(f'search anchor count={s.count(anchor)}')
p.write_text(s.replace(anchor,repl,1))
print('Margin Skeptic + minimal phase-fix helper applied')

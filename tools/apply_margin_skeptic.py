#!/usr/bin/env python3
from pathlib import Path

# Backport only the factual helper predicates required by Margin Skeptic.
h=Path('src/leviathan_fundamentals.h');s=h.read_text()
anchor='''inline bool zeroing_quiet(const Position& pos, Move move, bool capture) {
    return capture || (move.is_ok() && type_of(pos.moved_piece(move)) == PAWN);
}

inline bool protected_scope_move'''
repl='''inline bool zeroing_quiet(const Position& pos, Move move, bool capture) {
    return capture || (move.is_ok() && type_of(pos.moved_piece(move)) == PAWN);
}

inline Piece margin_moving_piece(const Position& pos, Move move) {
    if (!move.is_ok())
        return NO_PIECE;
    Piece pc = pos.piece_on(move.from_sq());
    return pc != NO_PIECE ? pc : pos.piece_on(move.to_sq());
}

inline bool quiet_tactical_tension(const Position& pos,
                                   Move move,
                                   bool capture,
                                   bool givesCheck) {
    if (capture || givesCheck)
        return false;
    Piece pc = margin_moving_piece(pos, move);
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
print('Margin Skeptic + minimal helper dependencies applied')

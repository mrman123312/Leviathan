/*
  Project Leviathan deterministic fundamentals layer.

  This layer is intentionally non-neural. It classifies cheap chess/search
  regimes from information Stockfish already has in the hot path and can
  redistribute search depth between forcing/dangerous moves and quiet/stable
  late moves. Authority 1 is rescue-only. Authority 2 may also overdrive safe
  quiet branches to pay for the rescued scope.
*/
#ifndef LEVIATHAN_FUNDAMENTALS_H_INCLUDED
#define LEVIATHAN_FUNDAMENTALS_H_INCLUDED

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>

#include "position.h"
#include "types.h"

namespace Stockfish::Leviathan::Fundamentals {

struct State {
    bool enabled = false;
    int  authority = 0;  // 0 off, 1 rescue-only, 2 balanced allocator

    int forcingBuyback = 384;     // reduction units, 1024 ~= one ply
    int recaptureBuyback = 256;
    int passerBuyback = 320;
    int endgameBuyback = 128;
    int quietOverdrive = 160;
    int rule50PawnBonus = 3072;

    bool zugzwangGuard = true;
    bool sacrificeRescue = true;
    bool rule50Pressure = true;

    State() {
        if (const char* e = std::getenv("LEVIATHAN_FUNDAMENTALS"))
            enabled = *e && std::string(e) != "0" && std::string(e) != "false";
        if (const char* a = std::getenv("LEVIATHAN_FUNDAMENTALS_AUTHORITY"))
            authority = std::clamp(std::atoi(a), 0, 2);
    }
};

// The engine constructs and configures this process-wide state before search.
// An inline variable avoids the thread-safe initialization-guard branch that a
// function-local non-trivial static otherwise places on every hot state() call.
inline State globalState;
inline State& state() { return globalState; }

inline void set_enabled(bool v) { state().enabled = v; }
inline void set_authority(int v) { state().authority = std::clamp(v, 0, 2); }
inline void set_forcing_buyback(int v) { state().forcingBuyback = std::clamp(v, 0, 2048); }
inline void set_recapture_buyback(int v) { state().recaptureBuyback = std::clamp(v, 0, 2048); }
inline void set_passer_buyback(int v) { state().passerBuyback = std::clamp(v, 0, 2048); }
inline void set_endgame_buyback(int v) { state().endgameBuyback = std::clamp(v, 0, 1024); }
inline void set_quiet_overdrive(int v) { state().quietOverdrive = std::clamp(v, 0, 1024); }
inline void set_rule50_pawn_bonus(int v) { state().rule50PawnBonus = std::clamp(v, 0, 16384); }
inline void set_zugzwang_guard(bool v) { state().zugzwangGuard = v; }
inline void set_sacrifice_rescue(bool v) { state().sacrificeRescue = v; }
inline void set_rule50_pressure(bool v) { state().rule50Pressure = v; }

inline bool ready() { return state().enabled && state().authority > 0; }

inline int non_pawn_count(const Position& pos) {
    return pos.count<KNIGHT>() + pos.count<BISHOP>() + pos.count<ROOK>() + pos.count<QUEEN>();
}

inline bool low_material(const Position& pos) {
    return pos.count<ALL_PIECES>() <= 10 || non_pawn_count(pos) <= 3;
}

inline bool zugzwang_risk(const Position& pos) {
    if (!ready() || !state().zugzwangGuard)
        return false;

    // Conservative detector: sparse positions with very little non-pawn
    // mobility are where null-move assumptions are structurally least safe.
    return pos.count<ALL_PIECES>() <= 9 && non_pawn_count(pos) <= 2;
}

inline Piece moving_piece(const Position& pos, Move move) {
    if (!move.is_ok())
        return NO_PIECE;
    Piece pc = pos.piece_on(move.from_sq());
    return pc != NO_PIECE ? pc : pos.piece_on(move.to_sq());
}

inline bool pawn_move(const Position& pos, Move move) {
    if (!move.is_ok())
        return false;
    if (move.type_of() == PROMOTION || move.type_of() == EN_PASSANT)
        return true;
    Piece pc = moving_piece(pos, move);
    return pc != NO_PIECE && type_of(pc) == PAWN;
}

inline Color mover_color(const Position& pos, Move move) {
    Piece pc = moving_piece(pos, move);
    return pc != NO_PIECE ? color_of(pc) : ~pos.side_to_move();
}

inline int king_distance(Square a, Square b) {
    return std::max(std::abs(int(file_of(a)) - int(file_of(b))),
                    std::abs(int(rank_of(a)) - int(rank_of(b))));
}

inline bool advanced_pawn_move(const Position& pos, Move move) {
    return pawn_move(pos, move) && relative_rank(mover_color(pos, move), move.to_sq()) >= RANK_6;
}

// Rank-five pawn commitments are often the point where a quiet move stops being
// reversible: it can fix a structure, create a protected passer, or begin a
// promotion race. Stockfish's local history/SEE signals can undervalue them well
// before they become a rank-six tactical event.
inline bool critical_pawn_commitment(const Position& pos, Move move) {
    return pawn_move(pos, move) && relative_rank(mover_color(pos, move), move.to_sq()) >= RANK_5;
}

inline bool recapture(Move move, Square prevSq, bool capture) {
    return capture && prevSq != SQ_NONE && move.to_sq() == prevSq;
}

inline bool zeroing_quiet(const Position& pos, Move move, bool capture) {
    return capture || pawn_move(pos, move);
}

// Quiet moves around a king are disproportionately likely to be prophylaxis,
// mating preparation, defender removal preparation, or an escape-square change.
// Preserve them as candidates; alpha-beta still decides whether they are good.
inline bool king_zone_maneuver(const Position& pos, Move move) {
    Piece pc = moving_piece(pos, move);
    if (pc == NO_PIECE || type_of(pc) == PAWN || type_of(pc) == KING)
        return false;

    Color us = color_of(pc);
    return king_distance(move.to_sq(), pos.square<KING>(~us)) <= 2;
}

// A quiet move by a currently attacked non-pawn piece is a high-regret pruning
// class: the obvious capture/retreat candidates dominate history, while a rare
// tactical retreat, interposition, or counter-threat can look locally ordinary.
inline bool threatened_piece_quiet(const Position& pos,
                                   Move move,
                                   bool capture,
                                   bool givesCheck) {
    if (capture || givesCheck)
        return false;

    Piece pc = moving_piece(pos, move);
    if (pc == NO_PIECE || type_of(pc) == PAWN || type_of(pc) == KING)
        return false;

    Color us = color_of(pc);
    return pos.attackers_to_exist(move.from_sq(), pos.pieces(), ~us);
}

inline bool precision_king_move(const Position& pos, Move move, bool capture, bool givesCheck) {
    if (capture || givesCheck)
        return false;

    Piece pc = moving_piece(pos, move);
    if (pc == NO_PIECE || type_of(pc) != KING)
        return false;

    // King geometry dominates sparse endings and long rule-50 conversions.
    return low_material(pos) || pos.rule50_count() >= 50;
}

// Called before do_move() from shallow-pruning gates. This is deliberately a
// candidate-preservation predicate, not an evaluation claim. Its job is to stop
// irreversible, tactical-boundary, and king-geometry moves from disappearing
// before the full search has a chance to judge them.
inline bool protected_scope_move(const Position& pos,
                                 Move move,
                                 Square prevSq,
                                 bool capture,
                                 bool givesCheck) {
    if (!ready())
        return false;

    return givesCheck || move.type_of() == PROMOTION || critical_pawn_commitment(pos, move)
        || recapture(move, prevSq, capture)
        || threatened_piece_quiet(pos, move, capture, givesCheck)
        || king_zone_maneuver(pos, move)
        || precision_king_move(pos, move, capture, givesCheck);
}

// Soundness policy: this does not claim a move is good. It only says the move
// belongs to a class where pruning mistakes are unusually expensive.
inline bool rescue_bad_see(const Position& pos,
                           Move move,
                           Square prevSq,
                           bool capture,
                           bool givesCheck) {
    if (!ready() || !state().sacrificeRescue)
        return false;

    if (move.type_of() == PROMOTION)
        return true;

    // v2.1: a recapture is not automatically worthy of rescue. Keep only
    // near-balanced recaptures; obviously losing exchanges should still be
    // allowed to die in SEE pruning instead of inflating the tree.
    if (recapture(move, prevSq, capture))
        return pos.see_ge(move, -300);

    // Checking sacrifices retain the broad rescue path. They are rare and are
    // exactly where a purely material SEE gate can miss forced tactical ideas.
    if (givesCheck && !pos.see_ge(move, 0))
        return true;

    // Strength v5: non-checking sacrifices close to the enemy king are a classic
    // delayed-compensation failure mode. SEE sees the immediate material loss;
    // it does not see line opening, removal of a key defender, or a mating net a
    // few plies later. Rescue only this geometrically narrow family.
    if (capture)
    {
        Piece pc = moving_piece(pos, move);
        if (pc != NO_PIECE && type_of(pc) != KING)
        {
            Color us = color_of(pc);
            if (king_distance(move.to_sq(), pos.square<KING>(~us)) <= 2)
                return true;
        }
    }

    return false;
}

// Post-move detector: if a quiet non-pawn move deliberately leaves the moved
// piece inside enemy attack, the line contains tactical tension that ordinary
// quiet-history reductions are poorly suited to summarize. This covers quiet
// sacrifices and interference moves without granting them move authority.
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

inline int lmr_adjustment(const Position& pos,
                          Move move,
                          Square prevSq,
                          Depth depth,
                          int moveCount,
                          bool pvNode,
                          bool capture,
                          bool givesCheck) {
    if (!ready())
        return 0;

    int delta = 0;

    if (givesCheck)
        delta -= state().forcingBuyback;
    if (recapture(move, prevSq, capture))
        delta -= state().recaptureBuyback;
    if (move.type_of() == PROMOTION || advanced_pawn_move(pos, move))
        delta -= state().passerBuyback;
    else if (critical_pawn_commitment(pos, move))
        delta -= std::max(96, state().passerBuyback / 2);

    // v2.1: blanket endgame buyback made every sparse branch expensive. Keep
    // the extra protection concentrated on early candidates and forcing moves.
    if (low_material(pos) && (moveCount <= 4 || capture || givesCheck))
        delta -= state().endgameBuyback;

    // Strength v5: preserve close PV rivals instead of assuming late quiet
    // history is enough evidence. This is deliberately sub-ply: it buys a little
    // more proof only where the node is already on the principal-value frontier.
    if (pvNode && depth >= 6 && moveCount >= 2 && moveCount <= 8 && !capture && !givesCheck)
        delta -= 160;

    // Search quiet attacking/prophylactic maneuvers near the enemy king more
    // seriously. These moves often have delayed value and weak local history.
    if (!capture && !givesCheck && depth >= 5 && moveCount <= 10 && king_zone_maneuver(pos, move))
        delta -= std::max(128, state().forcingBuyback / 2);

    // Quiet sacrifices/interference moves are precisely where a low-history LMR
    // can hide compensation beyond the reduced horizon.
    if (depth >= 6 && moveCount <= 10 && quiet_tactical_tension(pos, move, capture, givesCheck))
        delta -= 256;

    // Sparse king moves are tempo/proof moves, not ordinary quiets.
    if (depth >= 5 && precision_king_move(pos, move, capture, givesCheck))
        delta -= std::max(96, state().endgameBuyback);

    // v2.1.2: the v2.1 post-move identity bug accidentally let ordinary pawn
    // moves participate in the speed budget, and the 100-game phase-fix showed
    // that globally excluding every pawn was too expensive at 100 ms. Preserve
    // that useful selectivity, but now classify the mover correctly so sixth-
    // rank/seventh-rank pawn moves and promotions are explicitly protected.
    // This creates a real irreversibility gradient instead of all-pawn/all-safe.
    if (state().authority >= 2 && depth >= 5 && moveCount >= 6 && !pvNode && !capture
        && !givesCheck && move.type_of() != PROMOTION && !advanced_pawn_move(pos, move)
        && pos.rule50_count() < 70 && !low_material(pos))
    {
        const int lateness   = std::min(moveCount - 5, 8);
        const int depthScale = std::min(int(depth), 10) + 2;
        delta += state().quietOverdrive * lateness * depthScale / 48;
    }

    return std::clamp(delta, -2048, 768);
}

inline int quiet_ordering_bonus(const Position& pos, Move move) {
    if (!ready() || !state().rule50Pressure || pos.rule50_count() < 70)
        return 0;

    // Captures are scored in another MovePicker stage. Push pawn moves upward
    // when the fifty-move counter is dangerous because they reset it.
    return pawn_move(pos, move) ? state().rule50PawnBonus : 0;
}

inline bool allow_null_move(const Position& pos) {
    return !zugzwang_risk(pos);
}

struct Counters {
    std::atomic<unsigned long long> forcing{0};
    std::atomic<unsigned long long> recaptures{0};
    std::atomic<unsigned long long> passers{0};
    std::atomic<unsigned long long> quietOverdrives{0};
    std::atomic<unsigned long long> nullGuards{0};
    std::atomic<unsigned long long> pruneRescues{0};
};

inline Counters& counters() {
    static Counters c;
    return c;
}

}  // namespace Stockfish::Leviathan::Fundamentals

#endif

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

inline State& state() {
    static State s;
    return s;
}

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
    if (!move.is_ok()) return NO_PIECE;
    Piece pc = pos.piece_on(move.from_sq());
    return pc != NO_PIECE ? pc : pos.piece_on(move.to_sq());
}

inline bool pawn_move(const Position& pos, Move move) {
    if (!move.is_ok()) return false;
    if (move.type_of() == PROMOTION || move.type_of() == EN_PASSANT) return true;
    Piece pc = moving_piece(pos, move);
    return pc != NO_PIECE && type_of(pc) == PAWN;
}

inline Color mover_color(const Position& pos, Move move) {
    Piece pc = moving_piece(pos, move);
    return pc != NO_PIECE ? color_of(pc) : pos.side_to_move();
}

inline bool advanced_pawn_move(const Position& pos, Move move) {
    return pawn_move(pos, move) && relative_rank(mover_color(pos, move), move.to_sq()) >= RANK_6;
}

inline bool recapture(Move move, Square prevSq, bool capture) {
    return capture && prevSq != SQ_NONE && move.to_sq() == prevSq;
}

inline bool zeroing_quiet(const Position& pos, Move move, bool capture) {
    return capture || pawn_move(pos, move);
}

inline bool protected_scope_move(const Position& pos,
                                 Move move,
                                 Square prevSq,
                                 bool capture,
                                 bool givesCheck) {
    if (!ready())
        return false;

    return givesCheck || move.type_of() == PROMOTION || advanced_pawn_move(pos, move)
        || recapture(move, prevSq, capture);
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
    return givesCheck && !pos.see_ge(move, 0);
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

    // v2.1: blanket endgame buyback made every sparse branch expensive. Keep
    // the extra protection concentrated on early candidates and forcing moves.
    if (low_material(pos) && (moveCount <= 4 || capture || givesCheck))
        delta -= state().endgameBuyback;

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

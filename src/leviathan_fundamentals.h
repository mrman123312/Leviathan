/*
  Project Leviathan deterministic fundamentals layer.

  P09 race profile: the competition binary is intentionally fixed to the
  already-tested configuration (Fundamentals enabled, Authority 1,
  sacrifice/zugzwang/rule50 guards enabled, Quiet Overdrive unreachable).
  The generic P01 branch remains the research/configurable rollback path.
*/
#ifndef LEVIATHAN_FUNDAMENTALS_H_INCLUDED
#define LEVIATHAN_FUNDAMENTALS_H_INCLUDED

#include <algorithm>
#include <atomic>

#include "position.h"
#include "types.h"

namespace Stockfish::Leviathan::Fundamentals {

struct State {
    bool enabled = true;
    int  authority = 1;

    int forcingBuyback = 384;
    int recaptureBuyback = 256;
    int passerBuyback = 320;
    int endgameBuyback = 128;
    int quietOverdrive = 0;
    int rule50PawnBonus = 3072;

    bool zugzwangGuard = true;
    bool sacrificeRescue = true;
    static constexpr bool rule50Pressure = true;
};

inline State& state() {
    static State s;
    return s;
}

// Compatibility setters remain for the UCI/research surface. Race-hot decisions
// below are compile-time fixed; changes to these fields do not alter that profile.
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
inline void set_rule50_pressure(bool) {}

inline constexpr bool ready() { return true; }

inline int non_pawn_count(const Position& pos) {
    return pos.count<KNIGHT>() + pos.count<BISHOP>() + pos.count<ROOK>() + pos.count<QUEEN>();
}

inline bool low_material(const Position& pos) {
    return pos.count<ALL_PIECES>() <= 10 || non_pawn_count(pos) <= 3;
}

inline bool zugzwang_risk(const Position& pos) {
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

inline bool protected_scope_move(const Position& pos,
                                 Move move,
                                 Square prevSq,
                                 bool capture,
                                 bool givesCheck) {
    return givesCheck || move.type_of() == PROMOTION || advanced_pawn_move(pos, move)
        || recapture(move, prevSq, capture);
}

inline bool rescue_bad_see(const Position& pos,
                           Move move,
                           Square prevSq,
                           bool capture,
                           bool givesCheck) {
    if (move.type_of() == PROMOTION)
        return true;
    if (recapture(move, prevSq, capture))
        return pos.see_ge(move, -300);
    return givesCheck && !pos.see_ge(move, 0);
}

inline int lmr_adjustment(const Position& pos,
                          Move move,
                          Square prevSq,
                          Depth,
                          int moveCount,
                          bool,
                          bool capture,
                          bool givesCheck) {
    int delta = 0;
    if (givesCheck)
        delta -= 384;
    if (recapture(move, prevSq, capture))
        delta -= 256;
    if (move.type_of() == PROMOTION || advanced_pawn_move(pos, move))
        delta -= 320;
    if (low_material(pos) && (moveCount <= 4 || capture || givesCheck))
        delta -= 128;
    return std::clamp(delta, -2048, 768);
}

inline int quiet_ordering_bonus(const Position& pos, Move move) {
    if (pos.rule50_count() < 70)
        return 0;
    return pawn_move(pos, move) ? 3072 : 0;
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

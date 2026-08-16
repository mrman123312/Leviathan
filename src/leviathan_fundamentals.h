/*
  Project Leviathan deterministic fundamentals layer.

  v2.2 turns the original move-category rescue layer into a cheap dimensional
  allocator. The layer still never changes legality, evaluation, terminal
  values, TT semantics, or NNUE. It only changes move ordering, pruning vetoes,
  null-move permission, and bounded LMR allocation.

  Core rule:
      spend depth where being wrong is asymmetric;
      recover the budget from genuinely stable late quiet branches.

  v2.2 also fixes a v2.1 phase bug: lmr_adjustment() is called after do_move(),
  so reading pos.moved_piece(move) there reads the now-empty from square. All
  move classification below is deliberately valid both before and after a move.
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

    int forcingBuyback = 384;  // reduction units, 1024 ~= one ply
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

inline int square_distance(Square a, Square b) {
    return std::max(std::abs(int(file_of(a)) - int(file_of(b))),
                    std::abs(int(rank_of(a)) - int(rank_of(b))));
}

// Safe on both sides of do_move(). Before the move, the from square contains
// the mover. After the move it is empty and the to square contains the mover
// (or promoted piece).
inline Piece moving_piece(const Position& pos, Move move) {
    if (!move.is_ok())
        return NO_PIECE;

    Piece pc = pos.piece_on(move.from_sq());
    return pc != NO_PIECE ? pc : pos.piece_on(move.to_sq());
}

inline Color mover_color(const Position& pos, Move move) {
    Piece pc = moving_piece(pos, move);
    return pc != NO_PIECE ? color_of(pc) : pos.side_to_move();
}

inline bool pawn_move(const Position& pos, Move move) {
    if (!move.is_ok())
        return false;
    if (move.type_of() == PROMOTION || move.type_of() == EN_PASSANT)
        return true;
    Piece pc = moving_piece(pos, move);
    return pc != NO_PIECE && type_of(pc) == PAWN;
}

inline bool king_move(const Position& pos, Move move) {
    Piece pc = moving_piece(pos, move);
    return pc != NO_PIECE && type_of(pc) == KING;
}

inline bool heavy_move(const Position& pos, Move move) {
    Piece pc = moving_piece(pos, move);
    if (pc == NO_PIECE)
        return false;
    PieceType pt = type_of(pc);
    return pt == ROOK || pt == QUEEN;
}

inline bool advanced_pawn_move(const Position& pos, Move move) {
    if (!pawn_move(pos, move))
        return false;
    return relative_rank(mover_color(pos, move), move.to_sq()) >= RANK_6;
}

inline bool recapture(Move move, Square prevSq, bool capture) {
    return capture && prevSq != SQ_NONE && move.to_sq() == prevSq;
}

inline bool zeroing_quiet(const Position& pos, Move move, bool capture) {
    return capture || pawn_move(pos, move);
}

inline Bitboard advanced_pawn_candidates(const Position& pos, Color c) {
    constexpr Bitboard WhiteAdvanced = Rank4BB | Rank5BB | Rank6BB | Rank7BB;
    constexpr Bitboard BlackAdvanced = Rank2BB | Rank3BB | Rank4BB | Rank5BB;
    return pos.pieces(c, PAWN) & (c == WHITE ? WhiteAdvanced : BlackAdvanced);
}

// Passed-pawn geometry is deliberately simple and cheap. This is not an
// evaluation term; it only identifies branches where pruning errors have an
// unusually high cost.
inline bool passed_pawn(const Position& pos, Color c, Square s) {
    Bitboard file = file_bb(s);
    Bitboard files = file | shift<EAST>(file) | shift<WEST>(file);
    Bitboard enemyPawns = pos.pieces(~c, PAWN) & files;
    const int ourRank = int(relative_rank(c, s));

    while (enemyPawns)
    {
        Square e = pop_lsb(enemyPawns);
        if (int(relative_rank(c, e)) > ourRank)
            return false;
    }
    return true;
}

struct PasserThreat {
    Square sq = SQ_NONE;
    int urgency = 0;  // 0 none, 1 watch, 2 serious, 3 critical, 4 promotion-imminent
};

inline PasserThreat most_urgent_passer(const Position& pos, Color c) {
    PasserThreat best;
    Bitboard pawns = advanced_pawn_candidates(pos, c);

    while (pawns)
    {
        Square s = pop_lsb(pawns);
        if (!passed_pawn(pos, c, s))
            continue;

        Rank rr = relative_rank(c, s);
        int urgency = rr >= RANK_7 ? 4 : rr >= RANK_6 ? 3 : rr >= RANK_5 ? 2 : 1;
        if (urgency > best.urgency)
            best = {s, urgency};
    }

    return best;
}

inline PasserThreat enemy_passer_threat(const Position& pos, Move move) {
    const Color enemy = ~mover_color(pos, move);
    if (!advanced_pawn_candidates(pos, enemy))
        return {};
    return most_urgent_passer(pos, enemy);
}

inline bool moves_toward(Square from, Square to, Square target) {
    return square_distance(to, target) < square_distance(from, target);
}

inline bool passer_defense_move(const Position& pos, Move move, const PasserThreat& threat) {
    if (!move.is_ok() || threat.urgency < 2)
        return false;

    const Color enemy = ~mover_color(pos, move);
    const Square to = move.to_sq();
    const Square from = move.from_sq();
    const Square blockSq = Square(int(threat.sq) + int(pawn_push(enemy)));

    if (is_ok(blockSq) && to == blockSq)
        return true;

    if (king_move(pos, move))
    {
        Square promotionSq = make_square(file_of(threat.sq), enemy == WHITE ? RANK_8 : RANK_1);
        return moves_toward(from, to, threat.sq) || moves_toward(from, to, promotionSq);
    }

    if (heavy_move(pos, move))
        return file_of(to) == file_of(threat.sq) || rank_of(to) == rank_of(threat.sq);

    return false;
}

inline bool own_passer_support_move(const Position& pos, Move move, const PasserThreat& own) {
    return king_move(pos, move) && own.urgency >= 2
        && moves_toward(move.from_sq(), move.to_sq(), own.sq);
}

inline bool active_heavy_move(const Position& pos,
                              Move move,
                              bool givesCheck,
                              const PasserThreat& threat) {
    if (!heavy_move(pos, move) || !low_material(pos))
        return false;

    if (givesCheck)
        return true;

    const Color us = mover_color(pos, move);
    if (relative_rank(us, move.to_sq()) >= RANK_7)
        return true;

    return threat.urgency >= 3
        && (file_of(move.to_sq()) == file_of(threat.sq)
            || rank_of(move.to_sq()) == rank_of(threat.sq));
}

inline bool stable_late_quiet(const Position& pos,
                              Move move,
                              const PasserThreat& threat,
                              Depth depth,
                              int moveCount,
                              bool pvNode,
                              bool capture,
                              bool givesCheck) {
    if (state().authority < 2 || depth < 5 || moveCount < 6 || pvNode || capture || givesCheck)
        return false;

    if (move.type_of() == PROMOTION || pawn_move(pos, move) || low_material(pos)
        || pos.rule50_count() >= 60 || pos.checkers() || threat.urgency >= 2)
        return false;

    return true;
}

inline bool zugzwang_risk(const Position& pos) {
    if (!ready() || !state().zugzwangGuard)
        return false;

    const int pieces = pos.count<ALL_PIECES>();
    const int nonPawns = non_pawn_count(pos);

    if (pieces <= 9 && nonPawns <= 2)
        return true;

    if (pieces <= 12 && nonPawns <= 2)
    {
        PasserThreat w = most_urgent_passer(pos, WHITE);
        PasserThreat b = most_urgent_passer(pos, BLACK);
        return std::max(w.urgency, b.urgency) >= 2;
    }

    return false;
}

inline bool protected_scope_move(const Position& pos,
                                 Move move,
                                 Square prevSq,
                                 bool capture,
                                 bool givesCheck) {
    if (!ready())
        return false;

    if (givesCheck || move.type_of() == PROMOTION || advanced_pawn_move(pos, move)
        || recapture(move, prevSq, capture))
        return true;

    const bool sparse = low_material(pos);
    PasserThreat threat = enemy_passer_threat(pos, move);
    if (passer_defense_move(pos, move, threat))
        return true;

    if (!sparse)
        return false;

    PasserThreat own = king_move(pos, move) ? most_urgent_passer(pos, mover_color(pos, move))
                                            : PasserThreat{};
    return own_passer_support_move(pos, move, own)
        || active_heavy_move(pos, move, givesCheck, threat);
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

    if (capture)
    {
        PasserThreat threat = enemy_passer_threat(pos, move);
        if (threat.urgency >= 3 && move.to_sq() == threat.sq && pos.see_ge(move, -600))
            return true;
    }

    if (recapture(move, prevSq, capture))
        return pos.see_ge(move, -300);

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
    const bool sparse = low_material(pos);
    const PasserThreat threat = enemy_passer_threat(pos, move);

    if (givesCheck)
    {
        delta -= state().forcingBuyback;
        if (sparse || threat.urgency >= 2)
            delta -= state().forcingBuyback / 2;
    }

    if (recapture(move, prevSq, capture))
        delta -= state().recaptureBuyback;

    if (move.type_of() == PROMOTION || advanced_pawn_move(pos, move))
        delta -= state().passerBuyback;

    if (passer_defense_move(pos, move, threat))
        delta -= state().passerBuyback + state().endgameBuyback;

    PasserThreat own;
    if (sparse && king_move(pos, move))
        own = most_urgent_passer(pos, mover_color(pos, move));

    if (own_passer_support_move(pos, move, own))
        delta -= state().endgameBuyback + state().passerBuyback / 2;

    if (active_heavy_move(pos, move, givesCheck, threat))
        delta -= state().endgameBuyback + state().forcingBuyback / 3;

    if (sparse && (moveCount <= 4 || capture || givesCheck))
        delta -= state().endgameBuyback;

    // Authority 2 funds the expanded scope by reducing only genuinely stable
    // late quiet branches. Pawn detection is correct after do_move(), so
    // irreversible pawn moves are never accidentally used as the speed budget.
    if (stable_late_quiet(pos, move, threat, depth, moveCount, pvNode, capture, givesCheck))
    {
        const int lateness = std::min(moveCount - 5, 10);
        const int depthScale = std::min(int(depth), 12) + 2;
        delta += state().quietOverdrive * lateness * depthScale / 36;
    }

    return std::clamp(delta, -2560, 1024);
}

inline int quiet_ordering_bonus(const Position& pos, Move move) {
    if (!ready())
        return 0;

    int bonus = 0;

    if (state().rule50Pressure && pos.rule50_count() >= 70 && pawn_move(pos, move))
        bonus += state().rule50PawnBonus;

    const bool sparse = low_material(pos);
    PasserThreat threat = enemy_passer_threat(pos, move);
    if (passer_defense_move(pos, move, threat))
        bonus += 12288;

    if (!sparse)
        return bonus;

    PasserThreat own = king_move(pos, move) ? most_urgent_passer(pos, mover_color(pos, move))
                                            : PasserThreat{};
    if (own_passer_support_move(pos, move, own))
        bonus += 4096;

    if (active_heavy_move(pos, move, false, threat))
        bonus += 6144;

    return bonus;
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

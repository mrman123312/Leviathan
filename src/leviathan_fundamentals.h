/*
  Project Leviathan deterministic fundamentals layer.

  v2.2 is a consequence-aware dimensional search allocator. It never changes
  legality, evaluation, terminal values, TT semantics, or NNUE. It changes only
  move ordering, pruning vetoes, null-move permission, and bounded LMR budget.

  Core rule:
      spend depth where being wrong is asymmetric;
      recover that depth from genuinely stable late quiet branches.

  v2.2 also fixes a v2.1 phase bug: lmr_adjustment() runs after do_move(), so
  reading pos.moved_piece(move) there reads the now-empty from square. Every
  move classifier below is valid both before and after do_move().
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

// Fast exact low-material predicate. With >=20 total pieces there must be at
// least four non-pawns because at most sixteen pawns can exist, so the expensive
// four-type popcount can be skipped in normal middlegames/openings.
inline bool low_material(const Position& pos) {
    const int pieces = pos.count<ALL_PIECES>();
    if (pieces <= 10)
        return true;
    if (pieces >= 20)
        return false;
    return non_pawn_count(pos) <= 3;
}

inline int square_distance(Square a, Square b) {
    return std::max(std::abs(int(file_of(a)) - int(file_of(b))),
                    std::abs(int(rank_of(a)) - int(rank_of(b))));
}

inline Piece moving_piece(const Position& pos, Move move) {
    if (!move.is_ok())
        return NO_PIECE;
    Piece pc = pos.piece_on(move.from_sq());
    return pc != NO_PIECE ? pc : pos.piece_on(move.to_sq());
}

inline bool recapture(Move move, Square prevSq, bool capture) {
    return capture && prevSq != SQ_NONE && move.to_sq() == prevSq;
}

inline Bitboard advanced_pawn_candidates(const Position& pos, Color c, bool broad) {
    constexpr Bitboard WhiteBroad  = Rank4BB | Rank5BB | Rank6BB | Rank7BB;
    constexpr Bitboard BlackBroad  = Rank2BB | Rank3BB | Rank4BB | Rank5BB;
    constexpr Bitboard WhiteUrgent = Rank6BB | Rank7BB;
    constexpr Bitboard BlackUrgent = Rank2BB | Rank3BB;
    const Bitboard mask = broad ? (c == WHITE ? WhiteBroad : BlackBroad)
                                : (c == WHITE ? WhiteUrgent : BlackUrgent);
    return pos.pieces(c, PAWN) & mask;
}

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
    int urgency = 0;  // 0 none, 1 watch, 2 serious, 3 critical, 4 imminent
};

inline PasserThreat most_urgent_passer(const Position& pos,
                                       Color c,
                                       bool broad,
                                       Bitboard candidates = 0) {
    PasserThreat best;
    Bitboard pawns = candidates ? candidates : advanced_pawn_candidates(pos, c, broad);

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

struct MoveState {
    Piece pc = NO_PIECE;
    PieceType pt = NO_PIECE_TYPE;
    Color us = WHITE;
    bool pawn = false;
    bool king = false;
    bool heavy = false;
    bool sparse = false;
    Bitboard enemyCandidates = 0;
    PasserThreat enemyPasser{};
};

inline MoveState classify(const Position& pos, Move move, bool needPasser = true) {
    MoveState m;
    m.pc = moving_piece(pos, move);
    if (m.pc != NO_PIECE)
    {
        m.pt = type_of(m.pc);
        m.us = color_of(m.pc);
    }
    else
        m.us = pos.side_to_move();

    m.pawn = move.is_ok() && (move.type_of() == PROMOTION || move.type_of() == EN_PASSANT
                              || m.pt == PAWN);
    m.king = m.pt == KING;
    m.heavy = m.pt == ROOK || m.pt == QUEEN;
    m.sparse = low_material(pos);

    if (needPasser)
    {
        const Color enemy = ~m.us;
        m.enemyCandidates = advanced_pawn_candidates(pos, enemy, m.sparse);
        if (m.enemyCandidates)
            m.enemyPasser = most_urgent_passer(pos, enemy, m.sparse, m.enemyCandidates);
    }
    return m;
}

inline bool advanced_pawn_move(const MoveState& m, Move move) {
    return m.pawn && relative_rank(m.us, move.to_sq()) >= RANK_6;
}

inline bool moves_toward(Square from, Square to, Square target) {
    return square_distance(to, target) < square_distance(from, target);
}

inline bool passer_defense_move(const Position& pos, Move move, const MoveState& m) {
    const PasserThreat& threat = m.enemyPasser;
    if (!move.is_ok() || threat.urgency < 2)
        return false;

    const Color enemy = ~m.us;
    const Square to = move.to_sq();
    const Square from = move.from_sq();
    const Square blockSq = Square(int(threat.sq) + int(pawn_push(enemy)));

    if (is_ok(blockSq) && to == blockSq)
        return true;

    if (m.king)
    {
        Square promotionSq = make_square(file_of(threat.sq), enemy == WHITE ? RANK_8 : RANK_1);
        return moves_toward(from, to, threat.sq) || moves_toward(from, to, promotionSq);
    }

    if (m.heavy)
        return file_of(to) == file_of(threat.sq) || rank_of(to) == rank_of(threat.sq);

    return false;
}

inline PasserThreat own_passer(const Position& pos, const MoveState& m) {
    if (!m.sparse || !m.king)
        return {};
    Bitboard candidates = advanced_pawn_candidates(pos, m.us, true);
    return candidates ? most_urgent_passer(pos, m.us, true, candidates) : PasserThreat{};
}

inline bool own_passer_support_move(Move move, const MoveState& m, const PasserThreat& own) {
    return m.king && own.urgency >= 2 && moves_toward(move.from_sq(), move.to_sq(), own.sq);
}

inline bool active_heavy_move(Move move, const MoveState& m, bool givesCheck) {
    if (!m.heavy || !m.sparse)
        return false;

    if (givesCheck || relative_rank(m.us, move.to_sq()) >= RANK_7)
        return true;

    return m.enemyPasser.urgency >= 3
        && (file_of(move.to_sq()) == file_of(m.enemyPasser.sq)
            || rank_of(move.to_sq()) == rank_of(m.enemyPasser.sq));
}

inline bool dangerous_pawn_push(const Position& pos, Move move, const MoveState& m) {
    if (move.type_of() == PROMOTION)
        return true;
    if (!advanced_pawn_move(m, move))
        return false;

    const Rank rr = relative_rank(m.us, move.to_sq());
    return rr >= RANK_7 || passed_pawn(pos, m.us, move.to_sq());
}

inline bool stable_late_quiet(const Position& pos,
                              Move move,
                              const MoveState& m,
                              Depth depth,
                              int moveCount,
                              bool pvNode,
                              bool capture,
                              bool givesCheck) {
    if (state().authority < 2 || depth < 5 || moveCount < 6 || pvNode || capture || givesCheck)
        return false;

    if (move.type_of() == PROMOTION || m.pawn || m.sparse || pos.rule50_count() >= 60
        || pos.checkers() || m.enemyPasser.urgency >= 2)
        return false;

    return true;
}

inline bool zugzwang_risk(const Position& pos) {
    if (!ready() || !state().zugzwangGuard)
        return false;

    const int pieces = pos.count<ALL_PIECES>();
    if (pieces >= 13)
        return false;

    const int nonPawns = non_pawn_count(pos);
    if (pieces <= 9 && nonPawns <= 2)
        return true;

    if (nonPawns <= 2)
    {
        Bitboard wc = advanced_pawn_candidates(pos, WHITE, true);
        Bitboard bc = advanced_pawn_candidates(pos, BLACK, true);
        PasserThreat w = wc ? most_urgent_passer(pos, WHITE, true, wc) : PasserThreat{};
        PasserThreat b = bc ? most_urgent_passer(pos, BLACK, true, bc) : PasserThreat{};
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

    MoveState m = classify(pos, move);
    if (givesCheck || move.type_of() == PROMOTION || dangerous_pawn_push(pos, move, m)
        || recapture(move, prevSq, capture) || passer_defense_move(pos, move, m))
        return true;

    if (!m.sparse)
        return false;

    PasserThreat own = own_passer(pos, m);
    return own_passer_support_move(move, m, own) || active_heavy_move(move, m, givesCheck);
}

inline bool rescue_bad_see(const Position& pos,
                           Move move,
                           Square prevSq,
                           bool capture,
                           bool givesCheck) {
    if (!ready() || !state().sacrificeRescue)
        return false;

    if (move.type_of() == PROMOTION)
        return true;

    // Captures usually do not need the full dimensional scan. Only construct it
    // when sparse geometry or an urgent enemy pawn can actually affect the veto.
    const bool sparse = low_material(pos);
    if (capture)
    {
        Piece pc = moving_piece(pos, move);
        Color us = pc != NO_PIECE ? color_of(pc) : pos.side_to_move();
        Bitboard candidates = advanced_pawn_candidates(pos, ~us, sparse);
        if (candidates)
        {
            PasserThreat threat = most_urgent_passer(pos, ~us, sparse, candidates);
            if (threat.urgency >= 3 && move.to_sq() == threat.sq && pos.see_ge(move, -600))
                return true;
        }
    }

    if (recapture(move, prevSq, capture))
        return sparse ? pos.see_ge(move, -300) : pos.see_ge(move, -100);

    return givesCheck && sparse && !pos.see_ge(move, 0);
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

    MoveState m = classify(pos, move);
    int delta = 0;
    const bool emergency = m.sparse || m.enemyPasser.urgency >= 2;

    if (givesCheck)
        delta -= emergency ? state().forcingBuyback + state().forcingBuyback / 2
                           : state().forcingBuyback / 4;

    if (recapture(move, prevSq, capture))
        delta -= emergency ? state().recaptureBuyback : state().recaptureBuyback / 4;

    if (dangerous_pawn_push(pos, move, m))
        delta -= emergency ? state().passerBuyback + state().passerBuyback / 2
                           : state().passerBuyback;

    if (passer_defense_move(pos, move, m))
        delta -= state().passerBuyback + state().endgameBuyback;

    PasserThreat own = own_passer(pos, m);
    if (own_passer_support_move(move, m, own))
        delta -= state().endgameBuyback + state().passerBuyback / 2;

    if (active_heavy_move(move, m, givesCheck))
        delta -= state().endgameBuyback + state().forcingBuyback / 3;

    if (m.sparse && (moveCount <= 3 || capture || givesCheck) && m.enemyPasser.urgency >= 1)
        delta -= state().endgameBuyback;

    // Hard budget model. Reversible, stable late quiet branches fund expensive
    // dimensional rescue. Irreversible pawn moves and fragile endings cannot.
    if (stable_late_quiet(pos, move, m, depth, moveCount, pvNode, capture, givesCheck))
    {
        const int lateness = std::min(moveCount - 5, 10);
        const int depthScale = std::min(int(depth), 12) + 2;
        delta += 512 + state().quietOverdrive * lateness * depthScale / 24;
        if (moveCount >= 10 && depth >= 7)
            delta += 256;
    }

    return std::clamp(delta, -2560, 1536);
}

inline int quiet_ordering_bonus(const Position& pos, Move move) {
    if (!ready())
        return 0;

    MoveState m = classify(pos, move);
    int bonus = 0;

    if (state().rule50Pressure && pos.rule50_count() >= 70 && m.pawn)
        bonus += state().rule50PawnBonus;

    if (passer_defense_move(pos, move, m))
        bonus += 12288;

    if (!m.sparse)
        return bonus;

    PasserThreat own = own_passer(pos, m);
    if (own_passer_support_move(move, m, own))
        bonus += 4096;
    if (active_heavy_move(move, m, false))
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

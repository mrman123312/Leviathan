#include "evaluator.h"
#include "distilled_eval_weights.h"
#include <algorithm>
#include <array>
#include <cstdlib>

namespace leviathan {
namespace {

int count_piece(const Position& p, Color c, PieceType pt) {
    int n = 0;
    const Piece target = make_piece(c, pt);
    for (int sq = 0; sq < 64; ++sq)
        if (p.piece_at(sq) == target) ++n;
    return n;
}

std::array<int,8> pawn_files(const Position& p, Color c) {
    std::array<int,8> files{};
    const Piece pawn = make_piece(c, PieceType::Pawn);
    for (int sq = 0; sq < 64; ++sq)
        if (p.piece_at(sq) == pawn) ++files[sq & 7];
    return files;
}

int doubled_pawns(const Position& p, Color c) {
    const auto files = pawn_files(p, c);
    int total = 0;
    for (int n : files) total += std::max(0, n - 1);
    return total;
}

int isolated_pawns(const Position& p, Color c) {
    const auto files = pawn_files(p, c);
    int total = 0;
    for (int f = 0; f < 8; ++f) {
        if (!files[f]) continue;
        const bool left = f > 0 && files[f - 1] != 0;
        const bool right = f < 7 && files[f + 1] != 0;
        if (!left && !right) total += files[f];
    }
    return total;
}

int passed_pawns(const Position& p, Color c) {
    const Piece ours = make_piece(c, PieceType::Pawn);
    const Piece enemy = make_piece(opposite(c), PieceType::Pawn);
    int total = 0;
    for (int sq = 0; sq < 64; ++sq) {
        if (p.piece_at(sq) != ours) continue;
        const int f = sq & 7;
        const int r = sq >> 3;
        bool passed = true;
        for (int esq = 0; esq < 64 && passed; ++esq) {
            if (p.piece_at(esq) != enemy) continue;
            const int ef = esq & 7;
            const int er = esq >> 3;
            if (std::abs(ef - f) > 1) continue;
            if ((c == Color::White && er > r) || (c == Color::Black && er < r))
                passed = false;
        }
        if (passed) ++total;
    }
    return total;
}

int king_shield(const Position& p, Color c) {
    const Piece king = make_piece(c, PieceType::King);
    const Piece pawn = make_piece(c, PieceType::Pawn);
    int kingSq = -1;
    for (int sq = 0; sq < 64; ++sq)
        if (p.piece_at(sq) == king) { kingSq = sq; break; }
    if (kingSq < 0) return 0;

    const int kf = kingSq & 7;
    const int kr = kingSq >> 3;
    const int dr = c == Color::White ? 1 : -1;
    int total = 0;
    for (int df = -1; df <= 1; ++df) {
        const int f = kf + df;
        const int r = kr + dr;
        if (f >= 0 && f < 8 && r >= 0 && r < 8 && p.piece_at(r * 8 + f) == pawn)
            ++total;
    }
    return total;
}

int distilled_correction_white(const Position& p) {
    int correction = 0;
    for (int sq = 0; sq < 64; ++sq) {
        const Piece piece = p.piece_at(sq);
        if (piece == Piece::Empty) continue;
        const Color c = color_of(piece);
        const int pt = static_cast<int>(type_of(piece));
        const int canonicalSq = c == Color::White ? sq : (sq ^ 56);
        const int idx = (pt - 1) * 64 + canonicalSq;
        correction += (c == Color::White ? 1 : -1) * distilled_eval::kPsqt[idx];
    }

    const int bishopPair = (count_piece(p, Color::White, PieceType::Bishop) >= 2 ? 1 : 0)
                         - (count_piece(p, Color::Black, PieceType::Bishop) >= 2 ? 1 : 0);
    const int doubled = doubled_pawns(p, Color::White) - doubled_pawns(p, Color::Black);
    const int isolated = isolated_pawns(p, Color::White) - isolated_pawns(p, Color::Black);
    const int passed = passed_pawns(p, Color::White) - passed_pawns(p, Color::Black);
    const uint8_t rights = p.castling_rights();
    const int whiteRights = ((rights & 1) ? 1 : 0) + ((rights & 2) ? 1 : 0);
    const int blackRights = ((rights & 4) ? 1 : 0) + ((rights & 8) ? 1 : 0);
    const int castlingRights = whiteRights - blackRights;
    const int shield = king_shield(p, Color::White) - king_shield(p, Color::Black);
    const int tempo = p.side_to_move() == Color::White ? 1 : -1;

    const int extras[] = {bishopPair, doubled, isolated, passed, castlingRights, shield, tempo};
    for (size_t i = 0; i < distilled_eval::kExtra.size(); ++i)
        correction += extras[i] * distilled_eval::kExtra[i];

    return std::clamp(correction, -distilled_eval::kMaxCorrection, distilled_eval::kMaxCorrection);
}

} // namespace

Evaluation BaselineEvaluator::evaluate(const Position& position) const {
    return evaluate_position(position);
}

const EvaluatorDescriptor& BaselineEvaluator::descriptor() const {
    static constexpr EvaluatorDescriptor d{
        "leviathan-baseline-v0",
        "native",
        "none",
        "Leviathan project",
        EvaluatorOrigin::Native
    };
    return d;
}

Evaluation DistilledEvaluator::evaluate(const Position& position) const {
    Evaluation out = evaluate_position(position);
    const int baselineWhite = position.side_to_move() == Color::White ? out.mean_cp : -out.mean_cp;
    const int totalWhite = baselineWhite + distilled_correction_white(position);
    out.mean_cp = position.side_to_move() == Color::White ? totalWhite : -totalWhite;
    out.provenance = 2;
    return out;
}

const EvaluatorDescriptor& DistilledEvaluator::descriptor() const {
    static constexpr EvaluatorDescriptor d{
        "leviathan-distilled-v1",
        "stockfish18-teacher",
        "linear-residual-sf18-d5-seed8910-v1",
        "project-native",
        EvaluatorOrigin::Hybrid
    };
    return d;
}

const Evaluator& baseline_evaluator() {
    static const BaselineEvaluator evaluator;
    return evaluator;
}

const Evaluator& distilled_evaluator() {
    static const DistilledEvaluator evaluator;
    return evaluator;
}

const Evaluator& default_evaluator() {
    return distilled_evaluator();
}

} // namespace leviathan

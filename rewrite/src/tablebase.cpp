#include "tablebase.h"
#include <cstdint>
#include <string>

#ifdef LEVIATHAN_HAS_FATHOM
#include <tbprobe.h>
#endif

namespace leviathan {

const TablebaseDescriptor& NullTablebase::descriptor() const {
    static constexpr TablebaseDescriptor d{"none", "native", "Leviathan project", "none"};
    return d;
}

const Tablebase& null_tablebase() {
    static const NullTablebase tb;
    return tb;
}

const TablebaseDescriptor& FathomTablebase::descriptor() const {
    static constexpr TablebaseDescriptor d{
        "fathom-c9c6fef0",
        "fathom",
        "MIT",
        "c9c6fef0dddc05d2e242c183acf5833149ab676d"
    };
    return d;
}

FathomTablebase::~FathomTablebase() { reset(); }

bool FathomTablebase::initialize(std::string_view path) {
    reset();
#ifdef LEVIATHAN_HAS_FATHOM
    std::string owned(path);
    if (!tb_init(owned.c_str())) return false;
    available_ = TB_LARGEST > 0;
    return available_;
#else
    (void) path;
    return false;
#endif
}

void FathomTablebase::reset() {
#ifdef LEVIATHAN_HAS_FATHOM
    if (available_) tb_free();
#endif
    available_ = false;
}

int FathomTablebase::max_pieces() const {
#ifdef LEVIATHAN_HAS_FATHOM
    return available_ ? static_cast<int>(TB_LARGEST) : 0;
#else
    return 0;
#endif
}

std::optional<Wdl> FathomTablebase::probe_wdl(const Position& position) const {
#ifdef LEVIATHAN_HAS_FATHOM
    if (!available_ || position.castling_rights() != 0 || position.halfmove_clock() != 0)
        return std::nullopt;

    uint64_t white = 0, black = 0;
    uint64_t kings = 0, queens = 0, rooks = 0, bishops = 0, knights = 0, pawns = 0;
    int pieces = 0;

    for (int sq = 0; sq < 64; ++sq) {
        const Piece piece = position.piece_at(sq);
        if (piece == Piece::Empty) continue;
        ++pieces;
        const uint64_t bit = uint64_t{1} << sq;
        (color_of(piece) == Color::White ? white : black) |= bit;
        switch (type_of(piece)) {
            case PieceType::King:   kings   |= bit; break;
            case PieceType::Queen:  queens  |= bit; break;
            case PieceType::Rook:   rooks   |= bit; break;
            case PieceType::Bishop: bishops |= bit; break;
            case PieceType::Knight: knights |= bit; break;
            case PieceType::Pawn:   pawns   |= bit; break;
            default: break;
        }
    }

    if (pieces > static_cast<int>(TB_LARGEST)) return std::nullopt;
    const unsigned ep = position.ep_square() >= 0 ? static_cast<unsigned>(position.ep_square()) : 0U;
    const unsigned result = tb_probe_wdl(
        white, black, kings, queens, rooks, bishops, knights, pawns,
        0U, 0U, ep, position.side_to_move() == Color::White
    );
    if (result == TB_RESULT_FAILED) return std::nullopt;
    switch (result) {
        case TB_LOSS:         return Wdl::Loss;
        case TB_BLESSED_LOSS: return Wdl::BlessedLoss;
        case TB_DRAW:         return Wdl::Draw;
        case TB_CURSED_WIN:   return Wdl::CursedWin;
        case TB_WIN:          return Wdl::Win;
        default:              return std::nullopt;
    }
#else
    (void) position;
    return std::nullopt;
#endif
}

} // namespace leviathan

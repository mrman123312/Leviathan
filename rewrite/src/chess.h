#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace leviathan {

enum class Color : uint8_t { White = 0, Black = 1 };
inline Color opposite(Color c) { return c == Color::White ? Color::Black : Color::White; }

enum class PieceType : uint8_t { None=0, Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6 };
enum class Piece : uint8_t {
    Empty=0,
    WP=1, WN=2, WB=3, WR=4, WQ=5, WK=6,
    BP=7, BN=8, BB=9, BR=10, BQ=11, BK=12
};

inline bool is_empty(Piece p) { return p == Piece::Empty; }
Color color_of(Piece p);
PieceType type_of(Piece p);
Piece make_piece(Color c, PieceType pt);

struct Move {
    uint8_t from = 0;
    uint8_t to = 0;
    uint8_t promotion = 0;
    uint8_t flags = 0;

    enum Flag : uint8_t {
        Capture = 1 << 0,
        EnPassant = 1 << 1,
        Castle = 1 << 2,
        DoublePush = 1 << 3
    };

    constexpr bool is_null() const { return from == 0 && to == 0 && promotion == 0 && flags == 0; }
    constexpr bool operator==(const Move&) const = default;
};

std::string square_name(int sq);
int parse_square(std::string_view s);
std::string move_to_uci(Move m);

class Position {
public:
    Position();

    static Position startpos();
    static std::optional<Position> from_fen(std::string_view fen);

    std::string fen() const;
    Color side_to_move() const { return side_; }
    Piece piece_at(int sq) const { return board_[sq]; }
    int ep_square() const { return ep_square_; }
    int halfmove_clock() const { return halfmove_; }
    int fullmove_number() const { return fullmove_; }
    uint8_t castling_rights() const { return castling_; }

    std::vector<Move> pseudo_legal_moves(bool captures_only=false) const;
    std::vector<Move> legal_moves(bool captures_only=false) const;
    bool make_move(Move m);
    bool in_check(Color c) const;
    bool attacked(int sq, Color by) const;
    uint64_t key() const;

    std::optional<Move> parse_uci_move(std::string_view text) const;

private:
    std::array<Piece,64> board_{};
    Color side_ = Color::White;
    uint8_t castling_ = 0;
    int ep_square_ = -1;
    int halfmove_ = 0;
    int fullmove_ = 1;

    int king_square(Color c) const;
};

int piece_value(PieceType pt);

struct Evaluation {
    int mean_cp = 0;
    uint16_t uncertainty = 0;
    uint16_t volatility = 0;
    uint16_t provenance = 1;
};

Evaluation evaluate_position(const Position& p);

} // namespace leviathan

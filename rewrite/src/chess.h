#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

// Chess has at most 218 legal moves in a position; 256 also safely covers the
// pseudo-legal control representation. Keeping this object inline removes a
// heap allocation from every generated search node.
class MoveList {
public:
    static constexpr std::size_t kCapacity = 256;

    void push_back(Move m) {
        // Never silently truncate the search tree. Capacity exhaustion is a
        // correctness failure and must crash loudly in every build mode.
        if(size_ >= kCapacity) std::abort();
        moves_[size_++] = m;
    }
    bool empty() const { return size_ == 0; }
    std::size_t size() const { return size_; }
    Move& operator[](std::size_t i) { return moves_[i]; }
    const Move& operator[](std::size_t i) const { return moves_[i]; }
    Move* begin() { return moves_.data(); }
    Move* end() { return moves_.data() + size_; }
    const Move* begin() const { return moves_.data(); }
    const Move* end() const { return moves_.data() + size_; }

private:
    std::array<Move,kCapacity> moves_{};
    std::size_t size_ = 0;
};

struct UndoState {
    Piece moved = Piece::Empty;
    Piece captured_on_to = Piece::Empty;
    Piece ep_captured = Piece::Empty;
    int ep_capture_square = -1;
    uint8_t castling = 0;
    int ep_square = -1;
    int halfmove = 0;
    int fullmove = 1;
    Color side = Color::White;
    int white_king = -1;
    int black_king = -1;
    uint64_t key = 0;
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

    // tactical_only means captures plus all promotions, including quiet ones.
    MoveList pseudo_legal_moves(bool tactical_only=false) const;
    std::vector<Move> legal_moves(bool tactical_only=false) const;
    bool make_move(Move m);
    bool make_move(Move m, UndoState& undo);
    void unmake_move(Move m, const UndoState& undo);
    bool in_check(Color c) const;
    bool attacked(int sq, Color by) const;
    uint64_t key() const { return key_; }

    std::optional<Move> parse_uci_move(std::string_view text) const;

private:
    std::array<Piece,64> board_{};
    std::array<int8_t,2> king_sq_{{-1,-1}};
    Color side_ = Color::White;
    uint8_t castling_ = 0;
    int ep_square_ = -1;
    int halfmove_ = 0;
    int fullmove_ = 1;
    uint64_t key_ = 0;

    int king_square(Color c) const { return king_sq_[static_cast<int>(c)]; }
    int canonical_ep_square() const;
    void recompute_key();
    void set_piece(int sq, Piece p);
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

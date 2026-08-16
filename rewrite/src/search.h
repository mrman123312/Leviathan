#pragma once
#include "chess.h"
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace leviathan {

enum class Bound : uint8_t { Exact, Lower, Upper };

struct TTEntry {
    uint64_t key = 0;
    int depth = -1;
    int score = 0;
    Bound bound = Bound::Exact;
    Move best{};
    uint16_t evidence = 0;
    uint8_t debt = 0;
};

struct SearchLimits {
    int max_depth = 5;
    int movetime_ms = 0;
};

struct SearchReport {
    Move best{};
    int score = 0;
    int completed_depth = 0;
    uint64_t nodes = 0;
    std::vector<Move> pv;
};

class SearchEngine {
public:
    SearchReport search(const Position& root, const SearchLimits& limits,
                        const std::vector<uint64_t>& game_history = {});
    void clear();

private:
    static constexpr int INF = 32000;
    static constexpr int MATE = 30000;
    std::unordered_map<uint64_t, TTEntry> tt_;
    uint64_t nodes_ = 0;
    bool stopped_ = false;
    std::chrono::steady_clock::time_point deadline_{};
    bool use_deadline_ = false;
    std::vector<uint64_t> history_;

    int negamax(const Position& p, int depth, int alpha, int beta, int ply);
    int quiescence(const Position& p, int alpha, int beta, int ply);
    bool repeated(uint64_t key) const;
    uint64_t context_key(const Position& p) const;
    bool time_up();
    std::vector<Move> ordered_moves(const Position& p, Move tt_move, bool captures_only=false) const;
    std::vector<Move> extract_pv(Position p, int max_len, std::vector<uint64_t> history) const;
};

} // namespace leviathan

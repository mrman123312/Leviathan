#pragma once
#include "chess.h"
#include "evaluator.h"
#include <array>
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
    // max_depth == 0 means no artificial depth ceiling; search is then bounded
    // by movetime (or the engine's hard MAX_PLY safety limit).
    int max_depth = 0;
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
    explicit SearchEngine(const Evaluator& evaluator = default_evaluator()) : evaluator_(&evaluator) {}

    SearchReport search(const Position& root, const SearchLimits& limits,
                        const std::vector<uint64_t>& game_history = {});
    void clear();
    const EvaluatorDescriptor& evaluator_descriptor() const { return evaluator_->descriptor(); }

private:
    static constexpr int INF = 32000;
    static constexpr int MATE = 30000;
    static constexpr int MAX_PLY = 128;
    const Evaluator* evaluator_;
    std::unordered_map<uint64_t, TTEntry> tt_;
    std::array<std::array<int,64>,64> quiet_history_{};
    std::array<std::array<Move,2>,MAX_PLY> killers_{};
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
    static int score_to_tt(int score, int ply);
    static int score_from_tt(int score, int ply);
    void reward_quiet(Move m, int depth, int ply);
    std::vector<Move> ordered_moves(const Position& p, Move tt_move, int ply,
                                    bool captures_only=false) const;
    std::vector<Move> extract_pv(Position p, int max_len, std::vector<uint64_t> history) const;
};

} // namespace leviathan

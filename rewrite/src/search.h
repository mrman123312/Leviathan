#pragma once
#include "chess.h"
#include "evaluator.h"
#include "transposition.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

namespace leviathan {

struct SearchLimits {
    int max_depth = 0;
    int movetime_ms = 0;
};

struct SearchReport {
    Move best{};
    int score = 0;
    int completed_depth = 0;
    uint64_t nodes = 0;
    uint64_t tt_hits = 0;
    uint64_t tt_stores = 0;
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
    TranspositionTable tt_;
    std::array<std::array<int,64>,64> quiet_history_{};
    std::array<std::array<Move,2>,MAX_PLY> killers_{};
    uint64_t nodes_ = 0;
    uint64_t tt_hits_ = 0;
    uint64_t tt_stores_ = 0;
    bool stopped_ = false;
    std::chrono::steady_clock::time_point deadline_{};
    bool use_deadline_ = false;
    std::vector<uint64_t> history_;

    int negamax(Position& p, int depth, int alpha, int beta, int ply);
    int quiescence(Position& p, int alpha, int beta, int ply);
    bool repeated(uint64_t key) const;
    bool history_sensitive(const Position& p) const;
    uint64_t context_key(const Position& p) const;
    bool time_up();
    static int score_to_tt(int score, int ply);
    static int score_from_tt(int score, int ply);
    void reward_quiet(Move m, int depth, int ply);
    MoveList ordered_moves(const Position& p, Move tt_move, int ply,
                           bool captures_only=false) const;
    std::vector<Move> extract_pv(Position p, int max_len, std::vector<uint64_t> history) const;
};

} // namespace leviathan

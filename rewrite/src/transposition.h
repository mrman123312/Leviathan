#pragma once
#include "chess.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace leviathan {

enum class Bound : uint8_t { Exact, Lower, Upper };

struct TTEntry {
    uint64_t key = 0;
    int score = 0;
    int16_t depth = -1;
    Bound bound = Bound::Exact;
    Move best{};
    uint16_t evidence = 0;
    uint8_t debt = 0;
};

class TranspositionTable {
public:
    static constexpr std::size_t kDefaultEntries = 1u << 19;

    explicit TranspositionTable(std::size_t entries = kDefaultEntries) {
        std::size_t n = 1;
        while(n < std::max<std::size_t>(1, entries)) n <<= 1;
        table_.resize(n);
        mask_ = n - 1;
    }

    void clear() { std::fill(table_.begin(), table_.end(), TTEntry{}); }

    const TTEntry* probe(uint64_t key) const {
        const TTEntry& e = table_[index(key)];
        return e.depth >= 0 && e.key == key ? &e : nullptr;
    }

    TTEntry* probe(uint64_t key) {
        TTEntry& e = table_[index(key)];
        return e.depth >= 0 && e.key == key ? &e : nullptr;
    }

    void store(const TTEntry& incoming) {
        TTEntry& slot = table_[index(incoming.key)];
        const bool empty = slot.depth < 0;
        if(empty){ slot=incoming; return; }

        const bool same = slot.key == incoming.key;
        if(same){
            const bool deeper = incoming.depth > slot.depth;
            const bool sameDepth = incoming.depth == slot.depth;
            const bool exactUpgrade = incoming.bound == Bound::Exact && slot.bound != Bound::Exact;
            const bool moveRecovery = slot.best.is_null() && !incoming.best.is_null();
            if(deeper || sameDepth || exactUpgrade || moveRecovery) slot=incoming;
            return;
        }

        // On collisions, protect substantially deeper information. A newcomer
        // may replace an unrelated entry only if it is close in depth, or if it
        // carries an exact result at no more than a one-ply disadvantage.
        const bool collisionRefresh = incoming.depth + 2 >= slot.depth;
        const bool exactCollision = incoming.bound == Bound::Exact && incoming.depth + 1 >= slot.depth;
        if(collisionRefresh || exactCollision) slot=incoming;
    }

    std::size_t capacity() const { return table_.size(); }

private:
    std::vector<TTEntry> table_;
    std::size_t mask_ = 0;

    std::size_t index(uint64_t key) const {
        return static_cast<std::size_t>(key) & mask_;
    }
};

} // namespace leviathan

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

// Fixed-capacity, cache-local TT. This deliberately avoids node-based heap
// allocation and pointer chasing from std::unordered_map. Direct mapping is a
// control design; cluster associativity can be tested later rather than assumed.
class TranspositionTable {
public:
    static constexpr std::size_t kDefaultEntries = 1u << 19; // 524,288 entries

    explicit TranspositionTable(std::size_t entries = kDefaultEntries) {
        std::size_t n = 1;
        while(n < std::max<std::size_t>(1, entries)) n <<= 1;
        table_.resize(n);
        mask_ = n - 1;
    }

    void clear() {
        std::fill(table_.begin(), table_.end(), TTEntry{});
    }

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
        const bool same = !empty && slot.key == incoming.key;
        const bool deeper = incoming.depth >= slot.depth;
        const bool exactUpgrade = incoming.bound == Bound::Exact && slot.bound != Bound::Exact;
        // Permit a slightly shallower replacement on collision so stale deep
        // entries cannot monopolize a direct-mapped slot forever.
        const bool collisionRefresh = !same && incoming.depth + 2 >= slot.depth;
        if(empty || same || deeper || exactUpgrade || collisionRefresh) slot = incoming;
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

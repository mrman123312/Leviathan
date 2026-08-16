#ifndef LEVIATHAN_EVIDENCE_H_INCLUDED
#define LEVIATHAN_EVIDENCE_H_INCLUDED

#include <algorithm>
#include <cstdint>

namespace Stockfish::Leviathan::Evidence {

using Mask = std::uint16_t;

enum class Kind : Mask {
    NONE               = 0,
    HISTORY            = Mask(1) << 0,
    RULE50             = Mask(1) << 1,
    EVAL_DISAGREEMENT  = Mask(1) << 2,
    CORRECTION_STRESS  = Mask(1) << 3,
    NULL_FRAGILITY     = Mask(1) << 4,
    PROBCUT_NEAR_PROOF = Mask(1) << 5,
    LMR_COUNTEREXAMPLE = Mask(1) << 6,
    CHILD_UNCERTAINTY  = Mask(1) << 7,
    RIVAL_AMBIGUITY    = Mask(1) << 8,
    PERSISTENT_WITNESS = Mask(1) << 9,
};

constexpr Mask bit(Kind k) { return static_cast<Mask>(k); }
constexpr Mask operator|(Kind a, Kind b) { return bit(a) | bit(b); }
constexpr Mask operator|(Mask a, Kind b) { return a | bit(b); }
constexpr bool has(Mask mask, Kind k) { return (mask & bit(k)) != 0; }

inline int class_count(Mask mask) {
    int count = 0;
    while (mask)
    {
        count += int(mask & 1U);
        mask >>= 1;
    }
    return count;
}

struct State {
    Mask mask = 0;
    int  debt = 0;

    void add(Kind kind, int weight = 1) {
        if (kind == Kind::NONE || weight <= 0)
            return;
        mask |= bit(kind);
        debt = std::min(5, debt + weight);
    }

    void merge(Mask otherMask, int inheritedDebt) {
        mask |= otherMask;
        debt = std::max(debt, std::clamp(inheritedDebt, 0, 5));
    }

    bool contains(Kind kind) const { return has(mask, kind); }
    int  classes() const { return class_count(mask); }
    bool multi_source() const { return debt >= 3 && classes() >= 2; }
    bool severe() const { return debt >= 4 && classes() >= 3; }
};

// Specialist authority routing. These helpers are deliberately conservative:
// typed evidence may remove shortcut authority, but never grants a shortcut
// that the scalar-debt safeguards would already have rejected.
inline bool tt_sensitive(const State& s) {
    const Mask ttMask = Kind::HISTORY | Kind::RULE50 | Kind::EVAL_DISAGREEMENT;
    return (s.mask & ttMask) && s.debt >= 2;
}

inline bool pruning_sensitive(const State& s) {
    const Mask pruningMask = Kind::NULL_FRAGILITY | Kind::LMR_COUNTEREXAMPLE
                           | Kind::CHILD_UNCERTAINTY | Kind::PERSISTENT_WITNESS;
    return (s.mask & pruningMask) && s.debt >= 2;
}

inline bool rival_sensitive(const State& s) {
    const Mask rivalMask = Kind::RIVAL_AMBIGUITY | Kind::PROBCUT_NEAR_PROOF
                         | Kind::PERSISTENT_WITNESS;
    return (s.mask & rivalMask) && s.debt >= 2;
}

inline double root_time_factor(const State& s) {
    if (s.severe())
        return 1.20;
    if (s.multi_source())
        return 1.12;
    if (s.debt >= 3)
        return 1.06;
    return 1.0;
}

constexpr Mask KNOWN_MASK = bit(Kind::HISTORY) | bit(Kind::RULE50) | bit(Kind::EVAL_DISAGREEMENT)
                          | bit(Kind::CORRECTION_STRESS) | bit(Kind::NULL_FRAGILITY)
                          | bit(Kind::PROBCUT_NEAR_PROOF) | bit(Kind::LMR_COUNTEREXAMPLE)
                          | bit(Kind::CHILD_UNCERTAINTY) | bit(Kind::RIVAL_AMBIGUITY)
                          | bit(Kind::PERSISTENT_WITNESS);

static_assert((KNOWN_MASK & (KNOWN_MASK + 1)) == 0,
              "Leviathan evidence kinds must remain a dense low-bit set");

}  // namespace Stockfish::Leviathan::Evidence

#endif  // LEVIATHAN_EVIDENCE_H_INCLUDED

/*
  Project Leviathan Chess Atlas runtime retrieval.

  Atlas entries are hints with provenance, never unchecked replacements for
  legal move generation or terminal evaluation. Exact/proven entries are kept
  distinct in the file format but are still used as ordering evidence here;
  Stockfish verifies the line normally.
*/

#ifndef LEVIATHAN_ATLAS_H_INCLUDED
#define LEVIATHAN_ATLAS_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>

#include "leviathan_attributes.h"
#include "position.h"
#include "types.h"

namespace Stockfish::Leviathan::Atlas {

struct Entry {
    u16 rawMove    = 0;
    int bonus      = 0;
    int confidence = 0;
    bool exact     = false;
};

struct State {
    bool enabled = false;
    int  weight  = 100;
    std::string file;
    std::unordered_multimap<u64, Entry> entries;
    bool loaded = false;

    State() {
        const char* e = std::getenv("LEVIATHAN_ATLAS");
        enabled = e && *e && std::string(e) != "0" && std::string(e) != "false";
        if (const char* w = std::getenv("LEVIATHAN_ATLAS_WEIGHT"))
            weight = std::clamp(std::atoi(w), 0, 400);
        if (const char* p = std::getenv("LEVIATHAN_ATLAS_FILE"))
            file = p;
    }
};

inline State& state() {
    static State s;
    return s;
}

inline bool load(const std::string& path) {
    auto& s = state();
    s.entries.clear();
    s.loaded = false;
    s.file   = path;
    if (path.empty())
        return false;

    std::ifstream in(path);
    if (!in)
        return false;

    std::string magic;
    in >> magic;
    if (magic != "LVTA1")
        return false;

    u64         key = 0;
    unsigned    raw = 0;
    int         bonus = 0, confidence = 0;
    std::string kind;
    while (in >> key >> raw >> bonus >> confidence >> kind)
    {
        if (raw > 65535 || bonus < -32768 || bonus > 32768 || confidence < 0
            || confidence > 1000)
        {
            s.entries.clear();
            return false;
        }
        s.entries.emplace(key, Entry{u16(raw), bonus, confidence, kind == "exact"});
    }

    if (!in.eof())
    {
        s.entries.clear();
        return false;
    }

    s.loaded = true;
    return true;
}

inline void set_enabled(bool v) {
    state().enabled = v;
    if (v && !state().loaded && !state().file.empty())
        load(state().file);
}
inline void set_weight(int v) { state().weight = std::clamp(v, 0, 400); }
inline bool set_file(const std::string& path) { return load(path); }
inline bool ready() { return state().enabled && state().loaded; }

LEVIATHAN_NOINLINE inline int ordering_bonus(const Position& pos, Move move) {
    const auto& s = state();
    if (!s.enabled || !s.loaded)
        return 0;

    const u64 key = u64(pos.key());
    auto [it, end] = s.entries.equal_range(key);
    int best = 0;
    for (; it != end; ++it)
        if (it->second.rawMove == move.raw())
        {
            const Entry& e = it->second;
            int b = e.bonus * e.confidence / 1000;
            if (e.exact)
                b += 512;  // proof provenance gets priority, not search bypass authority
            best = std::max(best, b);
        }

    return std::clamp(best * s.weight / 100, -4096, 8192);
}

}  // namespace Stockfish::Leviathan::Atlas

#endif  // LEVIATHAN_ATLAS_H_INCLUDED

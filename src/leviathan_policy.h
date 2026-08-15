/*
  Project Leviathan policy guidance for Stockfish.

  This file is part of a GPLv3-or-later Stockfish derivative.
  The policy is disabled by default. Environment variables remain supported for
  experiment wrappers, while explicit setters allow the same binary to be A/B
  configured through UCI options.
*/

#ifndef LEVIATHAN_POLICY_H_INCLUDED
#define LEVIATHAN_POLICY_H_INCLUDED

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

#include "position.h"
#include "types.h"

namespace Stockfish::Leviathan::Policy {

constexpr int FeatureCount = 12;
constexpr int HiddenSize   = 16;
using FeatureVector        = std::array<std::int16_t, FeatureCount>;

struct QuantizedModel {
    std::array<std::array<std::int8_t, FeatureCount>, HiddenSize> hiddenWeights{};
    std::array<std::int16_t, HiddenSize>                           hiddenBias{};
    std::array<std::int8_t, HiddenSize>                            outputWeights{};
    std::int32_t                                                   outputBias = 0;
    bool                                                           loaded     = false;
};

inline int parse_int_env(const char* name, int fallback, int low, int high) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw)
        return fallback;
    char* end = nullptr;
    long  v   = std::strtol(raw, &end, 10);
    if (!end || *end != '\0')
        return fallback;
    return std::clamp<int>(int(v), low, high);
}

inline bool parse_bool_env(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw)
        return fallback;
    const std::string s(raw);
    return s != "0" && s != "false" && s != "off";
}

inline bool read_i64(std::istream& in, long long& value) {
    in >> value;
    return bool(in);
}

inline QuantizedModel load_model_file(const std::string& path) {
    QuantizedModel model;
    if (path.empty())
        return model;

    std::ifstream in(path);
    if (!in)
        return model;

    std::string magic;
    int         features = 0, hidden = 0;
    in >> magic >> features >> hidden;
    if (!in || magic != "LVTP1" || features != FeatureCount || hidden != HiddenSize)
        return model;

    long long v = 0;
    for (int h = 0; h < HiddenSize; ++h)
        for (int f = 0; f < FeatureCount; ++f)
        {
            if (!read_i64(in, v) || v < -127 || v > 127)
                return QuantizedModel{};
            model.hiddenWeights[h][f] = std::int8_t(v);
        }

    for (int h = 0; h < HiddenSize; ++h)
    {
        if (!read_i64(in, v) || v < std::numeric_limits<std::int16_t>::min()
            || v > std::numeric_limits<std::int16_t>::max())
            return QuantizedModel{};
        model.hiddenBias[h] = std::int16_t(v);
    }

    for (int h = 0; h < HiddenSize; ++h)
    {
        if (!read_i64(in, v) || v < -127 || v > 127)
            return QuantizedModel{};
        model.outputWeights[h] = std::int8_t(v);
    }

    if (!read_i64(in, v) || v < std::numeric_limits<std::int32_t>::min()
        || v > std::numeric_limits<std::int32_t>::max())
        return QuantizedModel{};

    model.outputBias = std::int32_t(v);
    model.loaded     = true;
    return model;
}

struct State {
    bool           enabled = false;
    int            weight  = 100;
    std::string    path;
    QuantizedModel model;

    State() {
        enabled = parse_bool_env("LEVIATHAN_POLICY", false);
        weight  = parse_int_env("LEVIATHAN_POLICY_WEIGHT", 100, 0, 400);
        if (const char* p = std::getenv("LEVIATHAN_POLICY_FILE"))
            path = p;
        if (!path.empty())
            model = load_model_file(path);
    }
};

inline State& state() {
    static State s;
    return s;
}

inline void set_enabled(bool v) { state().enabled = v; }
inline void set_weight(int v) { state().weight = std::clamp(v, 0, 400); }
inline bool set_model_path(const std::string& path) {
    state().path  = path;
    state().model = load_model_file(path);
    return state().model.loaded;
}

inline const std::string& model_path() { return state().path; }
inline int policy_weight() { return state().weight; }
inline bool policy_enabled() { return state().enabled; }
inline const QuantizedModel& model() { return state().model; }

inline int centered_file(Square s) { return 2 * (int(s) & 7) - 7; }
inline int rank_for(Color us, Square s) {
    int rank = int(s) >> 3;
    return us == WHITE ? rank : 7 - rank;
}

inline FeatureVector features(const Position& pos, Move move) {
    const Color  us   = pos.side_to_move();
    const Square from = move.from_sq();
    const Square to   = move.to_sq();
    const Piece  pc   = pos.moved_piece(move);
    const int    ff   = centered_file(from);
    const int    tf   = centered_file(to);
    const int    fr   = rank_for(us, from);
    const int    tr   = rank_for(us, to);

    const int fromCenter = 14 - std::abs(ff) - std::abs(2 * fr - 7);
    const int toCenter   = 14 - std::abs(tf) - std::abs(2 * tr - 7);
    const int check      = bool(pos.check_squares(type_of(pc)) & to);
    const Bitboard enemyPawnAttacks = pos.attacks_by<PAWN>(~us);

    return FeatureVector{
      std::int16_t(ff),
      std::int16_t(2 * fr - 7),
      std::int16_t(tf),
      std::int16_t(2 * tr - 7),
      std::int16_t(tf - ff),
      std::int16_t(tr - fr),
      std::int16_t(type_of(pc)),
      std::int16_t(toCenter - fromCenter),
      std::int16_t(check * 8),
      std::int16_t(bool(enemyPawnAttacks & from) * 8),
      std::int16_t(bool(enemyPawnAttacks & to) * 8),
      std::int16_t(2 * (tr - fr))};
}

inline int raw_score(const Position& pos, Move move) {
    const auto& m = model();
    if (!m.loaded)
        return 0;

    const FeatureVector x = features(pos, move);
    std::int32_t out = m.outputBias;
    for (int h = 0; h < HiddenSize; ++h)
    {
        std::int32_t a = m.hiddenBias[h];
        for (int f = 0; f < FeatureCount; ++f)
            a += std::int32_t(m.hiddenWeights[h][f]) * x[f];
        a = std::clamp<std::int32_t>(a, 0, 127);
        out += std::int32_t(m.outputWeights[h]) * a;
    }
    return std::clamp<int>(out / 16, -2048, 2048);
}

inline int ordering_bonus(const Position& pos, Move move) {
    if (!policy_enabled() || !model().loaded)
        return 0;
    return std::clamp(raw_score(pos, move) * policy_weight() / 100, -8192, 8192);
}

inline bool ready() { return policy_enabled() && model().loaded; }

}  // namespace Stockfish::Leviathan::Policy

#endif  // LEVIATHAN_POLICY_H_INCLUDED

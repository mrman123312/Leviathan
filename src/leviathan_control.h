/*
  Project Leviathan metacognitive search controls.

  This file is part of a GPLv3-or-later Stockfish derivative.

  Design rule: every learned controller is fail-closed. If it is disabled,
  missing, malformed, or outside its authority level it contributes exactly
  zero search change. The trusted Stockfish parent therefore remains a hard
  rollback path in the same binary.
*/

#ifndef LEVIATHAN_CONTROL_H_INCLUDED
#define LEVIATHAN_CONTROL_H_INCLUDED

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

#include "leviathan_attributes.h"
#include "types.h"

namespace Stockfish::Leviathan::Control {

inline bool env_bool(const char* name, bool fallback = false) {
    const char* p = std::getenv(name);
    if (!p || !*p)
        return fallback;
    const std::string s(p);
    return s != "0" && s != "false" && s != "off";
}

inline int env_int(const char* name, int fallback, int lo, int hi) {
    const char* p = std::getenv(name);
    if (!p || !*p)
        return fallback;
    char* end = nullptr;
    long  v   = std::strtol(p, &end, 10);
    if (!end || *end != '\0')
        return fallback;
    return std::clamp(int(v), lo, hi);
}

inline std::string env_string(const char* name) {
    const char* p = std::getenv(name);
    return p ? std::string(p) : std::string();
}

template<int FeatureCount, int MaxHeads = 4>
struct LinearEnsemble {
    struct Result {
        int mean   = 0;
        int spread = 0;
    };

    std::array<std::array<std::int16_t, FeatureCount>, MaxHeads> weights{};
    std::array<std::int32_t, MaxHeads>                            bias{};
    int                                                           heads  = 0;
    int                                                           scale  = 1;
    bool                                                          loaded = false;

    void clear() { *this = LinearEnsemble{}; }

    bool load(const std::string& path, const char* expectedMagic) {
        clear();
        if (path.empty())
            return false;

        std::ifstream in(path);
        if (!in)
            return false;

        std::string magic;
        int         features = 0;
        in >> magic >> features >> heads >> scale;
        if (!in || magic != expectedMagic || features != FeatureCount || heads < 1
            || heads > MaxHeads || scale < 1 || scale > 1'000'000)
        {
            clear();
            return false;
        }

        long long v = 0;
        for (int h = 0; h < heads; ++h)
        {
            for (int f = 0; f < FeatureCount; ++f)
            {
                in >> v;
                if (!in || v < std::numeric_limits<std::int16_t>::min()
                    || v > std::numeric_limits<std::int16_t>::max())
                {
                    clear();
                    return false;
                }
                weights[h][f] = std::int16_t(v);
            }

            in >> v;
            if (!in || v < std::numeric_limits<std::int32_t>::min()
                || v > std::numeric_limits<std::int32_t>::max())
            {
                clear();
                return false;
            }
            bias[h] = std::int32_t(v);
        }

        loaded = true;
        return true;
    }

    Result eval(const std::array<int, FeatureCount>& x) const {
        if (!loaded || heads <= 0)
            return {};

        int total = 0;
        int lo    = 1000000;
        int hi    = -1000000;
        for (int h = 0; h < heads; ++h)
        {
            std::int64_t acc = bias[h];
            for (int f = 0; f < FeatureCount; ++f)
                acc += std::int64_t(weights[h][f]) * std::clamp(x[f], -4096, 4096);

            const int y = std::clamp(int(acc / scale), -1000, 1000);
            total += y;
            lo = std::min(lo, y);
            hi = std::max(hi, y);
        }
        return {total / heads, hi - lo};
    }
};

constexpr int MetaFeatures = 8;
constexpr int RiskFeatures = 12;

struct State {
    bool metaEnabled       = false;
    int  metaAuthority     = 0;  // 0 off, 1 deepen-only, 2 bidirectional time control
    int  metaMaxPercent    = 30;
    std::string metaFile;
    LinearEnsemble<MetaFeatures> meta;

    bool riskEnabled       = false;
    int  riskAuthority     = 0;  // 0 off, 1 veto reductions, 2 also permit bounded extra reduction
    int  riskThreshold     = 650;
    int  riskVeto          = 1536;  // reduction units, 1024 ~= one ply
    std::string riskFile;
    LinearEnsemble<RiskFeatures> risk;

    bool specialistEnabled = false;
    int  specialistVeto    = 768;

    State() {
        metaEnabled      = env_bool("LEVIATHAN_META", false);
        metaAuthority    = env_int("LEVIATHAN_META_AUTHORITY", 0, 0, 2);
        metaMaxPercent   = env_int("LEVIATHAN_META_MAX_PERCENT", 30, 0, 100);
        metaFile         = env_string("LEVIATHAN_META_FILE");
        if (!metaFile.empty())
            meta.load(metaFile, "LVTM1");

        riskEnabled      = env_bool("LEVIATHAN_RISK", false);
        riskAuthority    = env_int("LEVIATHAN_RISK_AUTHORITY", 0, 0, 2);
        riskThreshold    = env_int("LEVIATHAN_RISK_THRESHOLD", 650, 0, 1000);
        riskVeto         = env_int("LEVIATHAN_RISK_VETO", 1536, 0, 4096);
        riskFile         = env_string("LEVIATHAN_RISK_FILE");
        if (!riskFile.empty())
            risk.load(riskFile, "LVTR1");

        specialistEnabled = env_bool("LEVIATHAN_SPECIALIST", false);
        specialistVeto    = env_int("LEVIATHAN_SPECIALIST_VETO", 768, 0, 4096);
    }
};

inline State& state() {
    static State s;
    return s;
}

inline void set_meta_enabled(bool v) { state().metaEnabled = v; }
inline void set_meta_authority(int v) { state().metaAuthority = std::clamp(v, 0, 2); }
inline void set_meta_max_percent(int v) { state().metaMaxPercent = std::clamp(v, 0, 100); }
inline bool set_meta_file(const std::string& path) {
    state().metaFile = path;
    return state().meta.load(path, "LVTM1");
}
inline bool meta_ready() {
    return state().metaEnabled && state().metaAuthority > 0 && state().meta.loaded;
}

inline void set_risk_enabled(bool v) { state().riskEnabled = v; }
inline void set_risk_authority(int v) { state().riskAuthority = std::clamp(v, 0, 2); }
inline void set_risk_threshold(int v) { state().riskThreshold = std::clamp(v, 0, 1000); }
inline void set_risk_veto(int v) { state().riskVeto = std::clamp(v, 0, 4096); }
inline bool set_risk_file(const std::string& path) {
    state().riskFile = path;
    return state().risk.load(path, "LVTR1");
}
inline bool risk_ready() {
    return state().riskEnabled && state().riskAuthority > 0 && state().risk.loaded;
}

inline void set_specialist_enabled(bool v) { state().specialistEnabled = v; }
inline void set_specialist_veto(int v) { state().specialistVeto = std::clamp(v, 0, 4096); }

// The meta model estimates search regret / probability that more computation
// changes the useful root decision. Ensemble disagreement is deliberately added
// as epistemic uncertainty rather than averaged away.
inline int meta_risk(Depth depth,
                     Depth stableDepth,
                     Value previousScore,
                     Value currentScore,
                     double bestMoveChanges,
                     int nodesEffort,
                     usize rootMoveCount,
                     bool decisive) {
    if (!meta_ready())
        return 500;

    const int delta = previousScore == -VALUE_INFINITE ? 0 : std::abs(currentScore - previousScore);
    const std::array<int, MetaFeatures> x = {
      std::clamp(int(depth), 0, 128),
      std::clamp(int(stableDepth), 0, 64),
      std::clamp(delta / 8, 0, 256),
      std::clamp(int(bestMoveChanges * 16.0), 0, 256),
      std::clamp(nodesEffort / 1000, 0, 128),
      std::clamp(int(rootMoveCount), 0, 64),
      std::clamp(std::abs(currentScore) / 32, 0, 256),
      decisive ? 32 : 0};

    const auto r = state().meta.eval(x);
    return std::clamp(500 + r.mean + r.spread / 2, 0, 1000);
}

inline double meta_time_factor(Depth depth,
                               Depth stableDepth,
                               Value previousScore,
                               Value currentScore,
                               double bestMoveChanges,
                               int nodesEffort,
                               usize rootMoveCount,
                               bool decisive) {
    if (!meta_ready() || decisive)
        return 1.0;

    const int risk = meta_risk(depth, stableDepth, previousScore, currentScore, bestMoveChanges,
                               nodesEffort, rootMoveCount, decisive);
    const double maxDelta = state().metaMaxPercent / 100.0;

    if (state().metaAuthority == 1)
    {
        const double extra = std::max(0, risk - 500) / 500.0 * maxDelta;
        return 1.0 + extra;
    }

    // Bidirectional authority is intentionally symmetric and bounded. It is not
    // enabled by default; it must earn promotion after deepen-only calibration.
    const double signedRisk = (risk - 500) / 500.0;
    return std::clamp(1.0 + signedRisk * maxDelta, 1.0 - maxDelta, 1.0 + maxDelta);
}

inline int risk_score(Depth depth,
                      int moveCount,
                      int statScore,
                      int correctionValue,
                      bool pvNode,
                      bool cutNode,
                      bool allNode,
                      bool capture,
                      bool givesCheck,
                      Depth ttDepth,
                      Value staticEval,
                      Value alpha) {
    if (!risk_ready())
        return 500;

    const std::array<int, RiskFeatures> x = {
      std::clamp(int(depth), 0, 128),
      std::clamp(moveCount, 0, 128),
      std::clamp(statScore / 128, -256, 256),
      std::clamp(correctionValue / 8192, -256, 256),
      pvNode ? 32 : 0,
      cutNode ? 32 : 0,
      allNode ? 32 : 0,
      capture ? 32 : 0,
      givesCheck ? 32 : 0,
      std::clamp(int(ttDepth - depth), -64, 64),
      std::clamp((staticEval - alpha) / 16, -256, 256),
      std::clamp(std::abs(staticEval) / 32, 0, 256)};

    const auto r = state().risk.eval(x);
    return std::clamp(500 + r.mean + r.spread / 2, 0, 1000);
}

// Return an adjustment in Stockfish's reduction units. Negative means search
// deeper. Authority level 1 can only veto/reduce aggressiveness; authority 2
// may also add a small reduction in exceptionally low-risk cases.
LEVIATHAN_NOINLINE inline int lmr_adjustment(Depth depth,
                          int moveCount,
                          int statScore,
                          int correctionValue,
                          bool pvNode,
                          bool cutNode,
                          bool allNode,
                          bool capture,
                          bool givesCheck,
                          Depth ttDepth,
                          Value staticEval,
                          Value alpha) {
    if (!risk_ready())
        return 0;

    const int risk = risk_score(depth, moveCount, statScore, correctionValue, pvNode, cutNode, allNode,
                                capture, givesCheck, ttDepth, staticEval, alpha);
    int adjustment = 0;

    if (risk > state().riskThreshold)
    {
        const int denom = std::max(1, 1000 - state().riskThreshold);
        adjustment = -state().riskVeto * (risk - state().riskThreshold) / denom;
    }
    else if (state().riskAuthority >= 2 && risk < 250)
        adjustment = std::min(state().riskVeto / 3, (250 - risk) * state().riskVeto / 750);

    // First specialist: a conservative tactical verification mode. It does not
    // replace alpha-beta; when routed, it buys back reduction on volatile moves.
    if (state().specialistEnabled && risk >= state().riskThreshold
        && (givesCheck || capture || pvNode))
        adjustment -= state().specialistVeto;

    return std::clamp(adjustment, -4096, 1536);
}

}  // namespace Stockfish::Leviathan::Control

#endif  // LEVIATHAN_CONTROL_H_INCLUDED

/*
  Project Leviathan optional search-regret instrumentation.

  Disabled by default. When enabled it samples LMR events into JSONL using
  engine-native position keys and raw move encodings. This is research data;
  no trace state influences the search result.
*/
#ifndef LEVIATHAN_TRACE_H_INCLUDED
#define LEVIATHAN_TRACE_H_INCLUDED

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

#include "leviathan_attributes.h"
#include "types.h"

namespace Stockfish::Leviathan::Trace {

constexpr int RiskFeatures = 12;

struct State {
    std::string file;
    int samplePermille = 0;
    std::ofstream out;
    std::mutex mutex;
    std::atomic<u64> counter{0};

    State() {
        if (const char* p = std::getenv("LEVIATHAN_TRACE_FILE"))
            file = p;
        if (const char* s = std::getenv("LEVIATHAN_TRACE_SAMPLE_PERMILLE"))
            samplePermille = std::clamp(std::atoi(s), 0, 1000);
        if (!file.empty() && samplePermille > 0)
            out.open(file, std::ios::app);
    }
};

inline State& state() {
    static State s;
    return s;
}

inline bool ready() {
    const auto& s = state();
    return s.samplePermille > 0 && !s.file.empty();
}

inline void set_file(const std::string& path) {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.out.is_open())
        s.out.close();
    s.file = path;
    if (!path.empty() && s.samplePermille > 0)
        s.out.open(path, std::ios::app);
}

inline void set_sample_permille(int v) {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.samplePermille = std::clamp(v, 0, 1000);
    if (s.samplePermille > 0 && !s.file.empty() && !s.out.is_open())
        s.out.open(s.file, std::ios::app);
    if (s.samplePermille == 0 && s.out.is_open())
        s.out.close();
}

LEVIATHAN_NOINLINE inline std::array<int, RiskFeatures> features(Depth depth,
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
    return {
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
}

LEVIATHAN_NOINLINE inline void record_lmr(u64 parentKey,
                       u16 moveRaw,
                       const std::array<int, RiskFeatures>& x,
                       Value reducedValue,
                       Value finalValue,
                       Depth reducedDepth,
                       Depth fullDepth,
                       bool researched) {
    auto& s = state();
    if (s.samplePermille <= 0 || s.file.empty())
        return;
    const u64 n = s.counter.fetch_add(1, std::memory_order_relaxed);
    if (int(n % 1000) >= s.samplePermille)
        return;

    const int regret = std::max(0, int(finalValue - reducedValue));
    const bool dangerous = researched && regret >= 25;

    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.out.is_open())
        s.out.open(s.file, std::ios::app);
    if (!s.out)
        return;

    s.out << "{\"position_key\":" << parentKey << ",\"move_raw\":" << moveRaw
          << ",\"features\":[";
    for (int i = 0; i < RiskFeatures; ++i)
    {
        if (i)
            s.out << ',';
        s.out << x[i];
    }
    s.out << "],\"reduced_value\":" << reducedValue << ",\"final_value\":" << finalValue
          << ",\"reduced_depth\":" << reducedDepth << ",\"full_depth\":" << fullDepth
          << ",\"researched\":" << (researched ? "true" : "false")
          << ",\"regret_cp\":" << regret << ",\"regret\":"
          << (dangerous ? "true" : "false")
          << ",\"bonus\":" << std::clamp(regret * 8, 0, 8192)
          << ",\"confidence\":" << std::clamp(regret * 10, 0, 1000)
          << ",\"kind\":\"episode\",\"source\":\"lmr-trace\"}\n";
}

}  // namespace Stockfish::Leviathan::Trace
#endif

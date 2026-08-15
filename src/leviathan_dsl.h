/*
  Project Leviathan bounded Search DSL.

  Offline Search Foundry candidates compile to this tiny bytecode. The DSL can
  only return a bounded LMR adjustment; it cannot touch board state, legality,
  terminal values, the TT, or NNUE. Authority 1 is conservative: positive
  (more aggressive) reductions are suppressed.
*/

#ifndef LEVIATHAN_DSL_H_INCLUDED
#define LEVIATHAN_DSL_H_INCLUDED

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "types.h"

namespace Stockfish::Leviathan::DSL {

constexpr int FeatureCount = 12;
constexpr int MaxInstructions = 32;

enum class Op {
    Add,
    MulAdd,
    IfGtAdd,
    IfLtAdd,
    Clamp
};

struct Instruction {
    Op  op      = Op::Add;
    int feature = 0;
    int a       = 0;
    int b       = 0;
};

struct State {
    bool enabled   = false;
    int  authority = 0;  // 0 off, 1 veto/deepen only, 2 bounded bidirectional
    int  weight    = 100;
    std::string file;
    std::vector<Instruction> code;
    bool loaded = false;

    State() {
        const char* e = std::getenv("LEVIATHAN_DSL");
        enabled = e && *e && std::string(e) != "0" && std::string(e) != "false";
        if (const char* a = std::getenv("LEVIATHAN_DSL_AUTHORITY"))
            authority = std::clamp(std::atoi(a), 0, 2);
        if (const char* w = std::getenv("LEVIATHAN_DSL_WEIGHT"))
            weight = std::clamp(std::atoi(w), 0, 400);
        if (const char* p = std::getenv("LEVIATHAN_DSL_FILE"))
            file = p;
    }
};

inline State& state() {
    static State s;
    return s;
}

inline bool load(const std::string& path) {
    auto& s = state();
    s.code.clear();
    s.loaded = false;
    s.file   = path;
    if (path.empty())
        return false;

    std::ifstream in(path);
    if (!in)
        return false;

    std::string magic;
    in >> magic;
    if (magic != "LVSD1")
        return false;

    std::string op;
    while (in >> op)
    {
        if (s.code.size() >= MaxInstructions)
        {
            s.code.clear();
            return false;
        }

        Instruction ins;
        if (op == "ADD")
        {
            ins.op = Op::Add;
            if (!(in >> ins.a))
                return false;
        }
        else if (op == "MULADD")
        {
            ins.op = Op::MulAdd;
            if (!(in >> ins.feature >> ins.a >> ins.b) || ins.feature < 0
                || ins.feature >= FeatureCount || ins.b < 1 || ins.b > 65536)
                return false;
        }
        else if (op == "IFGT")
        {
            ins.op = Op::IfGtAdd;
            if (!(in >> ins.feature >> ins.a >> ins.b) || ins.feature < 0
                || ins.feature >= FeatureCount)
                return false;
        }
        else if (op == "IFLT")
        {
            ins.op = Op::IfLtAdd;
            if (!(in >> ins.feature >> ins.a >> ins.b) || ins.feature < 0
                || ins.feature >= FeatureCount)
                return false;
        }
        else if (op == "CLAMP")
        {
            ins.op = Op::Clamp;
            if (!(in >> ins.a >> ins.b) || ins.a > ins.b)
                return false;
        }
        else
            return false;

        if (std::abs(ins.a) > 65536 || std::abs(ins.b) > 65536)
            return false;
        s.code.push_back(ins);
    }

    if (!in.eof() || s.code.empty())
    {
        s.code.clear();
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
inline void set_authority(int v) { state().authority = std::clamp(v, 0, 2); }
inline void set_weight(int v) { state().weight = std::clamp(v, 0, 400); }
inline bool set_file(const std::string& path) { return load(path); }
inline bool ready() { return state().enabled && state().authority > 0 && state().loaded; }

inline int eval(const std::array<int, FeatureCount>& x) {
    if (!ready())
        return 0;

    int out = 0;
    for (const auto& ins : state().code)
    {
        switch (ins.op)
        {
        case Op::Add : out += ins.a; break;
        case Op::MulAdd : out += std::clamp(x[ins.feature], -4096, 4096) * ins.a / ins.b; break;
        case Op::IfGtAdd : out += x[ins.feature] > ins.a ? ins.b : 0; break;
        case Op::IfLtAdd : out += x[ins.feature] < ins.a ? ins.b : 0; break;
        case Op::Clamp : out = std::clamp(out, ins.a, ins.b); break;
        }
        out = std::clamp(out, -4096, 2048);
    }

    if (state().authority == 1)
        out = std::min(out, 0);
    return std::clamp(out * state().weight / 100, -4096, 1536);
}

inline int lmr_adjustment(Depth depth,
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
    const std::array<int, FeatureCount> x = {
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
    return eval(x);
}

}  // namespace Stockfish::Leviathan::DSL

#endif  // LEVIATHAN_DSL_H_INCLUDED

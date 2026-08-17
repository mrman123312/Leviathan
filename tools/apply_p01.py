#!/usr/bin/env python3
from pathlib import Path

# DSL readiness
p=Path('src/leviathan_dsl.h');s=p.read_text()
old='''inline bool ready() { return state().enabled && state().authority > 0 && state().loaded; }

inline int eval(const std::array<int, FeatureCount>& x) {
    if (!ready())
        return 0;

    int out = 0;
    for (const auto& ins : state().code)
'''
new='''inline bool ready() {
    const auto& s = state();
    return s.enabled && s.authority > 0 && s.loaded;
}

inline int eval_ready(const std::array<int, FeatureCount>& x) {
    const auto& s = state();
    int out = 0;
    for (const auto& ins : s.code)
'''
assert s.count(old)==1
s=s.replace(old,new,1)
s=s.replace('''    if (state().authority == 1)
        out = std::min(out, 0);
    return std::clamp(out * state().weight / 100, -4096, 1536);
}

inline int lmr_adjustment''','''    if (s.authority == 1)
        out = std::min(out, 0);
    return std::clamp(out * s.weight / 100, -4096, 1536);
}

inline int eval(const std::array<int, FeatureCount>& x) {
    return ready() ? eval_ready(x) : 0;
}

inline int lmr_adjustment''',1)
needle='''                          Value staticEval,
                          Value alpha) {
    const std::array<int, FeatureCount> x = {'''
repl='''                          Value staticEval,
                          Value alpha) {
    if (!ready())
        return 0;

    const std::array<int, FeatureCount> x = {'''
assert s.count(needle)==1
s=s.replace(needle,repl,1).replace('''    return eval(x);
}

}  // namespace Stockfish::Leviathan::DSL''','''    return eval_ready(x);
}

}  // namespace Stockfish::Leviathan::DSL''',1)
p.write_text(s)

# Trace readiness
p=Path('src/leviathan_trace.h');s=p.read_text();anchor='''    return s;
}

inline void set_file''';repl='''    return s;
}

inline bool ready() {
    const auto& s = state();
    return s.samplePermille > 0 && !s.file.empty();
}

inline void set_file''';assert s.count(anchor)==1;p.write_text(s.replace(anchor,repl,1))

# MovePicker snapshots
p=Path('src/movepick.cpp');s=p.read_text();anchor='''        threatByLesser[KING]  = 0;
    }

    ExtMove* it = cur;''';repl='''        threatByLesser[KING]  = 0;
    }

    const bool leviathanPolicyReady = Type == QUIETS && Leviathan::Policy::ready();
    const bool leviathanAtlasReady  = Type == QUIETS && Leviathan::Atlas::ready();
    const bool leviathanRule50Ready = Type == QUIETS && Leviathan::Fundamentals::ready()
                                      && Leviathan::Fundamentals::state().rule50Pressure
                                      && pos.rule50_count() >= 70;

    ExtMove* it = cur;''';assert s.count(anchor)==1;s=s.replace(anchor,repl,1)
anchor='''            m.value += Leviathan::Policy::ordering_bonus(pos, m);
            m.value += Leviathan::Atlas::ordering_bonus(pos, m);
            m.value += Leviathan::Fundamentals::quiet_ordering_bonus(pos, m);''';repl='''            if (leviathanPolicyReady)
                m.value += Leviathan::Policy::ordering_bonus(pos, m);
            if (leviathanAtlasReady)
                m.value += Leviathan::Atlas::ordering_bonus(pos, m);
            if (leviathanRule50Ready)
                m.value += Leviathan::Fundamentals::quiet_ordering_bonus(pos, m);''';assert s.count(anchor)==1;p.write_text(s.replace(anchor,repl,1))

# Search snapshots
p=Path('src/search.cpp');s=p.read_text();anchor='''    int moveCount = 0;

    // Step 13. Loop through all pseudo-legal moves''';repl='''    int moveCount = 0;

    const bool leviathanRiskReady  = Leviathan::Control::risk_ready();
    const bool leviathanDslReady   = Leviathan::DSL::ready();
    const bool leviathanTraceReady = Leviathan::Trace::ready();

    // Step 13. Loop through all pseudo-legal moves''';assert s.count(anchor)==1;s=s.replace(anchor,repl,1)
assert s.count('        const u64 leviathanParentKey = u64(pos.key());')==1;s=s.replace('        const u64 leviathanParentKey = u64(pos.key());','        const u64 leviathanParentKey = leviathanTraceReady ? u64(pos.key()) : 0;',1)
anchor='''        r += Leviathan::Control::lmr_adjustment(
          depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,
          givesCheck, ttData.depth, ss->staticEval, alpha);
        r += Leviathan::DSL::lmr_adjustment(
          depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,
          givesCheck, ttData.depth, ss->staticEval, alpha);''';repl='''        if (leviathanRiskReady)
            r += Leviathan::Control::lmr_adjustment(
              depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,
              givesCheck, ttData.depth, ss->staticEval, alpha);
        if (leviathanDslReady)
            r += Leviathan::DSL::lmr_adjustment(
              depth, moveCount, ss->statScore, correctionValue, PvNode, cutNode, allNode, capture,
              givesCheck, ttData.depth, ss->staticEval, alpha);''';assert s.count(anchor)==1;s=s.replace(anchor,repl,1)
anchor='''        if (leviathanReducedValue != VALUE_NONE)
            Leviathan::Trace::record_lmr(''';repl='''        if (leviathanTraceReady && leviathanReducedValue != VALUE_NONE)
            Leviathan::Trace::record_lmr(''';assert s.count(anchor)==1;p.write_text(s.replace(anchor,repl,1))
print('P0+P1 applied')

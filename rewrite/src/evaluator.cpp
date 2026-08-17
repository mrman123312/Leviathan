#include "evaluator.h"

namespace leviathan {

Evaluation BaselineEvaluator::evaluate(const Position& position) const {
    return evaluate_position(position);
}

const EvaluatorDescriptor& BaselineEvaluator::descriptor() const {
    static constexpr EvaluatorDescriptor d{
        "leviathan-baseline-v0",
        "native",
        "none",
        "Leviathan project",
        EvaluatorOrigin::Native
    };
    return d;
}

const Evaluator& default_evaluator() {
    static const BaselineEvaluator evaluator;
    return evaluator;
}

} // namespace leviathan

#pragma once
#include "chess.h"
#include <string_view>

namespace leviathan {

enum class EvaluatorOrigin : uint8_t {
    Native,
    ImportedCode,
    ExternalModel,
    Hybrid
};

struct EvaluatorDescriptor {
    std::string_view id;
    std::string_view donor_id;
    std::string_view model_id;
    std::string_view license;
    EvaluatorOrigin origin = EvaluatorOrigin::Native;
};

class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual Evaluation evaluate(const Position& position) const = 0;
    virtual const EvaluatorDescriptor& descriptor() const = 0;
};

class BaselineEvaluator final : public Evaluator {
public:
    Evaluation evaluate(const Position& position) const override;
    const EvaluatorDescriptor& descriptor() const override;
};

const Evaluator& default_evaluator();

} // namespace leviathan

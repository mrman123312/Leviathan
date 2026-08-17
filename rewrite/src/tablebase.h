#pragma once
#include "chess.h"
#include <optional>
#include <string_view>

namespace leviathan {

enum class Wdl : int8_t {
    Loss = -2,
    BlessedLoss = -1,
    Draw = 0,
    CursedWin = 1,
    Win = 2
};

struct TablebaseDescriptor {
    std::string_view id;
    std::string_view donor_id;
    std::string_view license;
    std::string_view source_revision;
};

class Tablebase {
public:
    virtual ~Tablebase() = default;
    virtual std::optional<Wdl> probe_wdl(const Position& position) const = 0;
    virtual const TablebaseDescriptor& descriptor() const = 0;
    virtual int max_pieces() const = 0;
};

class NullTablebase final : public Tablebase {
public:
    std::optional<Wdl> probe_wdl(const Position&) const override { return std::nullopt; }
    const TablebaseDescriptor& descriptor() const override;
    int max_pieces() const override { return 0; }
};

// Optional adapter for the pinned jdart1/Fathom donor. The implementation is
// compiled only when LEVIATHAN_FATHOM_DIR points at materialized pinned source.
class FathomTablebase final : public Tablebase {
public:
    FathomTablebase() = default;
    ~FathomTablebase() override;

    bool initialize(std::string_view path);
    void reset();
    bool available() const { return available_; }

    std::optional<Wdl> probe_wdl(const Position& position) const override;
    const TablebaseDescriptor& descriptor() const override;
    int max_pieces() const override;

private:
    bool available_ = false;
};

const Tablebase& null_tablebase();

} // namespace leviathan

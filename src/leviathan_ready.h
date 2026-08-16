/*
  PROJECT LEVIATHAN: immutable-per-search readiness snapshot.

  UCI options and model/file configuration are established before a `go`.
  Snapshot those global predicates once for each worker instead of repeatedly
  rebuilding them at every node and quiet MovePicker list.
*/

#ifndef LEVIATHAN_READY_H_INCLUDED
#define LEVIATHAN_READY_H_INCLUDED

#include "leviathan_atlas.h"
#include "leviathan_control.h"
#include "leviathan_dsl.h"
#include "leviathan_fundamentals.h"
#include "leviathan_policy.h"
#include "leviathan_trace.h"
#include "types.h"

namespace Stockfish::Leviathan::Ready {

using Mask = u8;

enum Flag : Mask {
    Risk   = 1U << 0,
    Dsl    = 1U << 1,
    Trace  = 1U << 2,
    Policy = 1U << 3,
    Atlas  = 1U << 4,
    Rule50 = 1U << 5,
};

constexpr bool has(Mask mask, Flag flag) { return (mask & flag) != 0; }

inline Mask snapshot() {
    Mask mask = 0;
    if (Control::risk_ready())
        mask |= Risk;
    if (DSL::ready())
        mask |= Dsl;
    if (Trace::ready())
        mask |= Trace;
    if (Policy::ready())
        mask |= Policy;
    if (Atlas::ready())
        mask |= Atlas;

    const auto& fundamentals = Fundamentals::state();
    if (fundamentals.enabled && fundamentals.authority > 0 && fundamentals.rule50Pressure)
        mask |= Rule50;
    return mask;
}

}  // namespace Stockfish::Leviathan::Ready

#endif  // #ifndef LEVIATHAN_READY_H_INCLUDED

# Leviathan Speed v3 — Phase A

Goal: reduce wall-clock cost without changing search topology, evaluation, move ordering, legality, TT semantics, or terminal values.

Frozen strength reference: `fbccfb6eb5cd335b1ce8fc5c5efad9e36be4e19d` (v2.1.2 critical-pawn).

Phase A targets avoidable framework overhead observed in `speed-v3-hotspot-profile`:

- dormant DSL hot-loop feature construction;
- dormant trace feature construction / record calls;
- dormant Policy / Atlas quiet-ordering calls;
- repeated Fundamentals classification of the same move.

Hard gate: same deterministic bench node count and same bench best-move sequence as the reference. Any functional divergence is not a lossless speed optimization and must be moved to the strength track.

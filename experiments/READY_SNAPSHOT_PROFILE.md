# Optional-organ readiness profile

The profile used optimized AVX2 `-pg` builds and the active 3,210,480-node bench. Function-call counts are causal diagnostics; sampled self-time percentages have 0.01-second granularity and are not speed evidence.

| Function / path | Active fbcc reference | P0+P1+P2+P3 stack |
|---|---:|---:|
| NNUE `apply_combined` calls | 2,907,075 | 2,907,075 |
| NNUE network evaluations | 1,234,256 | 1,234,256 |
| `MovePicker::next_move` calls | 7,921,118 | 7,921,118 |
| Disabled DSL LMR adjustment calls | 2,872,729 | 0 |
| Disabled Trace feature construction calls | 1,661,869 | 0 |
| Disabled Trace record calls | 1,695,255 | 0 |
| Per-node Risk readiness calls | implicit in adjustment path | 1,076,149 |
| Per-node DSL state/readiness calls | implicit in adjustment path | 1,076,149 |
| Per-node Trace state/readiness calls | implicit in record path | 1,073,655 |
| `Fundamentals::state()` calls | 5,131,637 | 2,622,348 |

P1 successfully removes the more expensive move-level dormant paths, but it converts global readiness into a node-local cache. P4 tests the stronger invariant: optional-organ readiness is immutable during a UCI search and can be represented by one per-worker bitmask captured at `start_searching()`.

The P4 prototype changes no active-organ computation. Position-dependent rule-50 eligibility still checks the current halfmove clock; only the global Fundamentals/rule-50 configuration is snapshotted.

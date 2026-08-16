# Persistent search headroom V2

This experiment measures the value and risk of preserving search state across two
chronological game plies. It does **not** implement a persistent search forest and
does not establish strength.

## Three-arm causal design

- **Warm:** TT entries and history tables survive the root and intermediate searches.
- **History-only:** the same prefixes are searched, then a diagnostic UCI command
  clears only the TT before the target.
- **Cold:** `Clear Hash` is issued before the target. In this Leviathan/Stockfish
  lineage that calls both `tt.clear()` and `threads.clear()`, clearing TT and worker
  histories.

All target searches receive 24,000 nodes. Execution order rotates among the three
separate engine processes. An official frozen-parent oracle searches every unique
selected move with 200,000 nodes forced at the root.

## Local 50-position diagnostic

| Comparison | Oracle W/L/T | Mean regret advantage | Median target wall ratio | Depth advantage |
|---|---:|---:|---:|---:|
| Warm vs cold | 5/5/40 | -0.28 cp | 1.27235 | +0.84 |
| History-only vs cold | 5/8/37 | -0.28 cp | 1.18454 | +0.46 |
| Warm vs history-only | 5/4/41 | 0.00 cp incremental TT effect | — | +0.38 |

Warm and cold selected the same move in 80% of positions. Warm searches reached
an average 2.6 greater selective depth than cold.

The first 20-position prefix looked negative (warm/cold 1/4/15), but the
predeclared expansion to all 50 positions became 5/5/40. The correct conclusion
is therefore **no demonstrated decision-quality effect**, not that persistence is
harmful. The computational headroom is large enough to justify further research.

## Surviving design constraint

Exact TT evidence appears safer than undifferentiated heuristic carryover, but this
panel is too small to prove even that distinction. A future persistent mechanism
must attach provenance and invalidation rules to reused state, use equal total
memory, and compare against the normal warm TT rather than an artificially cold
engine. The next justified prototype is a bounded likely-reply frontier used for
ordering only; bound reuse remains forbidden until repetition, rule-50, depth,
window, and mate-distance semantics are proven valid.

The hosted replication is tracked as W023. Local wall time is treated as a screen,
not a platform-independent speed measurement.


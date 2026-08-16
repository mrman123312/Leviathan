# Rival Ambiguity Reuse v3 — screen result

Source idea: Stockfish already pays for an excluded-move singular search to test whether the TT move is uniquely strong. The continuous fact that a rival nearly challenged the TT move is mostly discarded after singular/multicut/negative-extension classification.

Screen: 40 equal-node + 40 fixed-time games on a common deterministic panel.

- control: 51.25% equal-node / 51.25% fixed-time; bench 3,210,480 nodes.
- gentle: 52.5% / 52.5%; bench 3,246,093 nodes.
- medium: 43.75% / 45.0%; reject.
- PV-only: **55.0% equal-node / 52.5% fixed-time**; bench **2,757,927 nodes**.

PV-only mechanism:
- record `leviathanRivalAmbiguity = 1` when the already-paid singular excluded-move search returns `value >= singularBeta && value < beta` and the score is non-decisive;
- on later quiet, non-check rival moves at depth >= 6 / moveCount <= 10, buy back 320 reduction units only when `ss->ttPv` is true.

Decision: PV-only is a promotion candidate, not a proven gain. Freeze exact implementation and test on fresh 100-game equal-node and fixed-time holdouts.

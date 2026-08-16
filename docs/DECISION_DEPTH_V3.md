# Decision Depth v3

Research question: should a forced move consume the same search-depth budget as a genuine decision?

A position with exactly one legal move has branching factor one. Advancing through that move does not choose among alternatives. A depth metric intended to ration decision complexity can therefore preserve one ply of nominal depth across that exact corridor.

Initial profiles:

1. `control`: v2.1.2 critical-pawn reference.
2. `forced-check`: preserve one unit of depth only when in check and exactly one legal evasion exists.
3. `forced-any`: preserve one unit of depth whenever exactly one legal move exists.

No probabilistic "near forced" classification is permitted in this experiment.

Gates:
- equal-node games are primary for search quality;
- fixed-time games measure whether legal-move detection costs erase the gain;
- deterministic correctness/perft remains unchanged;
- any profile below 50% in the screen is rejected under the current project rule.

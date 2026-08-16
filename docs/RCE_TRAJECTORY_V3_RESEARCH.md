# RCE trajectory-search research

## Question
Can chronological lookahead be decoupled from exhaustive ply-depth so Leviathan can inspect much longer futures without pretending to solve the entire game tree?

## Regular route
Sparse near-alpha verification. v2.5 found an equal-node signal but unacceptable tree expansion.

## Bold route
Adversarial Trajectory Lattice (ATL): preserve several candidate futures, repeatedly extend their principal continuations, explicitly retain opponent alternatives, and use long-horizon trajectory stability as advisory information.

## Revolutionary route
Decision Distance: distinguish chronological plies from genuine branching decisions. Forced and low-choice corridors should be represented as macro-edges. Search budget belongs at decision frontiers and proof obligations, not uniformly at every chronological move.

## Hybrid probe
Use the proven Stockfish/Leviathan search as a local tactical oracle. Stitch short PV segments into long trajectories, maintain a bounded adversarial beam, and compare trajectory-derived root choices against a much deeper Stockfish oracle under a strict equal-node advisory budget.

This is a research organ, not a promotion. It earns integration only if it reduces deep-oracle root regret on held-out positions and later survives direct games.

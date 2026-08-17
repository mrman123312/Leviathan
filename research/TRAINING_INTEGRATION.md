# Fundamentals Ultra training/data integration

This directory imports only architecture-independent, provenance-verified research from `leviathan/rewrite-v4-dual-training-data`.

It deliberately does **not** alter the playing engine or evaluator. Public Lc0 native targets and Stockfish-derived teacher/training views stay lineage-linked so one raw position is not double-counted as independent evidence.

Promotion rule: a future learned evaluator must beat the frozen Stockfish NNUE / Fundamentals controls on grouped fresh holdouts, equal-node deep-oracle regret, equal-time regret, and match gates before it can replace the production evaluator.

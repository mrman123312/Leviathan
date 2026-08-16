# Leviathan Rewrite v0

A greenfield chess-engine core for the Leviathan project.

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Smoke test:

```text
uci
isready
position startpos
perft 4
go depth 4
quit
```

Known v0 limitations: deliberately simple evaluation/search, no Syzygy, no NNUE, no SMP, no opening book, no advanced time manager. Those are withheld until the independent rules/search contracts are validated.

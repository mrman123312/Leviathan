# Leviathan: strength-first ARC-AGI-1 research

**One frozen Qwen3-1.7B-Base. No neural training. Exact grid reasoning, not ARC-Easy.**

Extract this complete folder and double-click **RUN_ARC_AGI_1.bat**. It runs 24 fixed public evaluation tasks with plain-Qwen, symbolic-search, and hybrid controls. **RUN_ARC_AGI_1_FULL.bat** runs all 400. The existing v7 CUDA environment and cached Qwen are reused; no installs, drive scanning or downloads. Both scripts retain console output and save task-by-task progress in timestamped `results_strength` folders, then open RESULTS.html. Keep the command window open while inference/search executes.

The original **RUN_ARC_EASY.bat** remains available and unchanged. It is a different benchmark.

## What changed

Counterexample-constrained program repair; two hypothesis languages; object/grid/panel/topology representations; inverse-goal joins and binary composition; multiple support-derived neural slots; current-evidence-gated middle-layer recurrence and cell interventions; dual-form skill memory; exact macro expansion; and a no-progress scheduler. These are internal algorithms and interventions in one parameter owner, not a collection of LLM agents.

Every final synthesized program must fit every visible example. Hidden query outputs are withheld from the solver and are read only after all predictions are sealed. A demonstration fit is not counted as solving a task. Two guesses per query are allowed in the reported metric and every grid cell must match.

## Evidence, not promises

See `evidence/strength` for actual local/native test records and the public TRAINING development control. Symbolic-only scores do not show a neural-Qwen gain. Native tiny random Qwen tests do not establish 1.7B GPU accuracy. No claim is made that this implementation has beaten ARC-AGI-1.

See [the implementation/proof/limits report](docs/25-strength-arc-agi.md) and [the machine-readable policy](spec/strength-arc.toml). The next real pretrained run determines whether model proposals/internal interventions improve the exact-grid result beyond the symbolic control.

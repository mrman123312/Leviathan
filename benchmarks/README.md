# Single-model benchmark

Run from the repository root:

```bash
python -m pip install -e .
python benchmarks/benchmark_single_model.py
```

The benchmark compares a 582-parameter `UnifiedMoP` with a 591-parameter dense
conditional MLP over seeds `7`, `17` and `29`, using the same 768 examples, 1,200
minibatch updates and identical initial dense path per seed.

The primary teacher is deliberately conditional and low rank. It asks whether the
proposed parameter basis can recover the structure it was designed for; it is not a
neutral general-capability benchmark. Training contexts are one-hot and the composition
split combines two held-out context coordinates. A dense nonlinear teacher is the
negative control.

The benchmark contains two routing ablations:

1. train with every basis, then prune to Top-2 at inference;
2. warm up with every basis for one third of training, then continue the same model,
   optimizer and checkpoint with Top-2 routing.

Promotion gates cover exact insertion parity, matched total parameters, held-out and
composition MSE, improvement over post-hoc pruning and estimated active MACs. Measured
NumPy latency is the median of 300 batch-512 runs after 30 warmups on the final seed. It
is reported separately and cannot be replaced by the MAC estimate.

The checked-in result is `results/single_model_v0.4.0.json`. Latency is specific to the
recorded Python/NumPy/machine environment and should be regenerated for a new runtime.

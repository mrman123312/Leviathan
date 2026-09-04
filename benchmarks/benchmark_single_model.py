"""Benchmark one tensorized MoP model against a matched dense baseline.

The synthetic task is deliberately narrow: learn several context-conditioned linear
operators from one-hot training contexts, then evaluate fresh samples and unseen
two-context compositions.  It tests the proposed conditional parameter substrate; it
does not stand in for language, agency, or AGI evaluation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from math import sqrt
from statistics import mean, median, pstdev

import numpy as np
from numpy.typing import NDArray

from leviathan.mop import AdamOptimizer, MoPConfig, UnifiedMoP

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TaskShape:
    input_dim: int = 12
    context_dim: int = 8
    output_dim: int = 6
    basis_count: int = 8
    rank: int = 3


class DenseConditionalMLP:
    """Parameter-matched dense residual MLP baseline."""

    def __init__(self, shape: TaskShape, hidden_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        joint_dim = shape.input_dim + shape.context_dim
        self.shape = shape
        self.hidden_dim = hidden_dim
        self.parameters: dict[str, FloatArray] = {
            "base_weight": rng.normal(
                0.0,
                1.0 / sqrt(shape.input_dim),
                size=(shape.input_dim, shape.output_dim),
            ),
            "base_bias": np.zeros(shape.output_dim, dtype=np.float64),
            "hidden_weight": rng.normal(
                0.0,
                1.0 / sqrt(joint_dim),
                size=(joint_dim, hidden_dim),
            ),
            "hidden_bias": np.zeros(hidden_dim, dtype=np.float64),
            "output_weight": np.zeros((hidden_dim, shape.output_dim), dtype=np.float64),
        }

    @property
    def parameter_count(self) -> int:
        return sum(value.size for value in self.parameters.values())

    @property
    def estimated_macs(self) -> int:
        shape = self.shape
        return (
            shape.input_dim * shape.output_dim
            + (shape.input_dim + shape.context_dim) * self.hidden_dim
            + self.hidden_dim * shape.output_dim
        )

    def forward(self, inputs: FloatArray, context: FloatArray) -> FloatArray:
        joint = np.concatenate((inputs, context), axis=1)
        hidden = np.tanh(joint @ self.parameters["hidden_weight"] + self.parameters["hidden_bias"])
        return (
            inputs @ self.parameters["base_weight"]
            + self.parameters["base_bias"]
            + hidden @ self.parameters["output_weight"]
        )

    def loss_and_gradients(
        self,
        inputs: FloatArray,
        context: FloatArray,
        targets: FloatArray,
    ) -> tuple[float, dict[str, FloatArray]]:
        joint = np.concatenate((inputs, context), axis=1)
        hidden = np.tanh(joint @ self.parameters["hidden_weight"] + self.parameters["hidden_bias"])
        predictions = (
            inputs @ self.parameters["base_weight"]
            + self.parameters["base_bias"]
            + hidden @ self.parameters["output_weight"]
        )
        residual = predictions - targets
        output_gradient = 2.0 * residual / residual.size
        hidden_gradient = (output_gradient @ self.parameters["output_weight"].T) * (
            1.0 - np.square(hidden)
        )
        return float(np.mean(np.square(residual))), {
            "base_weight": inputs.T @ output_gradient,
            "base_bias": np.sum(output_gradient, axis=0),
            "output_weight": hidden.T @ output_gradient,
            "hidden_weight": joint.T @ hidden_gradient,
            "hidden_bias": np.sum(hidden_gradient, axis=0),
        }


class DenseAdam:
    def __init__(
        self,
        parameters: Mapping[str, FloatArray],
        *,
        learning_rate: float,
    ) -> None:
        self.learning_rate = learning_rate
        self.step_count = 0
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}

    def step(
        self,
        parameters: Mapping[str, FloatArray],
        gradients: Mapping[str, FloatArray],
    ) -> None:
        self.step_count += 1
        for name, parameter in parameters.items():
            gradient = gradients[name]
            self.first[name] = 0.9 * self.first[name] + 0.1 * gradient
            self.second[name] = 0.999 * self.second[name] + 0.001 * np.square(gradient)
            first_hat = self.first[name] / (1.0 - 0.9**self.step_count)
            second_hat = self.second[name] / (1.0 - 0.999**self.step_count)
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)


@dataclass(frozen=True, slots=True)
class Teacher:
    base: FloatArray
    down: FloatArray
    up: FloatArray


def make_teacher(shape: TaskShape, seed: int) -> Teacher:
    rng = np.random.default_rng(seed)
    return Teacher(
        base=rng.normal(0.0, 0.35, size=(shape.input_dim, shape.output_dim)),
        down=rng.normal(
            0.0,
            0.45,
            size=(shape.basis_count, shape.input_dim, shape.rank),
        ),
        up=rng.normal(
            0.0,
            0.45,
            size=(shape.basis_count, shape.rank, shape.output_dim),
        ),
    )


def sample_dataset(
    shape: TaskShape,
    teacher: Teacher,
    count: int,
    seed: int,
    *,
    compositions: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    inputs = rng.normal(0.0, 1.0, size=(count, shape.input_dim))
    context = np.zeros((count, shape.context_dim), dtype=np.float64)
    if compositions:
        first = rng.integers(0, shape.context_dim, size=count)
        offset = rng.integers(1, shape.context_dim, size=count)
        second = (first + offset) % shape.context_dim
        context[np.arange(count), first] = 0.5
        context[np.arange(count), second] = 0.5
    else:
        regimes = rng.integers(0, shape.context_dim, size=count)
        context[np.arange(count), regimes] = 1.0

    hidden = np.einsum("bi,eir->ber", inputs, teacher.down)
    updates = np.einsum("ber,ero->beo", hidden, teacher.up)
    targets = inputs @ teacher.base + np.einsum("be,beo->bo", context, updates)
    return inputs, context, targets


def mse(actual: FloatArray, expected: FloatArray) -> float:
    return float(np.mean(np.square(actual - expected)))


def route_entropy(gates: FloatArray) -> float:
    safe = np.clip(gates, 1e-12, 1.0)
    return float(np.mean(-np.sum(safe * np.log(safe), axis=1)))


def choose_dense_width(shape: TaskShape, target_parameters: int) -> int:
    base = shape.input_dim * shape.output_dim + shape.output_dim
    per_hidden = shape.input_dim + shape.context_dim + 1 + shape.output_dim
    return max(1, round((target_parameters - base) / per_hidden))


def train_seed(shape: TaskShape, seed: int, steps: int) -> tuple[dict[str, float], object]:
    teacher = make_teacher(shape, seed + 1000)
    train_x, train_c, train_y = sample_dataset(shape, teacher, 768, seed + 2000)
    test_x, test_c, test_y = sample_dataset(shape, teacher, 2048, seed + 3000)
    comp_x, comp_c, comp_y = sample_dataset(
        shape,
        teacher,
        2048,
        seed + 4000,
        compositions=True,
    )

    config = MoPConfig(
        input_dim=shape.input_dim,
        context_dim=shape.context_dim,
        output_dim=shape.output_dim,
        basis_count=shape.basis_count,
        rank=shape.rank,
        temperature=0.7,
        seed=seed,
    )
    dense_route_model = UnifiedMoP(config)
    sparse_model = UnifiedMoP(
        MoPConfig(
            input_dim=shape.input_dim,
            context_dim=shape.context_dim,
            output_dim=shape.output_dim,
            basis_count=shape.basis_count,
            rank=shape.rank,
            temperature=0.5,
            seed=seed,
        )
    )
    dense = DenseConditionalMLP(
        shape,
        hidden_dim=choose_dense_width(shape, dense_route_model.parameter_count),
        seed=seed,
    )
    # Give both models the identical residual base at step zero.
    shared = dense_route_model.state_dict()
    dense.parameters["base_weight"][...] = shared["base_weight"]
    dense.parameters["base_bias"][...] = shared["base_bias"]

    dense_route_optimizer = AdamOptimizer(dense_route_model, learning_rate=0.012)
    sparse_optimizer = AdamOptimizer(sparse_model, learning_rate=0.012)
    dense_optimizer = DenseAdam(dense.parameters, learning_rate=0.012)
    batch_rng = np.random.default_rng(seed + 5000)
    batches = batch_rng.integers(0, train_x.shape[0], size=(steps, 96))
    sparse_warmup = steps // 3
    for step, indices in enumerate(batches):
        _, gradients = dense_route_model.loss_and_gradients(
            train_x[indices],
            train_c[indices],
            train_y[indices],
            entropy_penalty=2e-4,
        )
        dense_route_optimizer.step(dense_route_model, gradients)
        _, sparse_gradients = sparse_model.loss_and_gradients(
            train_x[indices],
            train_c[indices],
            train_y[indices],
            active_bases=None if step < sparse_warmup else 2,
            entropy_penalty=2e-3,
        )
        sparse_optimizer.step(sparse_model, sparse_gradients)
        _, dense_gradients = dense.loss_and_gradients(
            train_x[indices],
            train_c[indices],
            train_y[indices],
        )
        dense_optimizer.step(dense.parameters, dense_gradients)

    dense_test = dense.forward(test_x, test_c)
    dense_composition = dense.forward(comp_x, comp_c)
    posthoc_all = dense_route_model.forward(test_x, test_c)
    posthoc_top_two = dense_route_model.forward(test_x, test_c, active_bases=2)
    sparse_all = sparse_model.forward(test_x, test_c)
    sparse_top_two = sparse_model.forward(test_x, test_c, active_bases=2)
    sparse_top_one = sparse_model.forward(test_x, test_c, active_bases=1)
    sparse_composition = sparse_model.forward(comp_x, comp_c, active_bases=2)
    return {
        "dense_test_mse": mse(dense_test, test_y),
        "dense_composition_mse": mse(dense_composition, comp_y),
        "posthoc_all_test_mse": mse(posthoc_all.output, test_y),
        "posthoc_top2_test_mse": mse(posthoc_top_two.output, test_y),
        "posthoc_route_entropy": route_entropy(posthoc_all.gates),
        "staged_all_test_mse": mse(sparse_all.output, test_y),
        "staged_top2_test_mse": mse(sparse_top_two.output, test_y),
        "staged_top1_test_mse": mse(sparse_top_one.output, test_y),
        "staged_top2_composition_mse": mse(sparse_composition.output, comp_y),
        "staged_route_entropy": route_entropy(sparse_all.gates),
        "mop_parameters": float(sparse_model.parameter_count),
        "dense_parameters": float(dense.parameter_count),
        "mop_top2_active_parameters": float(sparse_model.active_parameter_count(2)),
        "mop_top2_macs": float(sparse_model.estimated_macs(2)),
        "dense_macs": float(dense.estimated_macs),
    }, (sparse_model, dense_route_model, dense, test_x[:512], test_c[:512])


def train_nonlinear_control(shape: TaskShape, seed: int, steps: int) -> dict[str, float]:
    """Run an intentionally out-of-class task so a home-field win is not generalized."""

    hidden_dim = 19
    teacher_rng = np.random.default_rng(seed + 9000)
    teacher_base = teacher_rng.normal(
        0.0,
        0.35,
        size=(shape.input_dim, shape.output_dim),
    )
    teacher_hidden = teacher_rng.normal(
        0.0,
        1.0 / sqrt(shape.input_dim + shape.context_dim),
        size=(shape.input_dim + shape.context_dim, hidden_dim),
    )
    teacher_bias = teacher_rng.normal(0.0, 0.1, size=hidden_dim)
    teacher_output = teacher_rng.normal(
        0.0,
        0.4,
        size=(hidden_dim, shape.output_dim),
    )

    def sample(count: int, sample_seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        rng = np.random.default_rng(sample_seed)
        inputs = rng.normal(size=(count, shape.input_dim))
        context = np.zeros((count, shape.context_dim), dtype=np.float64)
        regimes = rng.integers(0, shape.context_dim, size=count)
        context[np.arange(count), regimes] = 1.0
        joint = np.concatenate((inputs, context), axis=1)
        targets = (
            inputs @ teacher_base + np.tanh(joint @ teacher_hidden + teacher_bias) @ teacher_output
        )
        return inputs, context, targets

    train_x, train_c, train_y = sample(768, seed + 9100)
    test_x, test_c, test_y = sample(2048, seed + 9200)
    model = UnifiedMoP(
        MoPConfig(
            input_dim=shape.input_dim,
            context_dim=shape.context_dim,
            output_dim=shape.output_dim,
            basis_count=shape.basis_count,
            rank=shape.rank,
            temperature=0.5,
            seed=seed,
        )
    )
    dense = DenseConditionalMLP(
        shape,
        hidden_dim=choose_dense_width(shape, model.parameter_count),
        seed=seed,
    )
    shared = model.state_dict()
    dense.parameters["base_weight"][...] = shared["base_weight"]
    dense.parameters["base_bias"][...] = shared["base_bias"]
    model_optimizer = AdamOptimizer(model, learning_rate=0.012)
    dense_optimizer = DenseAdam(dense.parameters, learning_rate=0.012)
    batches = np.random.default_rng(seed + 9300).integers(
        0,
        train_x.shape[0],
        size=(steps, 96),
    )
    sparse_warmup = steps // 3
    for step, indices in enumerate(batches):
        _, gradients = model.loss_and_gradients(
            train_x[indices],
            train_c[indices],
            train_y[indices],
            active_bases=None if step < sparse_warmup else 2,
            entropy_penalty=2e-3,
        )
        model_optimizer.step(model, gradients)
        _, dense_gradients = dense.loss_and_gradients(
            train_x[indices],
            train_c[indices],
            train_y[indices],
        )
        dense_optimizer.step(dense.parameters, dense_gradients)

    return {
        "dense_nonlinear_test_mse": mse(dense.forward(test_x, test_c), test_y),
        "mop_nonlinear_test_mse": mse(
            model.forward(test_x, test_c, active_bases=2).output,
            test_y,
        ),
    }


def benchmark_latency(function: Callable[[], object], repeats: int = 300) -> float:
    for _ in range(30):
        function()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    return median(samples)


def svd_reconstruction_error(shape: TaskShape, seed: int) -> float:
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=(shape.input_dim, shape.output_dim))
    left, singular, right = np.linalg.svd(weight, full_matrices=False)
    reconstructed = (left * singular) @ right
    return float(np.max(np.abs(weight - reconstructed)))


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {"mean": mean(row[key] for row in rows), "std": pstdev(row[key] for row in rows)}
        for key in rows[0]
    }


def run(seeds: list[int], steps: int) -> dict[str, object]:
    shape = TaskShape()
    rows: list[dict[str, float]] = []
    latency_subject: object | None = None
    for seed in seeds:
        row, latency_subject = train_seed(shape, seed, steps)
        rows.append(row)
    nonlinear_rows = [train_nonlinear_control(shape, seed, steps) for seed in seeds]
    assert latency_subject is not None
    sparse_model, dense_route_model, dense, latency_x, latency_c = latency_subject
    latency = {
        "dense_batch512_us": benchmark_latency(lambda: dense.forward(latency_x, latency_c)),
        "mop_all_batch512_us": benchmark_latency(
            lambda: dense_route_model.forward(latency_x, latency_c)
        ),
        "mop_top2_batch512_us": benchmark_latency(
            lambda: sparse_model.forward(latency_x, latency_c, active_bases=2)
        ),
    }
    metrics = aggregate(rows)
    parity_model = UnifiedMoP(
        MoPConfig(
            input_dim=shape.input_dim,
            context_dim=shape.context_dim,
            output_dim=shape.output_dim,
            basis_count=shape.basis_count,
            rank=shape.rank,
            seed=991,
        )
    )
    parity_x = np.random.default_rng(992).normal(size=(128, shape.input_dim))
    parity_c = np.eye(shape.context_dim, dtype=np.float64)[np.arange(128) % shape.context_dim]
    parity_error = float(
        np.max(
            np.abs(
                parity_model.base_prediction(parity_x)
                - parity_model.forward(parity_x, parity_c).output
            )
        )
    )

    dense_mse = metrics["dense_test_mse"]["mean"]
    sparse_mse = metrics["staged_top2_test_mse"]["mean"]
    posthoc_sparse_mse = metrics["posthoc_top2_test_mse"]["mean"]
    posthoc_all_mse = metrics["posthoc_all_test_mse"]["mean"]
    gates = {
        "zero_insertion_parity": parity_error <= 1e-12,
        "matched_total_parameters": abs(
            metrics["mop_parameters"]["mean"] / metrics["dense_parameters"]["mean"] - 1.0
        )
        <= 0.05,
        "staged_top2_beats_dense_mse": sparse_mse < dense_mse,
        "staged_top2_beats_posthoc_pruning": sparse_mse < posthoc_sparse_mse,
        "staged_top2_beats_full_route_mse": sparse_mse < posthoc_all_mse,
        "staged_top2_improves_composition_mse": (
            metrics["staged_top2_composition_mse"]["mean"]
            < metrics["dense_composition_mse"]["mean"]
        ),
        "top2_uses_at_most_60_percent_dense_macs": (
            metrics["mop_top2_macs"]["mean"] / metrics["dense_macs"]["mean"] <= 0.60
        ),
    }
    promote = all(gates.values())
    return {
        "scope": "conditional low-rank operator; not an AGI or language benchmark",
        "seeds": seeds,
        "training_steps_per_seed": steps,
        "task_shape": asdict(shape),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "platform": sys.platform,
        },
        "single_model_invariant": {
            "parameter_owners": 1,
            "routers": 1,
            "losses": 1,
            "optimizers": 1,
            "checkpoints": 1,
            "independent_internal_models": 0,
        },
        "parity": {
            "zero_insert_max_abs_error": parity_error,
            "full_svd_max_abs_error": svd_reconstruction_error(shape, 993),
        },
        "metrics": metrics,
        "nonlinear_negative_control": aggregate(nonlinear_rows),
        "latency": latency,
        "latency_protocol": {
            "model_seed": seeds[-1],
            "batch_size": 512,
            "warmup_repeats": 30,
            "timed_repeats": 300,
            "statistic": "median_microseconds",
        },
        "systems_observations": {
            "top2_active_parameter_fraction": (
                metrics["mop_top2_active_parameters"]["mean"] / metrics["mop_parameters"]["mean"]
            ),
            "top2_active_mac_fraction_vs_dense": (
                metrics["mop_top2_macs"]["mean"] / metrics["dense_macs"]["mean"]
            ),
            "top2_numpy_latency_ratio_vs_dense": (
                latency["mop_top2_batch512_us"] / latency["dense_batch512_us"]
            ),
            "measured_numpy_latency_improved": (
                latency["mop_top2_batch512_us"] < latency["dense_batch512_us"]
            ),
        },
        "promotion_gates": gates,
        "decision": (
            "promote_staged_sparse_operator; do_not_claim_numpy_latency_win"
            if promote
            else "do_not_promote; retain_the_simplest_passing_variant"
        ),
        "ablation_conclusion": (
            "post-hoc pruning failed; dense warmup followed by sparse training passed"
        ),
        "next_experiment": (
            "place the routed update inside one nonlinear sequence block, implement a fused "
            "kernel, and compare on real sequence tasks; recurrence remains disabled until "
            "that gate passes"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    arguments = parser.parse_args()
    if arguments.steps < 1 or not arguments.seeds:
        raise SystemExit("steps and seeds must be positive")
    print(json.dumps(run(arguments.seeds, arguments.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

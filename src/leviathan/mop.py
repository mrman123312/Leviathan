"""A trainable, tensorized Mixture-of-Parameters owned by one model.

This is the smallest executable test of Leviathan's parameter-substrate idea.  The
model computes one conditional weight update and one output:

    y = x W_base + b + sum_e g_e(c) (x A_e) B_e

``A`` and ``B`` are slices of two tensors, not independently callable experts.  They
have no identities, state, objectives, outputs, or optimizer ownership.  One router,
one loss, one gradient update, and one checkpoint own the entire function.

The implementation uses NumPy and explicit reverse-mode derivatives so the benchmark
can run without silently substituting several hosted models or requiring a heavyweight
training framework.  It is a research operator, not yet a language model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MoPConfig:
    input_dim: int
    context_dim: int
    output_dim: int
    basis_count: int
    rank: int
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        dimensions = {
            "input_dim": self.input_dim,
            "context_dim": self.context_dim,
            "output_dim": self.output_dim,
            "basis_count": self.basis_count,
            "rank": self.rank,
        }
        for name, value in dimensions.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if not isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")


@dataclass(frozen=True, slots=True)
class MoPForward:
    output: FloatArray
    gates: FloatArray
    active_basis_indices: NDArray[np.int64]


class UnifiedMoP:
    """One differentiable conditional operator with a tensorized parameter bank."""

    model_id = "unified-mop"

    def __init__(self, config: MoPConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self._parameters: dict[str, FloatArray] = {
            "base_weight": rng.normal(
                0.0,
                1.0 / sqrt(config.input_dim),
                size=(config.input_dim, config.output_dim),
            ),
            "base_bias": np.zeros(config.output_dim, dtype=np.float64),
            "router_weight": rng.normal(
                0.0,
                0.02,
                size=(config.context_dim, config.basis_count),
            ),
            "router_bias": np.zeros(config.basis_count, dtype=np.float64),
            "basis_down": rng.normal(
                0.0,
                1.0 / sqrt(config.input_dim),
                size=(config.basis_count, config.input_dim, config.rank),
            ),
            # A zero final factor makes insertion exactly function-preserving while
            # still allowing gradients to reach it on the first update.
            "basis_up": np.zeros(
                (config.basis_count, config.rank, config.output_dim),
                dtype=np.float64,
            ),
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.size for parameter in self._parameters.values())

    def active_parameter_count(self, active_bases: int | None = None) -> int:
        count = self._active_basis_count(active_bases)
        config = self.config
        always_active = (
            config.input_dim * config.output_dim
            + config.output_dim
            + config.context_dim * config.basis_count
            + config.basis_count
        )
        per_basis = config.input_dim * config.rank + config.rank * config.output_dim
        return always_active + count * per_basis

    def estimated_macs(self, active_bases: int | None = None) -> int:
        """Approximate multiply-accumulates per example, excluding softmax."""

        count = self._active_basis_count(active_bases)
        config = self.config
        return (
            config.input_dim * config.output_dim
            + config.context_dim * config.basis_count
            + count
            * (config.input_dim * config.rank + config.rank * config.output_dim + config.output_dim)
        )

    def state_dict(self) -> dict[str, FloatArray]:
        """Return a defensive copy of the one model parameter state."""

        return {name: value.copy() for name, value in self._parameters.items()}

    def load_state_dict(self, state: Mapping[str, FloatArray]) -> None:
        if set(state) != set(self._parameters):
            raise ValueError("state keys do not match the unified model")
        for name, expected in self._parameters.items():
            supplied = np.asarray(state[name], dtype=np.float64)
            if supplied.shape != expected.shape:
                raise ValueError(f"shape mismatch for {name}: {supplied.shape} != {expected.shape}")
            if not np.all(np.isfinite(supplied)):
                raise ValueError(f"state for {name} contains non-finite values")
            expected[...] = supplied

    def base_prediction(self, inputs: FloatArray) -> FloatArray:
        inputs = self._validate_matrix("inputs", inputs, self.config.input_dim)
        return inputs @ self._parameters["base_weight"] + self._parameters["base_bias"]

    def route(
        self,
        context: FloatArray,
        *,
        active_bases: int | None = None,
    ) -> tuple[FloatArray, NDArray[np.int64]]:
        context = self._validate_matrix("context", context, self.config.context_dim)
        count = self._active_basis_count(active_bases)
        logits = (
            context @ self._parameters["router_weight"] + self._parameters["router_bias"]
        ) / self.config.temperature

        if count < self.config.basis_count:
            selected = np.argpartition(logits, -count, axis=1)[:, -count:]
            masked = np.full_like(logits, -np.inf)
            np.put_along_axis(
                masked, selected, np.take_along_axis(logits, selected, axis=1), axis=1
            )
            logits = masked
        else:
            selected = np.broadcast_to(
                np.arange(self.config.basis_count, dtype=np.int64),
                (context.shape[0], self.config.basis_count),
            ).copy()

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        unnormalized = np.exp(shifted)
        gates = unnormalized / np.sum(unnormalized, axis=1, keepdims=True)
        return gates, selected

    def forward(
        self,
        inputs: FloatArray,
        context: FloatArray,
        *,
        active_bases: int | None = None,
    ) -> MoPForward:
        inputs, context = self._validate_batches(inputs, context)
        gates, selected = self.route(context, active_bases=active_bases)
        if selected.shape[1] == self.config.basis_count:
            hidden = np.einsum("bi,eir->ber", inputs, self._parameters["basis_down"])
            basis_outputs = np.einsum(
                "ber,ero->beo",
                hidden,
                self._parameters["basis_up"],
            )
            update = np.einsum("be,beo->bo", gates, basis_outputs)
        else:
            # Gather first, then compute: sparse inference must not evaluate inactive
            # basis transforms and merely mask their already-paid outputs.
            selected_down = self._parameters["basis_down"][selected]
            selected_up = self._parameters["basis_up"][selected]
            hidden = np.einsum("bi,bkir->bkr", inputs, selected_down)
            basis_outputs = np.einsum("bkr,bkro->bko", hidden, selected_up)
            selected_gates = np.take_along_axis(gates, selected, axis=1)
            update = np.einsum("bk,bko->bo", selected_gates, basis_outputs)
        output = self.base_prediction(inputs) + update
        return MoPForward(output=output, gates=gates, active_basis_indices=selected)

    def loss_and_gradients(
        self,
        inputs: FloatArray,
        context: FloatArray,
        targets: FloatArray,
        *,
        active_bases: int | None = None,
        entropy_penalty: float = 0.0,
    ) -> tuple[float, dict[str, FloatArray]]:
        """Return mean-square loss and exact gradients for all model parameters."""

        if entropy_penalty < 0.0 or not isfinite(entropy_penalty):
            raise ValueError("entropy_penalty must be finite and non-negative")
        inputs, context = self._validate_batches(inputs, context)
        targets = self._validate_matrix("targets", targets, self.config.output_dim)
        if targets.shape[0] != inputs.shape[0]:
            raise ValueError("inputs and targets must have equal batch size")

        parameters = self._parameters
        gates, _ = self.route(context, active_bases=active_bases)
        hidden = np.einsum("bi,eir->ber", inputs, parameters["basis_down"])
        basis_outputs = np.einsum("ber,ero->beo", hidden, parameters["basis_up"])
        predictions = (
            inputs @ parameters["base_weight"]
            + parameters["base_bias"]
            + np.einsum("be,beo->bo", gates, basis_outputs)
        )

        residual = predictions - targets
        mse = float(np.mean(np.square(residual)))
        safe_gates = np.clip(gates, 1e-12, 1.0)
        mean_entropy = float(np.mean(-np.sum(safe_gates * np.log(safe_gates), axis=1)))
        loss = mse + entropy_penalty * mean_entropy

        output_gradient = 2.0 * residual / residual.size
        gradients: dict[str, FloatArray] = {
            "base_weight": inputs.T @ output_gradient,
            "base_bias": np.sum(output_gradient, axis=0),
            "basis_up": np.empty_like(parameters["basis_up"]),
            "basis_down": np.empty_like(parameters["basis_down"]),
            "router_weight": np.empty_like(parameters["router_weight"]),
            "router_bias": np.empty_like(parameters["router_bias"]),
        }

        basis_output_gradient = gates[:, :, None] * output_gradient[:, None, :]
        gradients["basis_up"] = np.einsum("ber,beo->ero", hidden, basis_output_gradient)
        hidden_gradient = np.einsum(
            "beo,ero->ber",
            basis_output_gradient,
            parameters["basis_up"],
        )
        gradients["basis_down"] = np.einsum("bi,ber->eir", inputs, hidden_gradient)

        gate_gradient = np.einsum("bo,beo->be", output_gradient, basis_outputs)
        if entropy_penalty:
            gate_gradient += entropy_penalty * -(np.log(safe_gates) + 1.0) / inputs.shape[0]
        centered = gate_gradient - np.sum(gate_gradient * gates, axis=1, keepdims=True)
        logits_gradient = gates * centered / self.config.temperature
        gradients["router_weight"] = context.T @ logits_gradient
        gradients["router_bias"] = np.sum(logits_gradient, axis=0)

        return loss, gradients

    def save(self, path: str | Path) -> None:
        """Write the complete model to one checkpoint."""

        payload: dict[str, NDArray[np.generic]] = {
            "config_json": np.asarray(json.dumps(asdict(self.config), sort_keys=True)),
            **self._parameters,
        }
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **payload)

    @classmethod
    def load(cls, path: str | Path) -> UnifiedMoP:
        with np.load(Path(path), allow_pickle=False) as archive:
            config = MoPConfig(**json.loads(str(archive["config_json"].item())))
            model = cls(config)
            model.load_state_dict({name: archive[name] for name in model._parameters})
        return model

    def _validate_batches(
        self,
        inputs: FloatArray,
        context: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        inputs = self._validate_matrix("inputs", inputs, self.config.input_dim)
        context = self._validate_matrix("context", context, self.config.context_dim)
        if inputs.shape[0] != context.shape[0]:
            raise ValueError("inputs and context must have equal batch size")
        return inputs, context

    @staticmethod
    def _validate_matrix(name: str, value: FloatArray, width: int) -> FloatArray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"{name} must have shape (batch, {width})")
        if array.shape[0] < 1:
            raise ValueError(f"{name} batch must not be empty")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def _active_basis_count(self, active_bases: int | None) -> int:
        if active_bases is None:
            return self.config.basis_count
        if not 1 <= active_bases <= self.config.basis_count:
            raise ValueError("active_bases must be within the basis reservoir")
        return active_bases


class AdamOptimizer:
    """One optimizer state over the complete unified parameter dictionary."""

    def __init__(
        self,
        model: UnifiedMoP,
        *,
        learning_rate: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_gradient_norm: float | None = 10.0,
    ) -> None:
        if not isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1)")
        if not isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if max_gradient_norm is not None and (
            not isfinite(max_gradient_norm) or max_gradient_norm <= 0.0
        ):
            raise ValueError("max_gradient_norm must be positive when supplied")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_gradient_norm = max_gradient_norm
        self._model = model
        self._step = 0
        self._first = {name: np.zeros_like(value) for name, value in model._parameters.items()}
        self._second = {name: np.zeros_like(value) for name, value in model._parameters.items()}

    @property
    def step_count(self) -> int:
        return self._step

    def step(self, model: UnifiedMoP, gradients: Mapping[str, FloatArray]) -> None:
        if model is not self._model:
            raise ValueError("this optimizer is bound to a different unified model")
        if set(gradients) != set(model._parameters):
            raise ValueError("gradient keys do not match the unified model")
        checked = {
            name: np.asarray(gradients[name], dtype=np.float64) for name in model._parameters
        }
        for name, gradient in checked.items():
            if gradient.shape != model._parameters[name].shape:
                raise ValueError(f"gradient shape mismatch for {name}")
            if not np.all(np.isfinite(gradient)):
                raise ValueError(f"gradient for {name} contains non-finite values")

        if self.max_gradient_norm is not None:
            norm = sqrt(sum(float(np.sum(np.square(value))) for value in checked.values()))
            if norm > self.max_gradient_norm:
                scale = self.max_gradient_norm / norm
                checked = {name: value * scale for name, value in checked.items()}

        self._step += 1
        correction1 = 1.0 - self.beta1**self._step
        correction2 = 1.0 - self.beta2**self._step
        for name, parameter in model._parameters.items():
            gradient = checked[name]
            self._first[name] = self.beta1 * self._first[name] + (1.0 - self.beta1) * gradient
            self._second[name] = self.beta2 * self._second[name] + (1.0 - self.beta2) * np.square(
                gradient
            )
            first_hat = self._first[name] / correction1
            second_hat = self._second[name] / correction2
            parameter -= self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)

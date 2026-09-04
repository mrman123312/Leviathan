"""Mixture-of-Parameterized-Cells reference architecture.

This module grows Leviathan's exact MoP-0 parameter tiles into richer computational
cells without changing the inherited DeepSeek function at insertion.

The fundamental invariant is:

    output = donor_output                         when every new gate == 0

The live reference path can additionally compute:

* arbitrary cross-expert cell routes,
* sparse peer communication among cells assigned to the same token,
* disagreement-triggered associative recruitment,
* bounded ephemeral local cell state,
* low-rank learned refinements around inherited tile bodies.

Every behavioral bridge is independently zero-gated at insertion. This keeps one
global model, one parameter ownership system and one final output. Cells are pieces
of a single neural network, never independent language-model agents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
import math
from pathlib import Path
import tomllib
from typing import Any, Iterable

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[2] / "spec" / "parameter-cells.toml"
_BaseModule = nn.Module if nn is not None else object


class MoPStage(IntEnum):
    EXACT_TILES = 0
    INDEPENDENT_TILE_ROUTING = 1
    CONFIDENCE_AND_ABSTENTION = 2
    PROPOSAL_MESSAGES = 3
    ONE_COMMUNICATION_ROUND = 4
    DISAGREEMENT_RECRUITMENT = 5
    LOCAL_STATE = 6
    LEARNED_COALITIONS = 7
    LOCAL_PLASTICITY = 8
    CELL_LIFECYCLE = 9


class CellAction(str, Enum):
    COMMIT = "commit"
    COMMUNICATE = "communicate"
    RECRUIT = "recruit"


@dataclass(frozen=True, slots=True)
class CellIdentity:
    layer_index: int
    expert_index: int
    tile_index: int
    tiles_per_expert: int

    def __post_init__(self) -> None:
        if self.layer_index < 0 or self.expert_index < 0 or self.tile_index < 0:
            raise ValueError("cell indices must be non-negative")
        if self.tiles_per_expert <= 0:
            raise ValueError("tiles_per_expert must be positive")
        if self.tile_index >= self.tiles_per_expert:
            raise ValueError("tile_index exceeds tiles_per_expert")

    @property
    def local_cell_id(self) -> int:
        return self.expert_index * self.tiles_per_expert + self.tile_index

    @property
    def stable_key(self) -> str:
        return f"L{self.layer_index}:E{self.expert_index}:T{self.tile_index}"


@dataclass(frozen=True, slots=True)
class CellBudget:
    seed_cells: int = 64
    recruited_cells_per_round: int = 32
    max_active_cells: int = 256
    max_rounds: int = 2
    max_neighbors: int = 8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seed_cells > self.max_active_cells:
            raise ValueError("seed_cells cannot exceed max_active_cells")
        if self.recruited_cells_per_round > self.max_active_cells:
            raise ValueError("recruited_cells_per_round cannot exceed max_active_cells")


@dataclass(frozen=True, slots=True)
class DisagreementThresholds:
    communicate: float = 0.10
    recruit: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 <= self.communicate < self.recruit:
            raise ValueError("disagreement thresholds must satisfy 0 <= communicate < recruit")

    def action(self, disagreement: float) -> CellAction:
        if disagreement < 0:
            raise ValueError("disagreement must be non-negative")
        if disagreement < self.communicate:
            return CellAction.COMMIT
        if disagreement < self.recruit:
            return CellAction.COMMUNICATE
        return CellAction.RECRUIT


@dataclass(frozen=True, slots=True)
class CellMembraneConfig:
    control_dim: int = 32
    message_dim: int = 32
    recruitment_dim: int = 32
    confidence_dims: int = 4
    local_state_dim: int = 64
    low_rank: int = 16
    initial_influence: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.control_dim,
            self.message_dim,
            self.recruitment_dim,
            self.confidence_dims,
            self.local_state_dim,
            self.low_rank,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("cell membrane dimensions must be positive")
        if not 0.0 <= self.initial_influence <= 1.0:
            raise ValueError("initial_influence must be in [0, 1]")
        if self.initial_influence != 0.0:
            raise ValueError("new cell membranes must be exactly inert at insertion")


@dataclass(frozen=True, slots=True)
class CellExecutionConfig:
    """Reference execution controls; enabling code paths is not maturity promotion."""

    stage: MoPStage = MoPStage.EXACT_TILES
    independent_top_k: int = 144
    state_update_rate: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", MoPStage(int(self.stage)))
        if self.independent_top_k <= 0:
            raise ValueError("independent_top_k must be positive")
        if not 0.0 < self.state_update_rate <= 1.0:
            raise ValueError("state_update_rate must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class CellTelemetrySummary:
    active_cell_token_pairs: int
    unique_cells_seen: int
    mean_confidence: float
    mean_abstention: float
    mean_disagreement: float
    max_disagreement: float
    recommended_action: CellAction
    communication_rounds: int = 0
    recruited_cell_token_pairs: int = 0
    unique_recruited_cells: int = 0
    local_state_updates: int = 0
    independent_route_cell_token_pairs: int = 0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recommended_action"] = self.recommended_action.value
        return payload


@dataclass(frozen=True, slots=True)
class CellPatchReport:
    moe_modules: int
    wrapped_experts: int
    already_wrapped: int
    tile_width: int
    control_parameters: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class CellCoalition:
    cell_ids: tuple[int, ...]
    verified_successes: int = 0
    verified_failures: int = 0
    compiled: bool = False

    def __post_init__(self) -> None:
        canonical = tuple(sorted(set(int(cell) for cell in self.cell_ids)))
        if not canonical:
            raise ValueError("coalition must contain at least one cell")
        self.cell_ids = canonical

    @property
    def verified_trials(self) -> int:
        return self.verified_successes + self.verified_failures

    @property
    def success_rate(self) -> float:
        if not self.verified_trials:
            return 0.0
        return self.verified_successes / self.verified_trials

    def record(self, *, verified_success: bool) -> None:
        if verified_success:
            self.verified_successes += 1
        else:
            self.verified_failures += 1

    def may_compile(self, *, min_trials: int = 8, min_success_rate: float = 0.90) -> bool:
        return self.verified_trials >= min_trials and self.success_rate >= min_success_rate


class CoalitionRegistry:
    """Tracks repeated verified cell groups without silently promoting them."""

    def __init__(self) -> None:
        self._coalitions: dict[tuple[int, ...], CellCoalition] = {}

    def record(self, cell_ids: Iterable[int], *, verified_success: bool) -> CellCoalition:
        key = tuple(sorted(set(int(cell) for cell in cell_ids)))
        if not key:
            raise ValueError("cannot record an empty coalition")
        coalition = self._coalitions.setdefault(key, CellCoalition(key))
        coalition.record(verified_success=verified_success)
        return coalition

    def candidates(
        self,
        *,
        min_trials: int = 8,
        min_success_rate: float = 0.90,
    ) -> tuple[CellCoalition, ...]:
        return tuple(
            coalition
            for coalition in self._coalitions.values()
            if coalition.may_compile(
                min_trials=min_trials,
                min_success_rate=min_success_rate,
            )
        )

    def mark_compiled(self, cell_ids: Iterable[int]) -> None:
        key = tuple(sorted(set(int(cell) for cell in cell_ids)))
        if key not in self._coalitions:
            raise KeyError(key)
        self._coalitions[key].compiled = True


def load_parameter_cell_spec(path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def stage_sequence(path: Path = DEFAULT_SPEC_PATH) -> tuple[MoPStage, ...]:
    raw = load_parameter_cell_spec(path)
    values = raw["roadmap"]["stages"]
    parsed = tuple(MoPStage(int(value)) for value in values)
    if parsed != tuple(MoPStage):
        raise ValueError("parameter-cell stage roadmap must contain MoP-0 through MoP-9 in order")
    return parsed


def _phase_name(phase: Any) -> str:
    value = getattr(phase, "value", phase)
    return str(value)


def _gate_phase_allows_influence(phase: Any) -> bool:
    return _phase_name(phase) in {
        "gate_warmup",
        "selective_unfreeze",
        "continued_pretraining_and_agentic_posttraining",
        "retention_calibration_safety_evaluation",
        "shadow_evaluation",
        "candidate_promotion",
    }


if nn is not None:
    @dataclass(slots=True)
    class CellSignals:
        confidence: Any
        abstention: Any
        message: Any
        recruitment_query: Any
        refinement: Any
        control: Any


    class CellControlMembrane(nn.Module):
        """Cheap expressive membrane around inherited parameter tiles.

        Communication and local state are inputs to the same cell membrane rather
        than external agents. Their influence gates start at zero independently of
        the residual refinement gate.
        """

        def __init__(
            self,
            *,
            hidden_size: int,
            num_cells: int,
            config: CellMembraneConfig = CellMembraneConfig(),
        ) -> None:
            super().__init__()
            if hidden_size <= 0 or num_cells <= 0:
                raise ValueError("hidden_size and num_cells must be positive")
            self.hidden_size = int(hidden_size)
            self.num_cells = int(num_cells)
            self.config = config

            self.state_down = nn.Linear(hidden_size, config.control_dim, bias=False)
            self.cell_embedding = nn.Embedding(num_cells, config.control_dim)
            self.message_control = nn.Linear(config.message_dim, config.control_dim, bias=False)
            self.local_state_control = nn.Linear(
                config.local_state_dim, config.control_dim, bias=False
            )
            self.confidence_head = nn.Linear(config.control_dim, config.confidence_dims)
            self.abstention_head = nn.Linear(config.control_dim, 1)
            self.message_head = nn.Linear(config.control_dim, config.message_dim, bias=False)
            self.recruitment_head = nn.Linear(
                config.control_dim,
                config.recruitment_dim,
                bias=False,
            )
            self.refine_down = nn.Linear(hidden_size, config.low_rank, bias=False)
            self.refine_control = nn.Linear(config.control_dim, config.low_rank, bias=False)
            self.refine_up = nn.Linear(config.low_rank, hidden_size, bias=False)
            self.local_state_update = nn.GRUCell(
                config.control_dim + config.message_dim,
                config.local_state_dim,
            )

            self.influence_gate = nn.Parameter(torch.tensor(float(config.initial_influence)))
            self.communication_gate = nn.Parameter(torch.tensor(0.0))
            self.state_gate = nn.Parameter(torch.tensor(0.0))

            nn.init.normal_(self.cell_embedding.weight, std=0.01)
            nn.init.normal_(self.refine_up.weight, std=0.01)

        @staticmethod
        def _bounded_gate(parameter: Any) -> Any:
            return parameter.clamp(0.0, 1.0)

        @property
        def influence(self) -> Any:
            return self._bounded_gate(self.influence_gate)

        @property
        def communication_influence(self) -> Any:
            return self._bounded_gate(self.communication_gate)

        @property
        def state_influence(self) -> Any:
            return self._bounded_gate(self.state_gate)

        @staticmethod
        def _set_gate(parameter: Any, value: float, *, transplant_phase: Any, label: str) -> None:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} influence must be in [0, 1]")
            if value != 0.0 and not _gate_phase_allows_influence(transplant_phase):
                raise RuntimeError(f"{label} cannot affect output before gate warmup")
            with torch.no_grad():
                parameter.fill_(float(value))

        def set_influence(self, value: float, *, transplant_phase: Any) -> None:
            self._set_gate(
                self.influence_gate,
                value,
                transplant_phase=transplant_phase,
                label="cell membrane",
            )

        def set_communication_influence(self, value: float, *, transplant_phase: Any) -> None:
            self._set_gate(
                self.communication_gate,
                value,
                transplant_phase=transplant_phase,
                label="cell communication",
            )

        def set_state_influence(self, value: float, *, transplant_phase: Any) -> None:
            self._set_gate(
                self.state_gate,
                value,
                transplant_phase=transplant_phase,
                label="cell local state",
            )

        def forward(
            self,
            hidden_states: Any,
            cell_ids: Any,
            *,
            received_message: Any | None = None,
            local_state: Any | None = None,
        ) -> CellSignals:
            if hidden_states.ndim != 2:
                raise ValueError("hidden_states must be [assignments, hidden]")
            if cell_ids.ndim != 1 or cell_ids.shape[0] != hidden_states.shape[0]:
                raise ValueError("cell_ids must align with hidden_states")
            if cell_ids.numel() and (
                int(cell_ids.min().item()) < 0 or int(cell_ids.max().item()) >= self.num_cells
            ):
                raise IndexError("cell id out of range")

            control_pre = self.state_down(hidden_states.float()) + self.cell_embedding(cell_ids)
            if received_message is not None:
                if received_message.shape != (
                    hidden_states.shape[0], self.config.message_dim
                ):
                    raise ValueError("received_message shape mismatch")
                control_pre = control_pre + (
                    self.communication_influence
                    * self.message_control(received_message.float())
                )
            if local_state is not None:
                if local_state.shape != (
                    hidden_states.shape[0], self.config.local_state_dim
                ):
                    raise ValueError("local_state shape mismatch")
                control_pre = control_pre + (
                    self.state_influence * self.local_state_control(local_state.float())
                )

            control = torch.tanh(control_pre)
            confidence = torch.sigmoid(self.confidence_head(control))
            abstention = torch.sigmoid(self.abstention_head(control)).squeeze(-1)
            message = self.message_head(control)
            recruitment_query = self.recruitment_head(control)
            refinement = self.refine_up(
                torch.tanh(
                    self.refine_down(hidden_states.float()) + self.refine_control(control)
                )
            )
            return CellSignals(
                confidence=confidence,
                abstention=abstention,
                message=message,
                recruitment_query=recruitment_query,
                refinement=refinement,
                control=control,
            )

        def propose_state_update(
            self,
            prior_state: Any,
            signals: CellSignals,
            received_message: Any | None = None,
        ) -> Any:
            if received_message is None:
                received_message = torch.zeros(
                    signals.control.shape[0],
                    self.config.message_dim,
                    device=signals.control.device,
                    dtype=signals.control.dtype,
                )
            update_input = torch.cat(
                [signals.control.float(), received_message.float()], dim=-1
            )
            return self.local_state_update(update_input, prior_state.float())


    class SparseCellCommunication(nn.Module):
        """One bounded communication round among active messages in the same token group."""

        def __init__(self, message_dim: int, *, max_neighbors: int = 8) -> None:
            super().__init__()
            if message_dim <= 0 or max_neighbors <= 0:
                raise ValueError("message_dim and max_neighbors must be positive")
            self.message_dim = int(message_dim)
            self.max_neighbors = int(max_neighbors)
            self.q = nn.Linear(message_dim, message_dim, bias=False)
            self.k = nn.Linear(message_dim, message_dim, bias=False)
            self.v = nn.Linear(message_dim, message_dim, bias=False)
            self.out = nn.Linear(message_dim, message_dim, bias=False)

        def forward(self, messages: Any, group_ids: Any) -> Any:
            if messages.ndim != 2:
                raise ValueError("messages must be [assignments, message_dim]")
            if group_ids.ndim != 1 or group_ids.shape[0] != messages.shape[0]:
                raise ValueError("group_ids must align with messages")
            result = torch.zeros_like(messages)
            scale = 1.0 / math.sqrt(self.message_dim)

            for group in torch.unique(group_ids):
                idx = torch.where(group_ids == group)[0]
                local = messages[idx]
                q = self.q(local)
                k = self.k(local)
                v = self.v(local)
                scores = q @ k.transpose(0, 1) * scale
                neighbors = min(self.max_neighbors, int(local.shape[0]))
                top_values, top_indices = torch.topk(scores, k=neighbors, dim=-1)
                weights = torch.softmax(top_values, dim=-1)
                gathered = v[top_indices]
                attended = (weights.unsqueeze(-1) * gathered).sum(dim=-2)
                result[idx] = self.out(attended)
            return result


    class AssociativeCellRecruiter(nn.Module):
        """Content-addressed routing/recruitment over a learned cell-key table."""

        def __init__(self, num_cells: int, key_dim: int) -> None:
            super().__init__()
            if num_cells <= 0 or key_dim <= 0:
                raise ValueError("num_cells and key_dim must be positive")
            self.num_cells = int(num_cells)
            self.key_dim = int(key_dim)
            self.keys = nn.Parameter(torch.randn(num_cells, key_dim) * 0.01)

        def forward(
            self,
            queries: Any,
            *,
            k: int,
            excluded_cell_ids: Any | None = None,
        ) -> tuple[Any, Any]:
            if queries.ndim != 2 or queries.shape[-1] != self.key_dim:
                raise ValueError("queries must be [n, key_dim]")
            if k <= 0:
                raise ValueError("k must be positive")
            k = min(k, self.num_cells)
            scores = F.normalize(queries.float(), dim=-1) @ F.normalize(
                self.keys.float(), dim=-1
            ).transpose(0, 1)
            if excluded_cell_ids is not None and excluded_cell_ids.numel():
                scores[:, excluded_cell_ids.long()] = float("-inf")
            values, indices = torch.topk(scores, k=k, dim=-1)
            return indices, values


    class CellizedPackedExpertsWrapper(nn.Module):
        """DeepSeek packed experts plus a live, zero-gated parameter ecology.

        The donor expert route remains the exact output path at gate value zero. The
        reference stage can nevertheless execute peer communication, independent
        cross-expert cell routing, recruitment and ephemeral local-state updates so
        those mechanisms can be tested before they are allowed to control behavior.
        """

        def __init__(
            self,
            expert_bank: Any,
            *,
            tile_width: int = 128,
            membrane_config: CellMembraneConfig = CellMembraneConfig(),
            disagreement: DisagreementThresholds = DisagreementThresholds(),
            collect_telemetry: bool = True,
            execution: CellExecutionConfig = CellExecutionConfig(),
            budget: CellBudget = CellBudget(),
        ) -> None:
            super().__init__()
            if tile_width <= 0:
                raise ValueError("tile_width must be positive")
            for name in ("gate_up_proj", "down_proj"):
                if not hasattr(expert_bank, name):
                    raise TypeError(f"packed expert bank is missing {name}")
            self.expert_bank = expert_bank
            self.tile_width = int(tile_width)
            self.disagreement_thresholds = disagreement
            self.collect_telemetry = bool(collect_telemetry)
            self.execution = execution
            self.budget = budget
            self.last_telemetry: CellTelemetrySummary | None = None
            self.last_recruited_cell_ids: tuple[int, ...] = ()
            self.last_independent_cell_ids: tuple[int, ...] = ()

            if self.intermediate_size % self.tile_width:
                raise ValueError("tile_width must divide packed expert intermediate size")

            self.membrane = CellControlMembrane(
                hidden_size=self.hidden_size,
                num_cells=self.num_cells,
                config=membrane_config,
            )
            self.communication = SparseCellCommunication(
                membrane_config.message_dim,
                max_neighbors=budget.max_neighbors,
            )
            self.recruiter = AssociativeCellRecruiter(
                self.num_cells,
                membrane_config.recruitment_dim,
            )
            self.independent_route_query = nn.Linear(
                self.hidden_size,
                membrane_config.recruitment_dim,
                bias=False,
            )
            self.independent_route_gate = nn.Parameter(torch.tensor(0.0))
            self.recruitment_gate = nn.Parameter(torch.tensor(0.0))
            self.register_buffer(
                "local_state",
                torch.zeros(self.num_cells, membrane_config.local_state_dim),
                persistent=False,
            )

        @property
        def num_experts(self) -> int:
            declared = getattr(self.expert_bank, "num_experts", None)
            if declared is not None:
                return int(declared)
            return int(self.expert_bank.gate_up_proj.shape[0])

        @property
        def hidden_size(self) -> int:
            declared = getattr(self.expert_bank, "hidden_dim", None)
            if declared is not None:
                return int(declared)
            return int(self.expert_bank.down_proj.shape[1])

        @property
        def intermediate_size(self) -> int:
            declared = getattr(self.expert_bank, "intermediate_dim", None)
            if declared is not None:
                return int(declared)
            return int(self.expert_bank.down_proj.shape[-1])

        @property
        def tiles_per_expert(self) -> int:
            return self.intermediate_size // self.tile_width

        @property
        def num_cells(self) -> int:
            return self.num_experts * self.tiles_per_expert

        @property
        def independent_route_influence(self) -> Any:
            return self.independent_route_gate.clamp(0.0, 1.0)

        @property
        def recruitment_influence(self) -> Any:
            return self.recruitment_gate.clamp(0.0, 1.0)

        @property
        def control_parameter_count(self) -> int:
            modules = (
                self.membrane,
                self.communication,
                self.recruiter,
                self.independent_route_query,
            )
            total = sum(parameter.numel() for module in modules for parameter in module.parameters())
            total += self.independent_route_gate.numel() + self.recruitment_gate.numel()
            return int(total)

        def cell_id(self, expert_index: int, tile_index: int) -> int:
            if not 0 <= expert_index < self.num_experts:
                raise IndexError("expert index out of range")
            if not 0 <= tile_index < self.tiles_per_expert:
                raise IndexError("tile index out of range")
            return expert_index * self.tiles_per_expert + tile_index

        def cell_coordinates(self, cell_id: int) -> tuple[int, int]:
            if not 0 <= int(cell_id) < self.num_cells:
                raise IndexError("cell id out of range")
            return divmod(int(cell_id), self.tiles_per_expert)

        def _activate(self, gate_up: Any) -> Any:
            apply_gate = getattr(self.expert_bank, "_apply_gate", None)
            if callable(apply_gate):
                return apply_gate(gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            return self._activate_pair(gate, up)

        def _activate_pair(self, gate: Any, up: Any) -> Any:
            limit = float(getattr(self.expert_bank, "limit", 0.0) or 0.0)
            if limit > 0:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            act_fn = getattr(self.expert_bank, "act_fn", F.silu)
            return act_fn(gate) * up

        @staticmethod
        def _set_scalar_gate(
            parameter: Any,
            value: float,
            *,
            transplant_phase: Any,
            label: str,
        ) -> None:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} influence must be in [0, 1]")
            if value != 0.0 and not _gate_phase_allows_influence(transplant_phase):
                raise RuntimeError(f"{label} cannot affect output before gate warmup")
            with torch.no_grad():
                parameter.fill_(float(value))

        def set_influence(self, value: float, *, transplant_phase: Any) -> None:
            self.membrane.set_influence(value, transplant_phase=transplant_phase)

        def set_communication_influence(self, value: float, *, transplant_phase: Any) -> None:
            if value != 0.0 and self.execution.stage < MoPStage.ONE_COMMUNICATION_ROUND:
                raise RuntimeError("communication influence requires MoP-4 reference stage")
            self.membrane.set_communication_influence(
                value, transplant_phase=transplant_phase
            )

        def set_state_influence(self, value: float, *, transplant_phase: Any) -> None:
            if value != 0.0 and self.execution.stage < MoPStage.LOCAL_STATE:
                raise RuntimeError("local-state influence requires MoP-6 reference stage")
            self.membrane.set_state_influence(value, transplant_phase=transplant_phase)

        def set_recruitment_influence(self, value: float, *, transplant_phase: Any) -> None:
            if value != 0.0 and self.execution.stage < MoPStage.DISAGREEMENT_RECRUITMENT:
                raise RuntimeError("recruitment influence requires MoP-5 reference stage")
            self._set_scalar_gate(
                self.recruitment_gate,
                value,
                transplant_phase=transplant_phase,
                label="cell recruitment",
            )

        def set_independent_route_influence(self, value: float, *, transplant_phase: Any) -> None:
            if value != 0.0 and self.execution.stage < MoPStage.INDEPENDENT_TILE_ROUTING:
                raise RuntimeError("independent routing requires MoP-1 reference stage")
            self._set_scalar_gate(
                self.independent_route_gate,
                value,
                transplant_phase=transplant_phase,
                label="independent cell routing",
            )

        def set_reference_stage(self, stage: MoPStage | int) -> None:
            """Enable a reference code path without claiming the maturity gate passed."""
            self.execution = CellExecutionConfig(
                stage=MoPStage(int(stage)),
                independent_top_k=self.execution.independent_top_k,
                state_update_rate=self.execution.state_update_rate,
            )

        def reset_local_state(self) -> None:
            with torch.no_grad():
                self.local_state.zero_()

        def _cell_body(self, hidden_states: Any, cell_ids: Any) -> Any:
            """Execute the actual ancestral SwiGLU tile for arbitrary cross-expert cells."""
            if hidden_states.ndim != 2 or cell_ids.ndim != 1:
                raise ValueError("cell-body inputs must be [n, hidden] and [n]")
            if hidden_states.shape[0] != cell_ids.shape[0]:
                raise ValueError("cell-body ids must align with hidden states")
            output = torch.zeros(
                hidden_states.shape[0],
                self.hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            for cell_tensor in torch.unique(cell_ids):
                cid = int(cell_tensor.item())
                expert_index, tile_index = self.cell_coordinates(cid)
                rows = torch.where(cell_ids == cid)[0]
                start = tile_index * self.tile_width
                stop = start + self.tile_width
                local_hidden = hidden_states[rows]
                gate = F.linear(
                    local_hidden,
                    self.expert_bank.gate_up_proj[expert_index, start:stop],
                )
                up = F.linear(
                    local_hidden,
                    self.expert_bank.gate_up_proj[
                        expert_index,
                        self.intermediate_size + start : self.intermediate_size + stop,
                    ],
                )
                activated = self._activate_pair(gate, up)
                local_output = F.linear(
                    activated,
                    self.expert_bank.down_proj[expert_index, :, start:stop],
                )
                output[rows] = local_output.to(output.dtype)
            return output

        @staticmethod
        def _message_stats(messages: Any, group_ids: Any, token_count: int) -> tuple[Any, Any]:
            dim = int(messages.shape[-1])
            sums = torch.zeros(
                token_count, dim, dtype=torch.float32, device=messages.device
            )
            sq_sums = torch.zeros_like(sums)
            counts = torch.zeros(token_count, dtype=torch.float32, device=messages.device)
            sums.index_add_(0, group_ids, messages.float())
            sq_sums.index_add_(0, group_ids, messages.float().square())
            counts.index_add_(0, group_ids, torch.ones_like(group_ids, dtype=torch.float32))
            safe = counts.clamp_min(1.0)
            means = sums / safe[:, None]
            variance = (sq_sums / safe[:, None] - means.square()).clamp_min(0.0)
            return variance.mean(dim=-1), counts

        def _gather_local_state(self, cell_ids: Any) -> Any:
            return self.local_state[cell_ids.long()].to(dtype=torch.float32)

        def _update_local_state(
            self,
            cell_ids: Any,
            signals: CellSignals,
            received_message: Any,
        ) -> int:
            if not cell_ids.numel():
                return 0
            prior = self._gather_local_state(cell_ids)
            proposed = self.membrane.propose_state_update(
                prior,
                signals,
                received_message,
            ).detach()
            updated = 0
            rate = self.execution.state_update_rate
            with torch.no_grad():
                for cell_tensor in torch.unique(cell_ids):
                    cid = int(cell_tensor.item())
                    rows = torch.where(cell_ids == cid)[0]
                    mean_proposed = proposed[rows].mean(dim=0).to(self.local_state.dtype)
                    self.local_state[cid].mul_(1.0 - rate).add_(mean_proposed, alpha=rate)
                    updated += 1
            return updated

        def _run_independent_route(self, hidden_states: Any) -> tuple[Any, int]:
            top_k = min(self.execution.independent_top_k, self.num_cells)
            queries = self.independent_route_query(hidden_states.float())
            ids, scores = self.recruiter(queries, k=top_k)
            self.last_independent_cell_ids = tuple(
                sorted({int(value) for value in ids.detach().flatten().tolist()})
            )
            flat_ids = ids.reshape(-1)
            repeated_hidden = hidden_states.repeat_interleave(top_k, dim=0)
            cell_outputs = self._cell_body(repeated_hidden, flat_ids).reshape(
                hidden_states.shape[0], top_k, self.hidden_size
            )
            weights = torch.softmax(scores.float(), dim=-1).to(cell_outputs.dtype)
            routed = (cell_outputs * weights[..., None]).sum(dim=1)
            return routed, int(flat_ids.numel())

        def forward(self, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
            donor_final = torch.zeros_like(hidden_states)
            token_count = int(hidden_states.shape[0])
            assignment_hidden: list[Any] = []
            assignment_cell_ids: list[Any] = []
            assignment_token_ids: list[Any] = []
            assignment_route_weights: list[Any] = []

            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

            # Preserve the inherited expert arithmetic and accumulation order exactly.
            for expert_tensor in hit:
                expert_index = int(expert_tensor[0].item())
                if expert_index >= self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(mask[expert_index])
                route_weight = top_k_weights[token_idx, top_k_pos]
                gate_up = F.linear(
                    hidden_states[token_idx], self.expert_bank.gate_up_proj[expert_index]
                )
                activated = self._activate(gate_up)
                expert_output = torch.zeros(
                    (token_idx.shape[0], self.hidden_size),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )

                for tile_index, start in enumerate(
                    range(0, self.intermediate_size, self.tile_width)
                ):
                    stop = start + self.tile_width
                    contribution = F.linear(
                        activated[..., start:stop],
                        self.expert_bank.down_proj[expert_index, :, start:stop],
                    )
                    expert_output = expert_output + contribution.to(expert_output.dtype)
                    cid = self.cell_id(expert_index, tile_index)
                    assignment_hidden.append(hidden_states[token_idx])
                    assignment_cell_ids.append(
                        torch.full(
                            (token_idx.shape[0],),
                            cid,
                            dtype=torch.long,
                            device=hidden_states.device,
                        )
                    )
                    assignment_token_ids.append(token_idx)
                    assignment_route_weights.append(route_weight)

                expert_output = expert_output * route_weight[:, None]
                donor_final.index_add_(0, token_idx, expert_output.to(donor_final.dtype))

            if not assignment_hidden:
                self.last_telemetry = CellTelemetrySummary(
                    active_cell_token_pairs=0,
                    unique_cells_seen=0,
                    mean_confidence=0.0,
                    mean_abstention=0.0,
                    mean_disagreement=0.0,
                    max_disagreement=0.0,
                    recommended_action=CellAction.COMMIT,
                )
                return donor_final

            active_hidden = torch.cat(assignment_hidden, dim=0)
            active_cell_ids = torch.cat(assignment_cell_ids, dim=0)
            active_token_ids = torch.cat(assignment_token_ids, dim=0)
            active_route_weights = torch.cat(assignment_route_weights, dim=0)
            state = None
            if self.execution.stage >= MoPStage.LOCAL_STATE:
                state = self._gather_local_state(active_cell_ids)

            signals = self.membrane(active_hidden, active_cell_ids, local_state=state)
            received = torch.zeros_like(signals.message)
            communication_rounds = 0

            if self.execution.stage >= MoPStage.ONE_COMMUNICATION_ROUND:
                received = self.communication(signals.message, active_token_ids)
                signals = self.membrane(
                    active_hidden,
                    active_cell_ids,
                    received_message=received,
                    local_state=state,
                )
                communication_rounds = 1

            # New learned refinements remain a zero-gated residual around donor tiles.
            if float(self.membrane.influence.detach().item()) != 0.0:
                refinement = (
                    self.membrane.influence.to(signals.refinement.dtype)
                    * signals.refinement
                    * active_route_weights[:, None].to(signals.refinement.dtype)
                )
                donor_final.index_add_(
                    0,
                    active_token_ids,
                    refinement.to(donor_final.dtype),
                )

            per_token_disagreement, active_counts = self._message_stats(
                signals.message,
                active_token_ids,
                token_count,
            )

            recruited_ids_all: list[Any] = []
            recruited_messages_all: list[Any] = []
            recruited_confidence_all: list[Any] = []
            recruited_abstention_all: list[Any] = []
            recruited_token_ids_all: list[Any] = []
            recruited_received_all: list[Any] = []
            recruited_signals_all: list[CellSignals] = []
            recruited_pairs = 0

            if self.execution.stage >= MoPStage.DISAGREEMENT_RECRUITMENT:
                recruit_tokens = torch.where(
                    per_token_disagreement >= self.disagreement_thresholds.recruit
                )[0]
                for token_tensor in recruit_tokens:
                    token_id = int(token_tensor.item())
                    local_rows = torch.where(active_token_ids == token_id)[0]
                    current_ids = torch.unique(active_cell_ids[local_rows])
                    capacity = min(
                        self.budget.recruited_cells_per_round,
                        self.budget.max_active_cells - int(current_ids.numel()),
                        self.num_cells - int(current_ids.numel()),
                    )
                    if capacity <= 0:
                        continue
                    query = signals.recruitment_query[local_rows].mean(dim=0, keepdim=True)
                    rec_ids_2d, rec_scores_2d = self.recruiter(
                        query,
                        k=capacity,
                        excluded_cell_ids=current_ids,
                    )
                    rec_ids = rec_ids_2d[0]
                    rec_scores = rec_scores_2d[0]
                    rec_hidden = hidden_states[token_id : token_id + 1].expand(capacity, -1)
                    rec_state = None
                    if self.execution.stage >= MoPStage.LOCAL_STATE:
                        rec_state = self._gather_local_state(rec_ids)
                    rec_signals = self.membrane(rec_hidden, rec_ids, local_state=rec_state)
                    rec_received = torch.zeros_like(rec_signals.message)

                    # A second bounded round lets newly recruited cells actually enter
                    # the same token-local discussion before any new residual is formed.
                    if self.budget.max_rounds >= 2:
                        combined_hidden = torch.cat([active_hidden[local_rows], rec_hidden], dim=0)
                        combined_ids = torch.cat([active_cell_ids[local_rows], rec_ids], dim=0)
                        combined_messages = torch.cat(
                            [signals.message[local_rows], rec_signals.message], dim=0
                        )
                        combined_groups = torch.zeros(
                            combined_messages.shape[0],
                            dtype=torch.long,
                            device=combined_messages.device,
                        )
                        combined_received = self.communication(combined_messages, combined_groups)
                        combined_state = None
                        if self.execution.stage >= MoPStage.LOCAL_STATE:
                            combined_state = self._gather_local_state(combined_ids)
                        combined_signals = self.membrane(
                            combined_hidden,
                            combined_ids,
                            received_message=combined_received,
                            local_state=combined_state,
                        )
                        n_active = int(local_rows.numel())
                        # Replace the original cells' proposals with post-recruitment
                        # proposals so the discussion is genuinely bidirectional.
                        signals.confidence[local_rows] = combined_signals.confidence[:n_active]
                        signals.abstention[local_rows] = combined_signals.abstention[:n_active]
                        signals.message[local_rows] = combined_signals.message[:n_active]
                        signals.recruitment_query[local_rows] = (
                            combined_signals.recruitment_query[:n_active]
                        )
                        signals.refinement[local_rows] = combined_signals.refinement[:n_active]
                        signals.control[local_rows] = combined_signals.control[:n_active]
                        received[local_rows] = combined_received[:n_active]
                        rec_signals = CellSignals(
                            confidence=combined_signals.confidence[n_active:],
                            abstention=combined_signals.abstention[n_active:],
                            message=combined_signals.message[n_active:],
                            recruitment_query=combined_signals.recruitment_query[n_active:],
                            refinement=combined_signals.refinement[n_active:],
                            control=combined_signals.control[n_active:],
                        )
                        rec_received = combined_received[n_active:]
                        communication_rounds = max(communication_rounds, 2)

                    rec_body = self._cell_body(rec_hidden, rec_ids)
                    rec_output = rec_body
                    if float(self.membrane.influence.detach().item()) != 0.0:
                        rec_output = rec_output + (
                            self.membrane.influence.to(rec_output.dtype)
                            * rec_signals.refinement.to(rec_output.dtype)
                        )
                    if float(self.recruitment_influence.detach().item()) != 0.0:
                        rec_weights = torch.softmax(rec_scores.float(), dim=-1).to(rec_output.dtype)
                        rec_residual = (rec_output * rec_weights[:, None]).sum(dim=0)
                        donor_final[token_id] = donor_final[token_id] + (
                            self.recruitment_influence.to(donor_final.dtype)
                            * rec_residual.to(donor_final.dtype)
                        )

                    recruited_ids_all.append(rec_ids)
                    recruited_messages_all.append(rec_signals.message)
                    recruited_confidence_all.append(rec_signals.confidence)
                    recruited_abstention_all.append(rec_signals.abstention)
                    recruited_token_ids_all.append(
                        torch.full(
                            (capacity,),
                            token_id,
                            dtype=torch.long,
                            device=hidden_states.device,
                        )
                    )
                    recruited_received_all.append(rec_received)
                    recruited_signals_all.append(rec_signals)
                    recruited_pairs += capacity

            independent_pairs = 0
            if self.execution.stage >= MoPStage.INDEPENDENT_TILE_ROUTING:
                independent_output, independent_pairs = self._run_independent_route(hidden_states)
                beta = self.independent_route_influence
                if float(beta.detach().item()) != 0.0:
                    donor_final = (
                        (1.0 - beta.to(donor_final.dtype)) * donor_final
                        + beta.to(donor_final.dtype) * independent_output.to(donor_final.dtype)
                    )
            else:
                self.last_independent_cell_ids = ()

            local_state_updates = 0
            if self.execution.stage >= MoPStage.LOCAL_STATE:
                local_state_updates += self._update_local_state(
                    active_cell_ids,
                    signals,
                    received,
                )
                for rec_ids, rec_signals, rec_received in zip(
                    recruited_ids_all,
                    recruited_signals_all,
                    recruited_received_all,
                ):
                    local_state_updates += self._update_local_state(
                        rec_ids,
                        rec_signals,
                        rec_received,
                    )

            if recruited_ids_all:
                recruited_ids = torch.cat(recruited_ids_all, dim=0)
                recruited_messages = torch.cat(recruited_messages_all, dim=0)
                recruited_confidence = torch.cat(recruited_confidence_all, dim=0)
                recruited_abstention = torch.cat(recruited_abstention_all, dim=0)
                recruited_token_ids = torch.cat(recruited_token_ids_all, dim=0)
                self.last_recruited_cell_ids = tuple(
                    sorted({int(value) for value in recruited_ids.detach().tolist()})
                )
                telemetry_messages = torch.cat([signals.message, recruited_messages], dim=0)
                telemetry_confidence = torch.cat(
                    [signals.confidence, recruited_confidence], dim=0
                )
                telemetry_abstention = torch.cat(
                    [signals.abstention, recruited_abstention], dim=0
                )
                telemetry_tokens = torch.cat([active_token_ids, recruited_token_ids], dim=0)
                telemetry_cells = torch.cat([active_cell_ids, recruited_ids], dim=0)
            else:
                self.last_recruited_cell_ids = ()
                telemetry_messages = signals.message
                telemetry_confidence = signals.confidence
                telemetry_abstention = signals.abstention
                telemetry_tokens = active_token_ids
                telemetry_cells = active_cell_ids

            final_disagreement, telemetry_counts = self._message_stats(
                telemetry_messages,
                telemetry_tokens,
                token_count,
            )
            active_tokens = telemetry_counts > 0
            if bool(active_tokens.any()):
                mean_disagreement = float(final_disagreement[active_tokens].mean().item())
                max_disagreement = float(final_disagreement[active_tokens].max().item())
                scalar_confidence = telemetry_confidence.float().mean(dim=-1)
                conf_sums = torch.zeros(
                    token_count, dtype=torch.float32, device=hidden_states.device
                )
                abst_sums = torch.zeros_like(conf_sums)
                conf_sums.index_add_(0, telemetry_tokens, scalar_confidence)
                abst_sums.index_add_(0, telemetry_tokens, telemetry_abstention.float())
                safe = telemetry_counts.clamp_min(1.0)
                mean_confidence = float((conf_sums[active_tokens] / safe[active_tokens]).mean().item())
                mean_abstention = float((abst_sums[active_tokens] / safe[active_tokens]).mean().item())
            else:
                mean_disagreement = max_disagreement = 0.0
                mean_confidence = mean_abstention = 0.0

            if self.collect_telemetry:
                self.last_telemetry = CellTelemetrySummary(
                    active_cell_token_pairs=int(telemetry_cells.numel()),
                    unique_cells_seen=int(torch.unique(telemetry_cells).numel()),
                    mean_confidence=mean_confidence,
                    mean_abstention=mean_abstention,
                    mean_disagreement=mean_disagreement,
                    max_disagreement=max_disagreement,
                    recommended_action=self.disagreement_thresholds.action(mean_disagreement),
                    communication_rounds=communication_rounds,
                    recruited_cell_token_pairs=recruited_pairs,
                    unique_recruited_cells=len(self.last_recruited_cell_ids),
                    local_state_updates=local_state_updates,
                    independent_route_cell_token_pairs=independent_pairs,
                )

            return donor_final


    def _looks_like_moe_parent(module: Any) -> bool:
        return getattr(module, "experts", None) is not None and hasattr(module, "gate")


    def _looks_like_packed_bank(experts: Any) -> bool:
        return hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")


    def install_parameter_cell_reference(
        model: Any,
        *,
        tile_width: int = 128,
        membrane_config: CellMembraneConfig = CellMembraneConfig(),
        disagreement: DisagreementThresholds = DisagreementThresholds(),
        collect_telemetry: bool = True,
        execution: CellExecutionConfig = CellExecutionConfig(),
        budget: CellBudget = CellBudget(),
    ) -> CellPatchReport:
        """Install the zero-gated parameter ecology into packed routed MoE banks."""
        modules = tuple(model.modules())
        moe_modules = wrapped = already = control_parameters = 0

        for module in modules:
            if not _looks_like_moe_parent(module):
                continue
            experts = module.experts
            if isinstance(experts, CellizedPackedExpertsWrapper):
                moe_modules += 1
                already += experts.num_experts
                control_parameters += experts.control_parameter_count
                continue
            if not _looks_like_packed_bank(experts):
                continue

            wrapper = CellizedPackedExpertsWrapper(
                experts,
                tile_width=tile_width,
                membrane_config=membrane_config,
                disagreement=disagreement,
                collect_telemetry=collect_telemetry,
                execution=execution,
                budget=budget,
            )
            module.experts = wrapper
            moe_modules += 1
            wrapped += wrapper.num_experts
            control_parameters += wrapper.control_parameter_count

        if wrapped == 0 and already == 0:
            raise RuntimeError(
                "no packed routed expert banks were found for parameter-cell installation"
            )
        return CellPatchReport(
            moe_modules=moe_modules,
            wrapped_experts=wrapped,
            already_wrapped=already,
            tile_width=tile_width,
            control_parameters=control_parameters,
        )


    def restore_parameter_cell_reference(model: Any) -> int:
        """Restore donor packed expert banks and return the routed-expert count restored."""
        restored = 0
        modules = tuple(model.modules())
        for module in modules:
            if not _looks_like_moe_parent(module):
                continue
            experts = module.experts
            if isinstance(experts, CellizedPackedExpertsWrapper):
                restored += experts.num_experts
                module.experts = experts.expert_bank
        return restored


else:  # pragma: no cover
    CellSignals = None

    class _TorchRequired:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "parameter-cell neural execution requires PyTorch; "
                "install Leviathan with `python -m pip install -e '.[inference]'`"
            )

    CellControlMembrane = SparseCellCommunication = AssociativeCellRecruiter = _TorchRequired
    CellizedPackedExpertsWrapper = _TorchRequired

    def install_parameter_cell_reference(*args: Any, **kwargs: Any) -> CellPatchReport:
        raise RuntimeError("parameter-cell neural execution requires PyTorch")

    def restore_parameter_cell_reference(model: Any) -> int:
        return 0

"""Mixture-of-Parameterized-Cells reference architecture.

This module grows Leviathan's exact MoP-0 parameter tiles into richer computational
cells without changing the inherited DeepSeek function at insertion.

The key invariant is:

    cell_output = inherited_tile_output + influence_gate * learned_refinement

and influence_gate starts at exactly zero.

The cell membrane can already emit confidence, abstention, proposal messages and
recruitment queries. Those signals are observational until later stages earn the
right to affect routing or the residual stream. This keeps one global model, one
training objective and one final output; cells are components of one network, not
independent agents.
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
class CellTelemetrySummary:
    active_cell_token_pairs: int
    unique_cells_seen: int
    mean_confidence: float
    mean_abstention: float
    mean_disagreement: float
    max_disagreement: float
    recommended_action: CellAction

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


    class CellControlMembrane(nn.Module):
        """Cheap expressive membrane around inherited parameter tiles."""

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
            self.influence_gate = nn.Parameter(torch.tensor(float(config.initial_influence)))

            nn.init.normal_(self.cell_embedding.weight, std=0.01)
            nn.init.normal_(self.refine_up.weight, std=0.01)

        @property
        def influence(self) -> Any:
            return self.influence_gate.clamp(0.0, 1.0)

        def set_influence(self, value: float, *, transplant_phase: Any) -> None:
            if not 0.0 <= value <= 1.0:
                raise ValueError("cell influence must be in [0, 1]")
            if value != 0.0 and not _gate_phase_allows_influence(transplant_phase):
                raise RuntimeError("cell membrane cannot affect output before gate warmup")
            with torch.no_grad():
                self.influence_gate.fill_(float(value))

        def forward(self, hidden_states: Any, cell_ids: Any) -> CellSignals:
            if hidden_states.ndim != 2:
                raise ValueError("hidden_states must be [assignments, hidden]")
            if cell_ids.ndim != 1 or cell_ids.shape[0] != hidden_states.shape[0]:
                raise ValueError("cell_ids must align with hidden_states")
            if cell_ids.numel() and (
                int(cell_ids.min().item()) < 0 or int(cell_ids.max().item()) >= self.num_cells
            ):
                raise IndexError("cell id out of range")

            control = torch.tanh(
                self.state_down(hidden_states.float()) + self.cell_embedding(cell_ids)
            )
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
            )


    class SparseCellCommunication(nn.Module):
        """One bounded communication round among active messages in the same group."""

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
        """Content-addressed recruitment over a learned cell-key table."""

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
        """Exact tiled expert execution plus an inert expressive cell membrane."""

        def __init__(
            self,
            expert_bank: Any,
            *,
            tile_width: int = 128,
            membrane_config: CellMembraneConfig = CellMembraneConfig(),
            disagreement: DisagreementThresholds = DisagreementThresholds(),
            collect_telemetry: bool = True,
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
            self.last_telemetry: CellTelemetrySummary | None = None

            if self.intermediate_size % self.tile_width:
                raise ValueError("tile_width must divide packed expert intermediate size")
            self.membrane = CellControlMembrane(
                hidden_size=self.hidden_size,
                num_cells=self.num_cells,
                config=membrane_config,
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

        def cell_id(self, expert_index: int, tile_index: int) -> int:
            if not 0 <= expert_index < self.num_experts:
                raise IndexError("expert index out of range")
            if not 0 <= tile_index < self.tiles_per_expert:
                raise IndexError("tile index out of range")
            return expert_index * self.tiles_per_expert + tile_index

        def _activate(self, gate_up: Any) -> Any:
            apply_gate = getattr(self.expert_bank, "_apply_gate", None)
            if callable(apply_gate):
                return apply_gate(gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            limit = float(getattr(self.expert_bank, "limit", 0.0) or 0.0)
            if limit > 0:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            act_fn = getattr(self.expert_bank, "act_fn", F.silu)
            return act_fn(gate) * up

        def set_influence(self, value: float, *, transplant_phase: Any) -> None:
            self.membrane.set_influence(value, transplant_phase=transplant_phase)

        def forward(self, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
            final = torch.zeros_like(hidden_states)
            token_count = int(hidden_states.shape[0])
            message_dim = self.membrane.config.message_dim

            if self.collect_telemetry:
                message_sum = torch.zeros(
                    token_count, message_dim, dtype=torch.float32, device=hidden_states.device
                )
                message_sq_sum = torch.zeros_like(message_sum)
                assignment_count = torch.zeros(
                    token_count, dtype=torch.float32, device=hidden_states.device
                )
                confidence_sum = torch.zeros_like(assignment_count)
                abstention_sum = torch.zeros_like(assignment_count)
                unique_cells: set[int] = set()
                active_pairs = 0

            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_tensor in hit:
                expert_index = int(expert_tensor[0].item())
                if expert_index >= self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(mask[expert_index])
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
                    cell_id = self.cell_id(expert_index, tile_index)
                    ids = torch.full(
                        (token_idx.shape[0],),
                        cell_id,
                        dtype=torch.long,
                        device=hidden_states.device,
                    )
                    signals = self.membrane(hidden_states[token_idx], ids)
                    contribution = contribution + (
                        self.membrane.influence.to(contribution.dtype)
                        * signals.refinement.to(contribution.dtype)
                    )
                    expert_output = expert_output + contribution.to(expert_output.dtype)

                    if self.collect_telemetry:
                        message = signals.message.float()
                        scalar_confidence = signals.confidence.float().mean(dim=-1)
                        message_sum.index_add_(0, token_idx, message)
                        message_sq_sum.index_add_(0, token_idx, message.square())
                        assignment_count.index_add_(
                            0,
                            token_idx,
                            torch.ones_like(token_idx, dtype=torch.float32),
                        )
                        confidence_sum.index_add_(0, token_idx, scalar_confidence)
                        abstention_sum.index_add_(0, token_idx, signals.abstention.float())
                        unique_cells.add(cell_id)
                        active_pairs += int(token_idx.shape[0])

                expert_output = expert_output * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, expert_output.to(final.dtype))

            if self.collect_telemetry:
                safe_count = assignment_count.clamp_min(1.0)
                mean_message = message_sum / safe_count[:, None]
                message_variance = (
                    message_sq_sum / safe_count[:, None] - mean_message.square()
                ).clamp_min(0.0)
                per_token_disagreement = message_variance.mean(dim=-1)
                active_tokens = assignment_count > 0
                if bool(active_tokens.any()):
                    mean_disagreement = float(
                        per_token_disagreement[active_tokens].mean().item()
                    )
                    max_disagreement = float(
                        per_token_disagreement[active_tokens].max().item()
                    )
                    mean_confidence = float(
                        (confidence_sum[active_tokens] / safe_count[active_tokens]).mean().item()
                    )
                    mean_abstention = float(
                        (abstention_sum[active_tokens] / safe_count[active_tokens]).mean().item()
                    )
                else:
                    mean_disagreement = max_disagreement = 0.0
                    mean_confidence = mean_abstention = 0.0
                self.last_telemetry = CellTelemetrySummary(
                    active_cell_token_pairs=active_pairs,
                    unique_cells_seen=len(unique_cells),
                    mean_confidence=mean_confidence,
                    mean_abstention=mean_abstention,
                    mean_disagreement=mean_disagreement,
                    max_disagreement=max_disagreement,
                    recommended_action=self.disagreement_thresholds.action(mean_disagreement),
                )

            return final


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
    ) -> CellPatchReport:
        """Install inert parameter-cell membranes into packed routed MoE banks."""
        modules = tuple(model.modules())
        moe_modules = wrapped = already = control_parameters = 0

        for module in modules:
            if not _looks_like_moe_parent(module):
                continue
            experts = module.experts
            if isinstance(experts, CellizedPackedExpertsWrapper):
                moe_modules += 1
                already += experts.num_experts
                control_parameters += sum(
                    parameter.numel() for parameter in experts.membrane.parameters()
                )
                continue
            if not _looks_like_packed_bank(experts):
                continue

            wrapper = CellizedPackedExpertsWrapper(
                experts,
                tile_width=tile_width,
                membrane_config=membrane_config,
                disagreement=disagreement,
                collect_telemetry=collect_telemetry,
            )
            module.experts = wrapper
            moe_modules += 1
            wrapped += wrapper.num_experts
            control_parameters += sum(
                parameter.numel() for parameter in wrapper.membrane.parameters()
            )

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

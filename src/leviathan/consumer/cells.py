"""Dense-Qwen ancestral cells with bounded token-local neural discussion.

Cross-layer recruitment is not claimed: equal dimensions do not prove compatible
representations. Donor tensors are views, never a second registered model.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .quantization import slice_weight, supports_slicing

class SwiGLUCells:
    def __init__(self, donor: nn.Module, width: int = 128):
        self.donor = donor
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not hasattr(donor, name):
                raise TypeError(f"SwiGLU donor missing {name}")
            if not supports_slicing(getattr(donor, name)):
                raise TypeError(f"{name} has no format-aware tile accessor")
        self.hidden = donor.gate_proj.in_features
        self.intermediate = donor.gate_proj.out_features
        if width <= 0 or self.intermediate % width:
            raise ValueError("Cell width must divide FFN width")
        if donor.up_proj.in_features != self.hidden or donor.up_proj.out_features != self.intermediate:
            raise ValueError("Gate/up geometry mismatch")
        if donor.down_proj.in_features != self.intermediate or donor.down_proj.out_features != self.hidden:
            raise ValueError("Down geometry mismatch")
        self.width, self.count = width, self.intermediate // width
    def body(self, x: Tensor, ids: Tensor) -> Tensor:
        if x.ndim != 2 or ids.ndim != 1 or len(ids) != len(x):
            raise ValueError("Expected aligned [assignments, hidden] and [assignments]")
        if ids.dtype != torch.long or (ids.numel() and (ids.min() < 0 or ids.max() >= self.count)):
            raise ValueError("Invalid cell IDs")
        result = torch.zeros_like(x)
        for cid_t in torch.unique(ids):
            cid = int(cid_t)
            idx = (ids == cid).nonzero(as_tuple=True)[0]
            region = slice(cid * self.width, (cid + 1) * self.width)
            gate, up, down = self.donor.gate_proj, self.donor.up_proj, self.donor.down_proj
            def project(layer):
                b = None if layer.bias is None else layer.bias[region].to(x.dtype)
                return F.linear(x[idx], slice_weight(layer, rows=region).to(x.dtype), b)
            act = getattr(self.donor, "act_fn", F.silu)(project(gate)) * project(up)
            out = F.linear(act, slice_weight(down, cols=region).to(x.dtype))
            result = result.index_copy(0, idx, out)
        return result
    def reconstruct(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, self.hidden)
        result = torch.zeros_like(flat)
        for cid in range(self.count):
            ids = torch.full((len(flat),), cid, dtype=torch.long, device=x.device)
            result = result + self.body(flat, ids)
        if self.donor.down_proj.bias is not None:
            result = result + self.donor.down_proj.bias.to(result.dtype)
        return result.reshape(x.shape)

@dataclass(frozen=True)
class EcologyConfig:
    latent_dim: int = 64
    seed_cells: int = 4
    recruit_cells: int = 2
    max_cells: int = 8
    max_neighbors: int = 4
    rounds: int = 2
    recruit_threshold: float = 0.15
    coalition_size: int = 8
    macro_top_k: int = 0
    def __post_init__(self):
        if min(self.latent_dim, self.seed_cells, self.max_cells, self.max_neighbors,
               self.rounds, self.coalition_size) < 1:
            raise ValueError("Positive dimensions and budgets required")
        if self.seed_cells > self.max_cells or self.recruit_cells < 0:
            raise ValueError("Invalid cell budgets")
        if self.rounds > 2 or self.macro_top_k < 0:
            raise ValueError("Reference supports at most two rounds")
        if not math.isfinite(self.recruit_threshold) or self.recruit_threshold < 0:
            raise ValueError("Invalid disagreement threshold")

@dataclass
class CellResult:
    proposal: Tensor
    state: Tensor
    disagreement: Tensor
    confidence: Tensor
    abstention: Tensor
    ids: Tensor
    mask: Tensor
    recruited: int

class CellEcology(nn.Module):
    """Explicit per-token depth state; confidence/disagreement remain uncalibrated."""
    def __init__(self, bank: SwiGLUCells, config: EcologyConfig = EcologyConfig()):
        super().__init__()
        self.bank, self.config = bank, config
        d, n = config.latent_dim, bank.count
        self.keys = nn.Parameter(torch.randn(n, d) / math.sqrt(d))
        self.query = nn.Linear(d, d, bias=False)
        self.body_down = nn.Linear(bank.hidden, d, bias=False)
        self.embed = nn.Embedding(n, d)
        self.message = nn.Linear(d, d, bias=False)
        self.peer = nn.Linear(d, d, bias=False)
        self.recruit_query = nn.Linear(d, d, bias=False)
        self.state_cell = nn.GRUCell(d, d)
        self.confidence = nn.Linear(d, 4)
        self.abstention = nn.Linear(d, 1)
        self.communication_gate = nn.Parameter(torch.tensor(0.0))
        self.state_gate = nn.Parameter(torch.tensor(0.0))
        self.recruit_gate = nn.Parameter(torch.tensor(0.0))
        self.last_recruited = 0
    def _select(self, scores: Tensor, k: int) -> Tensor:
        if self.config.macro_top_k:
            c = self.config.coalition_size
            groups = math.ceil(self.bank.count / c)
            padded = F.pad(scores, (0, groups * c - self.bank.count), value=-torch.inf)
            group_scores = padded.reshape(len(scores), groups, c).amax(-1)
            gk = min(groups, max(self.config.macro_top_k, math.ceil(k / c)))
            chosen = group_scores.topk(gk, dim=-1).indices
            allowed = (torch.arange(groups, device=scores.device)[None, :, None]
                       == chosen[:, None, :]).any(-1).repeat_interleave(c, -1)[:, :self.bank.count]
            scores = scores.masked_fill(~allowed, -torch.inf)
        return scores.topk(k, -1).indices
    def _discuss(self, controls: Tensor, mask: Tensor) -> Tensor:
        msg = self.message(controls)
        scores = msg @ msg.transpose(-1, -2) / math.sqrt(msg.shape[-1])
        scores = scores.masked_fill(~mask[:, None, :], -torch.inf)
        k = min(self.config.max_neighbors, controls.shape[1])
        values, neighbors = scores.topk(k, dim=-1)
        weights = torch.softmax(values, -1)
        values_ = msg[:, None].expand(-1, msg.shape[1], -1, -1)
        gathered = values_.gather(2, neighbors[..., None].expand(-1, -1, -1, msg.shape[-1]))
        received = (weights[..., None] * gathered).sum(-2)
        return controls + torch.tanh(self.communication_gate) * torch.tanh(self.peer(received))
    @staticmethod
    def _disagreement(controls: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(controls.dtype)[..., None]
        mean = (controls * weights).sum(1) / weights.sum(1).clamp_min(1)
        return ((controls - mean[:, None]).square() * weights).sum((1, 2)) / (
            weights.sum(1).squeeze(-1).clamp_min(1) * controls.shape[-1])
    def forward(self, hidden: Tensor, latent: Tensor, state: Tensor | None = None) -> CellResult:
        n, d = latent.shape
        if hidden.shape != (n, self.bank.hidden):
            raise ValueError("Aligned hidden and latent tokens required")
        if state is None:
            state = latent.new_zeros(n, self.bank.count, d)
        if state.shape != (n, self.bank.count, d):
            raise ValueError("State belongs to another token cohort or cell bank")
        k = min(self.config.seed_cells, self.bank.count)
        cap = min(self.config.max_cells, self.bank.count)
        scores = self.query(latent) @ self.keys.T / math.sqrt(d)
        ids = self._select(scores, k)
        mask = torch.ones_like(ids, dtype=torch.bool)
        control = torch.tanh(latent[:, None] + self.embed(ids) +
                             torch.tanh(self.state_gate) * state.gather(
                                 1, ids[..., None].expand(-1, -1, d)))
        if self.config.rounds:
            control = self._discuss(control, mask)
        disagreement = self._disagreement(control, mask)
        extra = min(self.config.recruit_cells, cap - k)
        recruited = 0
        if extra > 0:
            rscores = self.recruit_query(control.mean(1)) @ self.keys.T / math.sqrt(d)
            rscores = rscores.scatter(1, ids, -torch.inf)
            more = rscores.topk(extra, -1).indices
            valid = (disagreement >= self.config.recruit_threshold)[:, None].expand(-1, extra)
            more_control = torch.tanh(latent[:, None] + self.embed(more) +
                                     torch.tanh(self.state_gate) * state.gather(
                                         1, more[..., None].expand(-1, -1, d)))
            ids = torch.cat((ids, more), 1)
            mask = torch.cat((mask, valid), 1)
            control = torch.cat((control, more_control), 1)
            recruited = int(valid.sum().detach())
            if self.config.rounds == 2:
                revised = self._discuss(control, mask)
                control = control + (torch.tanh(self.recruit_gate)
                                     * valid.any(-1)[:, None, None].to(control.dtype)
                                     * (revised - control))
        # Separate seed GEMMs: observing recruits must not change seed batch shapes.
        def projected_bodies(selected_ids: Tensor, selected_mask: Tensor) -> Tensor:
            repeated = hidden[:, None].expand(-1, selected_ids.shape[1], -1).reshape(-1, self.bank.hidden)
            result = hidden.new_zeros(len(repeated), self.bank.hidden)
            rows = selected_mask.reshape(-1).nonzero(as_tuple=True)[0]
            result = result.index_copy(0, rows, self.bank.body(
                repeated[rows], selected_ids.reshape(-1)[rows]))
            return self.body_down(result.to(self.body_down.weight.dtype)).reshape(n, -1, d)
        seed_bodies = projected_bodies(ids[:, :k], mask[:, :k])
        # Recruits use their own differentiable scores; seed mass stays unchanged.
        # Discrete top-k identities still require a routing learning objective.
        seed_coefficients = scores.gather(1, ids[:, :k]).softmax(-1)
        proposal = ((seed_bodies + control[:, :k]) * seed_coefficients[..., None]).sum(1)
        if extra:
            recruited_bodies = projected_bodies(ids[:, k:], mask[:, k:])
            rec_coefficients = rscores.gather(1, ids[:, k:]).softmax(-1)
            rec_coefficients = rec_coefficients * mask[:, k:].to(rec_coefficients.dtype)
            recruited_proposal = ((recruited_bodies + control[:, k:]) * rec_coefficients[..., None]).sum(1)
            proposal = proposal + torch.tanh(self.recruit_gate) * recruited_proposal
        old = state.gather(1, ids[..., None].expand(-1, -1, d))
        new = self.state_cell(control.reshape(-1, d), old.reshape(-1, d)).reshape_as(old)
        new = torch.where(mask[..., None], new, old)
        next_state = state.scatter(1, ids[..., None].expand(-1, -1, d), new)
        self.last_recruited = recruited
        return CellResult(proposal, next_state, self._disagreement(control, mask),
                          self.confidence(control).sigmoid(), self.abstention(control).sigmoid(),
                          ids, mask, recruited)

"""NRDF: a function-preserving recurrent adapter, not a trained reasoning claim.

The inherited Transformer/DeltaNet backbone executes normally. A small shared
Transformer recurrently refines token-local hypothesis slots, with optional real
ancestral cells. It never reuses or mutates the donor's causal KV/DeltaNet caches.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .cells import CellEcology, EcologyConfig, SwiGLUCells


@dataclass(frozen=True)
class NRDFConfig:
    latent_dim: int = 64
    slots: int = 4
    heads: int = 4
    min_loops: int = 1
    max_loops: int = 4
    step_size: float = 0.25
    delta_threshold: float = 0.0
    halting_threshold: float = 0.95
    learned_halting: bool = False
    step_conditioning: bool = False
    fast_rank: int = 4
    plastic_norm: float = 0.1
    chunk_tokens: int = 32
    cell_width: int = 128
    ancestral_cells: bool = False
    pulse_interval: int = 0

    def __post_init__(self):
        if self.pulse_interval < 0:
            raise ValueError("pulse_interval cannot be negative")
        if min(self.latent_dim, self.slots, self.heads, self.min_loops,
               self.max_loops, self.fast_rank, self.chunk_tokens, self.cell_width) <= 0:
            raise ValueError("Positive dimensions/budgets required")
        if self.latent_dim % self.heads or self.min_loops > self.max_loops:
            raise ValueError("Invalid attention or loop budget")
        if not 0 < self.step_size <= 1 or not 0 < self.halting_threshold <= 1:
            raise ValueError("Invalid step/halting threshold")
        if not math.isfinite(self.delta_threshold) or self.delta_threshold < 0:
            raise ValueError("Invalid delta threshold")
        if not math.isfinite(self.plastic_norm) or self.plastic_norm <= 0:
            raise ValueError("Invalid plastic norm bound")


@dataclass
class DepthTrace:
    loops: Tensor
    halt_probabilities: Tensor
    deltas: Tensor
    active_rows_per_loop: tuple[int, ...]
    recruited_pairs: int
    routes: tuple[Tensor, ...]


class FastOverlay(nn.Module):
    """Task-local low-rank fast weights, never optimizer-owned pretrained weights.

    A is generated from the current sequence of depth states and norm bounded.
    V is a shared learned basis. No module-global fast-state mutation occurs.
    """
    def __init__(self, dim: int, rank: int, norm: float):
        super().__init__()
        self.dim, self.rank, self.norm = dim, rank, norm
        self.propose = nn.Linear(dim, dim * rank)
        self.basis = nn.Parameter(torch.randn(rank, dim) / math.sqrt(dim))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor, prior: Tensor | None) -> tuple[Tensor, Tensor]:
        proposed = self.propose(x).reshape(len(x), self.dim, self.rank).tanh()
        if prior is None:
            prior = torch.zeros_like(proposed)
        state = 0.75 * prior + 0.25 * proposed
        norms = state.flatten(1).norm(dim=-1).clamp_min(1e-12)
        state = state * (self.norm / norms).clamp(max=1)[:, None, None]
        delta = torch.bmm(state, (x @ self.basis.T)[..., None]).squeeze(-1)
        return torch.tanh(self.gate) * delta, state


class RecurrentFabric(nn.Module):
    """A shared Transformer over hypothesis slots, with actual row compaction.

    Tokens never attend to each other here; their donor states already contain
    causal context. Batch permutation/chunking therefore preserves semantics.
    Soft slot selection is implemented; semantic branch invention is not claimed.
    """
    def __init__(self, hidden: int, config: NRDFConfig,
                 ecology: CellEcology | None = None):
        super().__init__()
        self.config, self.ecology = config, ecology
        d = config.latent_dim
        self.input_down = nn.Linear(hidden, d, bias=False)
        self.slot_embedding = nn.Parameter(torch.randn(config.slots, d) * 0.02)
        self.norm = nn.RMSNorm(d)
        self.attention = nn.MultiheadAttention(d, config.heads, batch_first=True, dropout=0)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.SiLU(), nn.Linear(d * 2, d))
        self.inject = nn.Linear(d * 2, d, bias=False)
        self.loop_features = nn.Linear(3, d, bias=False)
        self.slot_score = nn.Linear(d, 1, bias=False)
        self.halt_head = nn.Linear(d + 2, 1)
        nn.init.constant_(self.halt_head.bias, -3.0)
        self.fast = FastOverlay(d, config.fast_rank, config.plastic_norm)
        self.cell_gate = nn.Parameter(torch.zeros(()))
        self.pulse_gate = nn.Parameter(torch.zeros(()))
        self.pulse_bridge = None
        self.output = nn.Linear(d, hidden, bias=False)
        nn.init.normal_(self.output.weight, std=0.01)

    def forward(self, hidden: Tensor, *, loops: int | None = None,
                adaptive: bool = False, pulse: Tensor | None = None) -> tuple[Tensor, DepthTrace]:
        cfg = self.config
        max_loops = cfg.max_loops if loops is None else loops
        if max_loops < cfg.min_loops or max_loops > cfg.max_loops:
            raise ValueError("Requested loops outside configured bounds")
        if adaptive and self.training:
            raise ValueError("Hard halting is evaluation-only; train with sampled fixed depths")
        base = self.input_down(hidden.to(self.input_down.weight.dtype))
        n, d = base.shape
        if pulse is not None:
            if pulse.shape != base.shape:
                raise ValueError("Pulse must be aligned token-local latent data")
            base = base + pulse
        slots = base[:, None] + self.slot_embedding[None]
        active = torch.arange(n, device=base.device)
        result = base.new_zeros(n, d)
        counts = torch.zeros(n, dtype=torch.long, device=base.device)
        local_state = None
        fast_state = None
        halt_history, delta_history, routes, sizes = [], [], [], []
        recruited = 0
        for step in range(max_loops):
            sizes.append(len(active))
            if not len(active):
                break
            prev = slots.mean(1)
            injected = self.inject(torch.cat((slots, base[active, None].expand_as(slots)), -1))
            if cfg.step_conditioning:
                values = base.new_tensor([step / max_loops, 1 / max_loops, math.log1p(step)])
                injected = injected + self.loop_features(values)[None, None]
            normalized = self.norm(slots + injected)
            attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
            next_slots = slots + cfg.step_size * (attended + self.ffn(normalized))
            pooled = (next_slots * self.slot_score(next_slots).softmax(1)).sum(1)
            disagreement = next_slots.var(1, unbiased=False).mean(-1)
            if self.ecology is not None:
                cell = self.ecology(hidden[active], pooled, local_state)
                local_state = cell.state
                pooled = pooled + torch.tanh(self.cell_gate) * cell.proposal
                disagreement = cell.disagreement
                recruited += cell.recruited
                routes.append(cell.ids.detach().cpu())
            plastic, fast_state = self.fast(pooled, fast_state)
            pooled = pooled + plastic
            if cfg.pulse_interval and (step + 1) % cfg.pulse_interval == 0:
                if self.pulse_bridge is None:
                    raise RuntimeError("Pulse interval enabled without a same-model pulse bridge")
                pooled = pooled + torch.tanh(self.pulse_gate) * self.pulse_bridge(pooled)
            delta = (pooled - prev).float().norm(dim=-1) / prev.float().norm(dim=-1).clamp_min(1)
            halt = self.halt_head(torch.cat((pooled, delta[:, None].to(pooled.dtype),
                                            disagreement[:, None].to(pooled.dtype)), -1)).sigmoid().squeeze(-1)
            halt_history.append(base.new_zeros(n).index_copy(0, active, halt))
            delta_history.append(base.new_zeros(n).index_copy(0, active, delta.to(base.dtype)))
            stop = torch.zeros(len(active), dtype=torch.bool, device=base.device)
            if step + 1 >= cfg.min_loops and adaptive:
                if cfg.delta_threshold > 0:
                    stop = stop | (delta <= cfg.delta_threshold)
                if cfg.learned_halting:
                    stop = stop | (halt >= cfg.halting_threshold)
            if step + 1 == max_loops:
                stop = torch.ones_like(stop)
            result = result.index_copy(0, active[stop], pooled[stop])
            counts = counts.index_fill(0, active[stop], step + 1)
            keep = ~stop
            slots = next_slots[keep] + (pooled - next_slots.mean(1))[keep, None]
            active = active[keep]
            if local_state is not None:
                local_state = local_state[keep]
            fast_state = fast_state[keep]
        trace = DepthTrace(counts, torch.stack(halt_history, 1), torch.stack(delta_history, 1),
                           tuple(sizes), recruited, tuple(routes))
        return self.output(self.norm(result)), trace


class QwenNRDFWrapper(nn.Module):
    """Original Qwen FFN plus an independently trainable, zero-initialized graft.

    eval + gate zero can bypass the graft exactly. Training always builds its graph
    so the outer gate receives gradient. Auxiliary losses must train inner modules
    while this gate is zero; the implementation does not claim otherwise.
    """
    def __init__(self, donor: nn.Module, config: NRDFConfig = NRDFConfig(),
                 ecology_config: EcologyConfig | None = None):
        super().__init__()
        self.donor, self.config = donor, config
        self.hidden = int(donor.gate_proj.in_features)
        ecology = None
        if config.ancestral_cells:
            bank = SwiGLUCells(donor, config.cell_width)
            ecfg = ecology_config or EcologyConfig(latent_dim=config.latent_dim)
            if ecfg.latent_dim != config.latent_dim:
                raise ValueError("Ecology/fabric latent dimensions differ")
            ecology = CellEcology(bank, ecfg)
        self.fabric = RecurrentFabric(self.hidden, config, ecology)
        device = next(donor.parameters()).device
        if device.type == "meta":
            raise ValueError("Materialize donor before installing a recurrent graft")
        self.fabric.to(device=device, dtype=torch.float32)
        self.gate = nn.Parameter(torch.zeros((), device=device))
        self.loops = config.max_loops
        self.adaptive = False
        self.observe_at_zero = False
        self.last_traces: list[DepthTrace] = []
        self.enabled = True

    def set_influence(self, value: float, *, experimental: bool = False):
        if not math.isfinite(value) or not -0.99 <= value <= 0.99:
            raise ValueError("Influence must be finite and within [-0.99, 0.99]")
        if value and not experimental:
            raise RuntimeError("Nonzero unvalidated grafts require explicit experimental opt-in")
        with torch.no_grad():
            self.gate.fill_(math.atanh(value))

    def forward(self, x: Tensor) -> Tensor:
        baseline = self.donor(x)
        self.last_traces = []
        if not self.enabled:
            return baseline
        if not self.training and not self.observe_at_zero and float(self.gate.detach()) == 0:
            return baseline
        flat = x.reshape(-1, self.hidden)
        if not len(flat):
            return baseline
        parts = []
        for start in range(0, len(flat), self.config.chunk_tokens):
            delta, trace = self.fabric(flat[start:start + self.config.chunk_tokens],
                                       loops=self.loops, adaptive=self.adaptive)
            parts.append(delta)
            self.last_traces.append(DepthTrace(trace.loops.detach(), trace.halt_probabilities.detach(),
                                               trace.deltas.detach(), trace.active_rows_per_loop,
                                               trace.recruited_pairs, trace.routes))
        delta = torch.cat(parts).reshape_as(baseline).to(baseline.dtype)
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Non-finite recurrent proposal: zero times NaN is not safe")
        return baseline + torch.tanh(self.gate).to(baseline.dtype) * delta


def install_nrdf(model: nn.Module, config: NRDFConfig, *, layers: tuple[int, ...] = (-1,)) -> list[str]:
    """Select actual decoder MLPs; never patch the vision encoder by name accident."""
    if any(isinstance(m, QwenNRDFWrapper) for m in model.modules()):
        raise ValueError("NRDF is already installed; explicitly restore before reinstalling")
    candidates = [(name, module) for name, module in model.named_modules()
                  if name.endswith(".mlp") and ".layers." in name and "visual" not in name
                  and all(hasattr(module, key) for key in ("gate_proj", "up_proj", "down_proj"))]
    if not candidates:
        raise ValueError("No compatible Qwen decoder FFNs found")
    indices = tuple(i if i >= 0 else len(candidates) + i for i in layers)
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(candidates) for i in indices):
        raise ValueError("Invalid or duplicate layer selection")
    wrappers = []
    for index in indices:
        name, donor = candidates[index]
        wrapper = QwenNRDFWrapper(donor, config)
        if config.pulse_interval:
            from .pulse import PulseBridge
            wrapper.fabric.pulse_bridge = PulseBridge(
                model.get_input_embeddings(), model.get_output_embeddings(),
                config.latent_dim, wrapper.hidden).to(device=wrapper.gate.device)
        wrappers.append((name, wrapper))
    for name, wrapper in wrappers:
        parent_name, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), attr, wrapper)
    return [name for name, _ in wrappers]


def restore_nrdf(model: nn.Module) -> int:
    wrappers = [(n, m) for n, m in model.named_modules() if isinstance(m, QwenNRDFWrapper)]
    for name, module in wrappers:
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, module.donor)
    return len(wrappers)


def graft_parameters(model: nn.Module):
    """Return only graft tensors, with no duplicated/accidentally trainable donor."""
    seen = set()
    for module in model.modules():
        if isinstance(module, QwenNRDFWrapper):
            for parameter in [module.gate, *module.fabric.parameters()]:
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

"""One-model token pulses: optional discrete scratch checkpoints, not private CoT.

This untrained bridge is a research interface. A token decoded from a random
bridge has no assured semantic meaning; rationale supervision is still required.
"""
from __future__ import annotations
import weakref
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PulseBridge(nn.Module):
    def __init__(self, embedding: nn.Module, lm_head: nn.Module, latent_dim: int, hidden: int):
        super().__init__()
        self._embedding = weakref.ref(embedding)
        self._lm_head = weakref.ref(lm_head)
        self.readout = nn.Linear(latent_dim, hidden, bias=False)
        self.reencode = nn.Linear(hidden, latent_dim, bias=False)
        self.last_tokens: Tensor | None = None

    def _logits(self, latent: Tensor) -> Tensor:
        head = self._lm_head()
        if head is None:
            raise RuntimeError("The owning model no longer exists")
        hidden = self.readout(latent.to(self.readout.weight.dtype))
        weight = next(head.parameters())
        return head(hidden.to(device=weight.device, dtype=weight.dtype)).float()

    def alignment_loss(self, latent: Tensor, teacher_token_ids: Tensor) -> Tensor:
        logits = self._logits(latent)
        return F.cross_entropy(logits, teacher_token_ids.to(logits.device))

    def forward(self, latent: Tensor) -> Tensor:
        embedding = self._embedding()
        if embedding is None:
            raise RuntimeError("The owning model no longer exists")
        ids = self._logits(latent).argmax(-1)
        self.last_tokens = ids.detach().cpu()
        weight = next(embedding.parameters())
        encoded = embedding(ids.to(weight.device))
        return self.reencode(encoded.to(device=latent.device, dtype=self.reencode.weight.dtype))

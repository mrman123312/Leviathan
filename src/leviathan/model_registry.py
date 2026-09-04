"""Typed access to Leviathan's model substrate/teacher registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Iterable


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "spec" / "model-registry.toml"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    repo_id: str
    role: str
    stage: str
    license: str
    total_parameters_b: float
    active_parameters_b: float
    multimodal: bool
    enabled_for_download: bool
    priority: int
    notes: str = ""

    @property
    def active_fraction(self) -> float | None:
        if self.total_parameters_b <= 0:
            return None
        return self.active_parameters_b / self.total_parameters_b

    @property
    def is_base(self) -> bool:
        return self.stage == "base"

    @property
    def is_teacher(self) -> bool:
        return "teacher" in self.role


class ModelRegistry:
    """Read-only registry with explicit safety around giant checkpoint acquisition."""

    def __init__(self, models: Iterable[ModelSpec]) -> None:
        self._models = {model.id: model for model in models}
        if len(self._models) != len(list(self._models.values())):
            raise ValueError("duplicate model IDs")

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_REGISTRY_PATH) -> "ModelRegistry":
        with path.open("rb") as f:
            raw = tomllib.load(f)
        models = [ModelSpec(**item) for item in raw.get("models", [])]
        return cls(models)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown Leviathan model id: {model_id}") from exc

    def all(self) -> tuple[ModelSpec, ...]:
        return tuple(sorted(self._models.values(), key=lambda model: model.priority))

    def by_role(self, role_fragment: str) -> tuple[ModelSpec, ...]:
        return tuple(
            model
            for model in self.all()
            if role_fragment.lower() in model.role.lower()
        )

    def base_models(self) -> tuple[ModelSpec, ...]:
        return tuple(model for model in self.all() if model.is_base)

    def teachers(self) -> tuple[ModelSpec, ...]:
        return tuple(model for model in self.all() if model.is_teacher)

    def require_download_permission(self, model_id: str, *, allow_disabled: bool = False) -> ModelSpec:
        model = self.get(model_id)
        if not model.enabled_for_download and not allow_disabled:
            raise PermissionError(
                f"{model_id} is disabled for automatic download; "
                "explicit frontier-size opt-in is required"
            )
        return model

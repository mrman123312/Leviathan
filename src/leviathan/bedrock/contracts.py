"""Shared resource, evidence and trace contracts; no optimization or model heads."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
import time
from typing import Any


def stable_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    model_calls: int = 96
    layer_calls: int = 4096
    environment_steps: int = 20
    generated_tokens: int = 64
    branches: int = 4
    wall_seconds: float = 300.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite budget required: {key}")
            if key != "wall_seconds" and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"Integer budget required: {key}")


@dataclass
class Meter:
    budget: Budget = field(default_factory=Budget)
    used: dict[str, int] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)

    def charge(self, resource: str, amount: int = 1):
        if not isinstance(amount, int) or amount < 0:
            raise ValueError("Nonnegative integer charge required")
        if resource not in asdict(self.budget) or resource == "wall_seconds":
            raise ValueError(f"Unknown counter {resource}")
        self.check_time()
        value = self.used.get(resource, 0) + amount
        if value > getattr(self.budget, resource):
            raise BudgetExceeded(f"{resource} budget exhausted ({value})")
        self.used[resource] = value

    def check_time(self):
        if time.monotonic() - self.started > self.budget.wall_seconds:
            raise BudgetExceeded("Wall-time budget exhausted between operations")

    def snapshot(self):
        return {"used": dict(self.used), "limits": asdict(self.budget),
                "elapsed_seconds": time.monotonic() - self.started}


@dataclass(frozen=True)
class Outcome:
    """Host-supplied test result; not permission for a neural model to self-certify.

    The host installs the verifier. Python object separation is NOT a security
    sandbox. Do not deserialize arbitrary model text into an authoritative Outcome.
    """
    subject_hash: str
    verifier: str
    passed: bool | None
    evidence_id: str
    scope: str
    independent: bool = False
    detail: str = ""

    def __post_init__(self):
        if not all((self.subject_hash, self.verifier, self.evidence_id, self.scope)):
            raise ValueError("Outcome identity, evidence and scope are required")
        if self.passed is not None and type(self.passed) is not bool:
            raise ValueError("Verification is true/false/unknown, never a truthy string")

    def binds(self, value: Any) -> bool:
        return self.subject_hash == stable_hash(value)


class CausalTrace:
    """Dependency trace with explicit invalidation, not proof that edges are causal."""
    def __init__(self):
        self._records: dict[str, dict] = {}

    def add(self, kind: str, payload: Any, parents: tuple[str, ...] = ()) -> str:
        if any(p not in self._records for p in parents):
            raise ValueError("Unknown dependency")
        identifier = f"event-{len(self._records)}"
        self._records[identifier] = {"kind": kind, "payload": json.loads(json.dumps(payload, allow_nan=False)),
                                     "parents": tuple(parents), "valid": True}
        return identifier

    def invalidate(self, identifier: str) -> tuple[str, ...]:
        if identifier not in self._records:
            raise KeyError(identifier)
        affected = {identifier}
        for key, rec in self._records.items():
            if key in affected or affected.intersection(rec["parents"]):
                rec["valid"] = False
                affected.add(key)
        return tuple(k for k in self._records if k in affected)

    @property
    def records(self):
        return json.loads(json.dumps(self._records))


class Competence:
    """Evidence-deduplicated Beta counts. Posterior estimates are not calibration."""
    def __init__(self):
        self.counts: dict[tuple[str, str], list[int]] = {}
        self.seen: set[tuple[str, str, str]] = set()

    def record(self, domain: str, method: str, outcome: Outcome):
        key = (domain, method, outcome.evidence_id)
        if outcome.passed is None or not outcome.independent or key in self.seen:
            return
        self.seen.add(key)
        values = self.counts.setdefault((domain, method), [0, 0])
        values[0 if outcome.passed else 1] += 1

    def estimate(self, domain: str, method: str) -> dict:
        s, f = self.counts.get((domain, method), (0, 0))
        a, b = s + 1, f + 1
        return {"successes": s, "failures": f, "posterior_mean": a / (a + b),
                "posterior_std": math.sqrt(a*b/((a+b)**2*(a+b+1))), "calibrated": False}

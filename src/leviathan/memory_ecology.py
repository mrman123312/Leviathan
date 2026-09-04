"""Executable L5 epistemic memory and belief-state reference store.

This is intentionally conservative. It separates event history from current belief
state, preserves provenance, records contradictions instead of overwriting them, and
can persist an append-only journal to disk. It is not yet the learned self-organizing
memory ecology; it is the trustworthy substrate that later learned write/retrieve/
merge/forget policies must beat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable

from .types import Belief, Provenance, ProvenanceKind, UncertaintyKind


class MemoryKind(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    CONTRADICTED = "contradicted"
    DEPRECATED = "deprecated"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    kind: MemoryKind
    payload: Any
    confidence: float
    provenance: Provenance
    evidence_refs: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    utility: float = 0.5
    verified: bool = False
    independent_verifications: int = 0
    status: MemoryStatus = MemoryStatus.ACTIVE

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("memory id cannot be empty")
        for name, value in (("confidence", self.confidence), ("utility", self.utility)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.independent_verifications < 0:
            raise ValueError("independent_verifications cannot be negative")
        if self.kind in {MemoryKind.SEMANTIC, MemoryKind.PROCEDURAL} and not self.verified:
            raise ValueError("semantic/procedural memory requires verification")


@dataclass(frozen=True, slots=True)
class MemoryWrite:
    sequence: int
    operation: str
    record_id: str
    record: MemoryRecord


class MemoryEcology:
    """Append-only reference memory with explicit epistemic promotion rules."""

    def __init__(self, journal_path: str | Path | None = None) -> None:
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self._records: dict[str, MemoryRecord] = {}
        self._writes: list[MemoryWrite] = []
        if self.journal_path is not None and self.journal_path.exists():
            self._load_journal()

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records.values())

    @property
    def writes(self) -> tuple[MemoryWrite, ...]:
        return tuple(self._writes)

    def _serialize(self, write: MemoryWrite) -> str:
        record = write.record
        payload = {
            "sequence": write.sequence,
            "operation": write.operation,
            "record_id": write.record_id,
            "record": {
                **asdict(record),
                "kind": record.kind.value,
                "status": record.status.value,
                "provenance": {
                    **asdict(record.provenance),
                    "kind": record.provenance.kind.value,
                },
            },
        }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _deserialize(line: str) -> MemoryWrite:
        raw = json.loads(line)
        rec = raw["record"]
        prov = rec["provenance"]
        provenance = Provenance(
            kind=ProvenanceKind(prov["kind"]),
            source_id=prov["source_id"],
            trust_prior=float(prov["trust_prior"]),
            source_version=prov.get("source_version"),
        )
        record = MemoryRecord(
            id=rec["id"],
            kind=MemoryKind(rec["kind"]),
            payload=rec["payload"],
            confidence=float(rec["confidence"]),
            provenance=provenance,
            evidence_refs=tuple(rec.get("evidence_refs", ())),
            source_refs=tuple(rec.get("source_refs", ())),
            tags=tuple(rec.get("tags", ())),
            utility=float(rec.get("utility", 0.5)),
            verified=bool(rec.get("verified", False)),
            independent_verifications=int(rec.get("independent_verifications", 0)),
            status=MemoryStatus(rec.get("status", "active")),
        )
        return MemoryWrite(
            sequence=int(raw["sequence"]),
            operation=str(raw["operation"]),
            record_id=str(raw["record_id"]),
            record=record,
        )

    def _load_journal(self) -> None:
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            write = self._deserialize(line)
            if write.sequence != len(self._writes):
                raise ValueError("memory journal sequence is not append-only/contiguous")
            self._writes.append(write)
            self._records[write.record_id] = write.record

    def _append(self, operation: str, record: MemoryRecord) -> MemoryWrite:
        write = MemoryWrite(
            sequence=len(self._writes),
            operation=operation,
            record_id=record.id,
            record=record,
        )
        self._writes.append(write)
        self._records[record.id] = record
        if self.journal_path is not None:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(self._serialize(write) + "\n")
        return write

    def write(self, record: MemoryRecord) -> MemoryWrite:
        existing = self._records.get(record.id)
        if existing is not None and existing != record:
            raise ValueError("memory ids are immutable; use supersede/deprecate instead of overwrite")
        if existing is not None:
            return self._writes[-1] if self._writes else self._append("write", record)
        return self._append("write", record)

    def deprecate(self, record_id: str, *, reason_ref: str) -> MemoryWrite:
        record = self._records[record_id]
        updated = replace(
            record,
            status=MemoryStatus.DEPRECATED,
            evidence_refs=tuple(dict.fromkeys((*record.evidence_refs, reason_ref))),
        )
        return self._append("deprecate", updated)

    def promote_episode(
        self,
        record_id: str,
        *,
        new_id: str,
        destination: MemoryKind,
        verification_ref: str,
        verification_confidence: float,
        independence_score: float,
    ) -> MemoryWrite:
        if destination not in {MemoryKind.SEMANTIC, MemoryKind.PROCEDURAL}:
            raise ValueError("episode promotion destination must be semantic or procedural")
        source = self._records[record_id]
        if source.kind is not MemoryKind.EPISODIC:
            raise ValueError("only episodic records may be promoted by this operation")
        if not 0.0 <= verification_confidence <= 1.0:
            raise ValueError("verification_confidence must be in [0, 1]")
        if not 0.0 <= independence_score <= 1.0:
            raise ValueError("independence_score must be in [0, 1]")
        if verification_confidence < 0.80 or independence_score < 0.50:
            raise RuntimeError("memory promotion requires strong, meaningfully independent verification")

        # Verification can justify promotion but does not erase source provenance. The
        # stored confidence is bounded by the strongest explicit epistemic support.
        confidence = min(1.0, max(source.confidence, verification_confidence * independence_score))
        promoted = MemoryRecord(
            id=new_id,
            kind=destination,
            payload=source.payload,
            confidence=confidence,
            provenance=source.provenance,
            evidence_refs=tuple(dict.fromkeys((*source.evidence_refs, verification_ref))),
            source_refs=tuple(dict.fromkeys((*source.source_refs, source.id))),
            tags=source.tags,
            utility=source.utility,
            verified=True,
            independent_verifications=max(1, source.independent_verifications + 1),
        )
        return self._append("promote", promoted)

    def retrieve(
        self,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        tags: Iterable[str] = (),
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> tuple[MemoryRecord, ...]:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if limit <= 0:
            raise ValueError("limit must be positive")
        kind_set = set(kinds) if kinds is not None else None
        tag_set = set(tags)
        candidates = [
            record
            for record in self._records.values()
            if record.status is MemoryStatus.ACTIVE
            and record.confidence >= min_confidence
            and (kind_set is None or record.kind in kind_set)
            and (not tag_set or tag_set.intersection(record.tags))
        ]
        candidates.sort(key=lambda item: (item.utility, item.confidence), reverse=True)
        return tuple(candidates[:limit])


@dataclass(frozen=True, slots=True)
class BeliefRevision:
    sequence: int
    belief: Belief
    reason_ref: str


class BeliefStateStore:
    """Current model of reality with versioned, contradiction-aware revisions."""

    def __init__(self) -> None:
        self._current: dict[str, Belief] = {}
        self._history: list[BeliefRevision] = []

    @property
    def current(self) -> tuple[Belief, ...]:
        return tuple(self._current.values())

    @property
    def history(self) -> tuple[BeliefRevision, ...]:
        return tuple(self._history)

    def get(self, belief_id: str) -> Belief:
        return self._current[belief_id]

    def put(self, belief: Belief, *, reason_ref: str) -> BeliefRevision:
        if not 0.0 <= belief.confidence <= 1.0:
            raise ValueError("belief confidence must be in [0, 1]")
        prior = self._current.get(belief.id)
        if prior is not None:
            same_evidence = set(belief.evidence_refs) <= set(prior.evidence_refs)
            if belief.confidence > prior.confidence and same_evidence:
                raise RuntimeError("belief confidence cannot increase without new evidence")
        revision = BeliefRevision(len(self._history), belief, reason_ref)
        self._history.append(revision)
        self._current[belief.id] = belief
        return revision

    def apply_confidence_update(
        self,
        belief_id: str,
        *,
        posterior_confidence: float,
        evidence_ref: str,
        contradiction: bool = False,
    ) -> BeliefRevision:
        prior = self._current[belief_id]
        evidence_refs = tuple(dict.fromkeys((*prior.evidence_refs, evidence_ref)))
        contradiction_refs = prior.contradiction_refs
        status = prior.status
        if contradiction:
            contradiction_refs = tuple(dict.fromkeys((*prior.contradiction_refs, evidence_ref)))
            status = "contradicted" if posterior_confidence < 0.25 else prior.status
        revised = replace(
            prior,
            confidence=posterior_confidence,
            evidence_refs=list(evidence_refs),
            contradiction_refs=list(contradiction_refs),
            status=status,
        )
        return self.put(revised, reason_ref=evidence_ref)

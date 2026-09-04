from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from leviathan.memory_ecology import (
    BeliefStateStore,
    MemoryEcology,
    MemoryKind,
    MemoryRecord,
)
from leviathan.types import Belief, Provenance, ProvenanceKind, UncertaintyKind


class MemoryEcologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provenance = Provenance(
            kind=ProvenanceKind.REAL_OBSERVATION,
            source_id="obs-1",
            trust_prior=0.9,
        )

    def test_semantic_memory_cannot_be_written_unverified(self) -> None:
        with self.assertRaises(ValueError):
            MemoryRecord(
                id="fact-1",
                kind=MemoryKind.SEMANTIC,
                payload={"fact": "x"},
                confidence=0.8,
                provenance=self.provenance,
                verified=False,
            )

    def test_episode_promotes_only_with_independent_verification(self) -> None:
        memory = MemoryEcology()
        memory.write(
            MemoryRecord(
                id="ep-1",
                kind=MemoryKind.EPISODIC,
                payload={"result": "worked"},
                confidence=0.6,
                provenance=self.provenance,
                tags=("diagnosis",),
            )
        )
        with self.assertRaises(RuntimeError):
            memory.promote_episode(
                "ep-1",
                new_id="fact-1",
                destination=MemoryKind.SEMANTIC,
                verification_ref="self-check",
                verification_confidence=0.95,
                independence_score=0.1,
            )
        write = memory.promote_episode(
            "ep-1",
            new_id="fact-1",
            destination=MemoryKind.SEMANTIC,
            verification_ref="measurement-2",
            verification_confidence=0.95,
            independence_score=0.9,
        )
        self.assertTrue(write.record.verified)
        self.assertEqual(write.record.kind, MemoryKind.SEMANTIC)
        self.assertIn("ep-1", write.record.source_refs)

    def test_append_only_journal_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.jsonl"
            memory = MemoryEcology(path)
            memory.write(
                MemoryRecord(
                    id="ep-1",
                    kind=MemoryKind.EPISODIC,
                    payload={"state": 1},
                    confidence=0.7,
                    provenance=self.provenance,
                    utility=0.8,
                )
            )
            memory.promote_episode(
                "ep-1",
                new_id="fact-1",
                destination=MemoryKind.SEMANTIC,
                verification_ref="verifier-1",
                verification_confidence=0.9,
                independence_score=0.8,
            )
            restored = MemoryEcology(path)
            self.assertEqual(len(restored.writes), 2)
            self.assertEqual({record.id for record in restored.records}, {"ep-1", "fact-1"})

    def test_retrieval_filters_by_type_tag_confidence_and_utility(self) -> None:
        memory = MemoryEcology()
        for record in (
            MemoryRecord(
                "a", MemoryKind.EPISODIC, "a", 0.9, self.provenance, tags=("code",), utility=0.2
            ),
            MemoryRecord(
                "b", MemoryKind.EPISODIC, "b", 0.8, self.provenance, tags=("code",), utility=0.9
            ),
            MemoryRecord(
                "c", MemoryKind.EPISODIC, "c", 0.95, self.provenance, tags=("biology",), utility=1.0
            ),
        ):
            memory.write(record)
        found = memory.retrieve(tags=("code",), min_confidence=0.75, limit=2)
        self.assertEqual([record.id for record in found], ["b", "a"])

    def test_belief_state_is_current_state_not_history_and_needs_new_evidence_to_gain_confidence(self) -> None:
        store = BeliefStateStore()
        initial = Belief(
            id="b1",
            value="valve stuck",
            confidence=0.5,
            provenance=self.provenance,
            uncertainty=UncertaintyKind.EPISTEMIC,
            evidence_refs=["obs-1"],
        )
        store.put(initial, reason_ref="obs-1")
        with self.assertRaises(RuntimeError):
            store.put(
                Belief(
                    id="b1",
                    value="valve stuck",
                    confidence=0.8,
                    provenance=self.provenance,
                    uncertainty=UncertaintyKind.EPISTEMIC,
                    evidence_refs=["obs-1"],
                ),
                reason_ref="no-new-evidence",
            )
        store.apply_confidence_update(
            "b1",
            posterior_confidence=0.8,
            evidence_ref="measurement-2",
        )
        self.assertEqual(len(store.current), 1)
        self.assertEqual(len(store.history), 2)
        self.assertAlmostEqual(store.get("b1").confidence, 0.8)


if __name__ == "__main__":
    unittest.main()

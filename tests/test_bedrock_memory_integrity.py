from __future__ import annotations
from unittest.mock import patch
import unittest
import tempfile
from leviathan.memory_ecology import MemoryEcology,MemoryKind,MemoryRecord,BeliefStateStore
from leviathan.types import Belief,Provenance,ProvenanceKind,MetaState
from leviathan.cognitive_kernel import CognitiveCompiler,LeviathanCognitiveKernel,GoalState,NodeStatus

class BedrockMemoryIntegrityTests(unittest.TestCase):
    def setUp(self):self.prov=Provenance(ProvenanceKind.DETERMINISTIC_EXECUTION,"test",1.)
    def test_verified_payload_cannot_change_by_alias(self):
        memory=MemoryEcology();payload={"rule":[1,2]}
        rec=MemoryRecord("e",MemoryKind.EPISODIC,payload,.5,self.prov)
        memory.write(rec);payload["rule"][0]=99
        public=memory.records;public[0].payload["rule"][0]=88
        self.assertEqual(memory.records[0].payload["rule"],[1,2])
    def test_repeat_write_returns_correct_record(self):
        m=MemoryEcology();a=MemoryRecord("a",MemoryKind.EPISODIC,1,.5,self.prov)
        b=MemoryRecord("b",MemoryKind.EPISODIC,2,.5,self.prov)
        m.write(a);m.write(b)
        self.assertEqual(m.write(a).record_id,"a")
    def test_promotion_does_not_overwrite_existing_id(self):
        m=MemoryEcology();m.write(MemoryRecord("e",MemoryKind.EPISODIC,1,.5,self.prov))
        with self.assertRaises(ValueError):m.promote_episode("e",new_id="e",destination=MemoryKind.PROCEDURAL,
            verification_ref="v",verification_confidence=1.,independence_score=1.)
    def test_failed_storage_does_not_promote_in_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=MemoryEcology(tmp+"/journal")
            with patch("pathlib.Path.open",side_effect=OSError("disk full")):
                with self.assertRaises(OSError):m.write(MemoryRecord("e",MemoryKind.EPISODIC,1,.5,self.prov))
            self.assertEqual(m.records,());self.assertEqual(m.writes,())
    def test_belief_history_is_snapshot_not_alias(self):
        m=BeliefStateStore();b=Belief("b",{"v":1},.5,self.prov)
        m.put(b,reason_ref="first");b.confidence=1.;b.value["v"]=99
        read=m.get("b");read.confidence=1.
        self.assertEqual(m.get("b").confidence,.5);self.assertEqual(m.history[0].belief.value,{"v":1})
    def program(self):
        kernel=LeviathanCognitiveKernel("one")
        state=MetaState("test","goal",.5,.5,0.,0.,0.,.5,.5)
        return kernel.compile_problem(problem="test",task_type="test",goal=GoalState("test"),state=state)
    def test_one_episode_does_not_become_eight_successes(self):
        compiler=CognitiveCompiler();program,_=self.program()
        for _ in range(8):skill=compiler.observe(program,episode_id="same",verified_success=True)
        self.assertEqual(skill.trials,1);self.assertFalse(skill.ready_to_compile())
        with self.assertRaises(ValueError):compiler.observe(program,episode_id="same",verified_success=False)
    def test_failed_dependency_blocks_all_descendants(self):
        _,graph=self.program();root=graph.ready()[0].instruction.id
        graph.start(root);graph.fail(root,error="test")
        self.assertTrue(graph.complete)
        self.assertTrue(all(n.status in {NodeStatus.FAILED,NodeStatus.BLOCKED} for n in graph.nodes.values()))

if __name__=="__main__":unittest.main()

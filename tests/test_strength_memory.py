import unittest,tempfile
from leviathan.strength.contracts import *
from leviathan.strength.programs import *
from leviathan.strength.memory import SkillLibrary
from leviathan.bedrock.contracts import Outcome,stable_hash

class StrengthMemoryTests(unittest.TestCase):
    def setUp(self):
        self.p=parse('rot90(x)');self.g=((1,2),(3,4))
        self.task=ArcTask('world',(Example(self.g,self.p.run(self.g)),),(self.g,),split='synthetic')
    def test_memory_reload_and_current_evidence_filter(self):
        with tempfile.TemporaryDirectory() as d:
            m=SkillLibrary(d+'/m.jsonl');m.remember_candidate(self.task,self.p)
            n=SkillLibrary(d+'/m.jsonl')
            self.assertIn(self.p,n.retrieve(self.task))
            wrong=ArcTask('different',(Example(self.g,self.g),),(self.g,))
            self.assertEqual(n.retrieve(wrong),())
    def test_no_false_trusted_promotion(self):
        m=SkillLibrary();sid=m.remember_candidate(self.task,self.p)
        self.assertFalse(m.memory.records[0].verified)
        with self.assertRaises(ValueError):m.promote(sid,Outcome('wrong','host',True,'e','scope',True))
    def test_dependency_invalidation(self):
        m=SkillLibrary();a=m.remember_candidate(self.task,self.p)
        t=ArcTask('world2',(Example(self.g,self.p.run(self.g)),),(self.g,))
        p=parse('rot270(rot180(x))');b=m.remember_candidate(t,p,dependencies=(a,))
        self.assertEqual(set(m.invalidate(a,evidence_id='counterexample')),set((a,b)))
        self.assertFalse(m.retrieve(self.task))
    def test_macro_expansion_exact(self):
        macro=parse('rot90(crop(x, 0))');arg=parse('flip_lr(x)')
        expanded=SkillLibrary.expand(macro,arg)
        self.assertEqual(expanded.run(self.g),macro.run(arg.run(self.g)))
    def test_no_duplicate_task_inflation(self):
        m=SkillLibrary();p=parse('rot270(rot180(x))')
        for _ in range(4):m.remember_candidate(self.task,p)
        self.assertEqual(len(m.memory.records),1);self.assertFalse(m.macros())
    def test_neural_state_is_snapshot(self):
        m=SkillLibrary();state={'vectors':[[1,2]]}
        m.remember_candidate(self.task,self.p,neural_state=state);state['vectors'][0][0]=99
        self.assertEqual(m.memory.records[0].payload['neural_state']['vectors'],[[1,2]])

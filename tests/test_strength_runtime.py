import unittest
from leviathan.strength.contracts import *
from leviathan.strength.runtime import StrengthRuntime
from leviathan.strength.programs import parse
class StrengthRuntimeTests(unittest.TestCase):
    def test_actual_executed_graph_and_unknown_test_outcome(self):
        p=parse('rot90(x)');g=((1,2),(3,4));h=((3,2),(1,4))
        t=ArcTask('test',(Example(g,p.run(g)),),(h,))
        rt=StrengthRuntime(model_id='one',config=SearchConfig(max_depth=1,max_candidates=2000,neural_rounds=0))
        r=rt.solve_arc(t)
        self.assertTrue(r['graph_complete']);self.assertIn(p.run(h),r['attempts'][0])
        self.assertEqual(r['model_id'],'one');self.assertFalse(rt.memory.records[-1].verified)
        self.assertEqual(rt.memory.records[-1].payload['query_outcome'],'unknown')
    def test_evaluation_memory_mutation_refused(self):
        g=((1,2),(3,4));t=ArcTask('eval',(Example(g,g),),(g,),split='evaluation')
        r=StrengthRuntime(model_id='one',config=SearchConfig(max_depth=1,max_candidates=200,neural_rounds=0))
        with self.assertRaises(ValueError):r.solve_arc(t,remember_support=True)

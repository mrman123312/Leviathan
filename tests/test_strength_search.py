import unittest
from dataclasses import replace
from leviathan.strength.contracts import *
from leviathan.strength.programs import *
from leviathan.strength.search import *

A=((0,0,0,0),(0,1,2,0),(0,3,0,0),(0,0,0,0))
B=((0,0,0,0,0),(0,2,2,0,0),(0,1,0,0,0),(0,0,0,0,0))
C=((0,0,0,0),(0,3,2,0),(0,0,1,0),(0,0,0,0))

def task(p,queries=(C,)):
    return ArcTask('synthetic-composition',tuple(Example(g,p.run(g)) for g in (A,B)),queries)
CFG=SearchConfig(max_depth=2,max_candidates=4000,max_seconds=15,neural_rounds=0)

class StrengthSearchTests(unittest.TestCase):
    def test_composes_crop_rotation(self):
        p=parse('rot90(crop(x, 0))');t=task(p)
        r=StrengthSearch(CFG).solve(t)
        self.assertIn(p.run(C),r['attempts'][0]);self.assertFalse(r['query_labels_received'])
    def test_joint_color_substitution(self):
        p=parse('recolor(rot90(x), 1, 4)');t=task(p)
        r=StrengthSearch(CFG).solve(t)
        self.assertIn(p.run(C),r['attempts'][0])
    def test_exact_demos_required(self):
        t=ArcTask('inconsistent',(Example(A,A),Example(A,B)),(C,))
        r=StrengthSearch(CFG).solve(t)
        self.assertEqual(r['status'],'abstained');self.assertFalse(r['attempts'][0])
    def test_counterexample_witness_and_repeat(self):
        ledger=Counterexamples();p=Program();ex=(Example(((0,),),((4,),)),)
        w=ledger.register(p,(((0,),),),ex)
        self.assertEqual(w.first_wrong_cell,(0,0,0,4));self.assertIn(p.id,ledger.rejected)
    def test_wrong_complete_program_can_be_a_valid_prefix(self):
        # crop alone fails; rot90(crop(...)) is necessary. Don't ban the prefix.
        p=parse('rot90(crop(x, 0))');r=StrengthSearch(CFG).solve(task(p))
        self.assertTrue(r['attempts'][0]);self.assertGreater(r['budget']['candidates'],1)
    def test_bad_neural_guidance_does_not_replace_grammar(self):
        class Wrong:
            is_neural=False;last_calls=1
            def propose(self,*args,**kwargs):return ['rot90(x)']*4
        p=parse('flip_lr(crop(x, 0))')
        r=StrengthSearch(replace(CFG,neural_rounds=1),Wrong()).solve(task(p))
        self.assertIn(p.run(C),r['attempts'][0]);self.assertGreater(r['budget']['duplicate_programs'],0)
        self.assertFalse(r['neural_model_used'])
    def test_wrong_proposal_repaired_by_composition(self):
        class Partial:
            is_neural=False;last_calls=1
            def propose(self,*args,**kwargs):return ['crop(x, 0)']
        p=parse('rot90(crop(x, 0))')
        r=StrengthSearch(replace(CFG,neural_rounds=1),Partial()).solve(task(p))
        self.assertIn(p.run(C),r['attempts'][0]);self.assertTrue(r['neural_proposals'][0]['witness'])
    def test_no_progress_contract(self):
        c=ProgressController();self.assertTrue(c.allow('neural','same'));self.assertFalse(c.allow('neural','same'))
        self.assertTrue(c.allow('reframe','same'))
    def test_two_attempt_cap_and_whole_grid(self):
        r=StrengthSearch(CFG).solve(task(Program()))
        self.assertLessEqual(len(r['attempts'][0]),2)
        self.assertTrue(all(isinstance(g,tuple) for g in r['attempts'][0]))
    def test_search_is_deterministic_with_fixed_budget(self):
        t=task(parse('rot90(x)'))
        a=StrengthSearch(CFG).solve(t);b=StrengthSearch(CFG).solve(t)
        self.assertEqual(a['attempts'],b['attempts']);self.assertEqual(a['selected_programs'],b['selected_programs'])
    def test_inverse_goal_junction(self):
        p=parse('rot90(crop(x, 0))')
        r=StrengthSearch(CFG).solve(task(p))
        self.assertIn('inverse_goal_join',r['selected_sources'])

class BinaryCompositionTests(unittest.TestCase):
    def test_grid_product(self):
        p=parse('kronecker(x, x, 0)');a=((1,0),(0,1))
        self.assertEqual(p.run(a),((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)))
    def test_binary_roundtrip_and_input_substitution(self):
        p=parse('merge(x, flip_lr(x), 0)');self.assertEqual(parse(str(p)),p)
        q=p.replace_input(parse('rot90(x)'))
        g=((1,0,0),(0,2,0),(0,0,0))
        self.assertEqual(q.run(g),p.run(parse('rot90(x)').run(g)))
    def test_conflicting_merge_rejected(self):
        with self.assertRaises(ValueError):parse('merge(x, flip_lr(x), 0)').run(((1,2),))

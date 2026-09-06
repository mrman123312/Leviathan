import unittest
from unittest.mock import patch
from leviathan.strength.contracts import ArcTask,Example
from leviathan.strength.programs import parse
from leviathan.strength.search import CompiledGuidance,description_cost
from leviathan.strength.memory import SkillLibrary
from leviathan.strength.evaluation import score
import importlib.util
HAS=importlib.util.find_spec('torch') is not None

class StrengthGateTests(unittest.TestCase):
    def test_guidance_traverses_both_operands(self):
        p=parse('merge(rot90(x), flip_lr(crop(x, 0)), 0)')
        guide=CompiledGuidance();guide.add(p)
        self.assertGreater(guide.transitions[('crop','flip_lr')],0)
        self.assertGreater(guide.transitions[('flip_lr','merge')],0)
    def test_interpreted_source_pays_description_complexity(self):
        p=parse('def transform(x):\n    return [[x[r][c] for r in range(len(x))] for c in range(len(x[0]))]')
        self.assertGreater(description_cost(p),description_cost(parse('transpose(x)')))
    def test_malformed_empty_evaluation_is_not_perfect(self):
        with self.assertRaises(ValueError):score({}, {}, [])
        with self.assertRaises(ValueError):score({}, {'t':[]}, ['t'])
    def test_neural_coalition_evidence_stored_without_false_trust(self):
        g=((1,2),(3,4));p=parse('rot90(x)');task=ArcTask('world',(Example(g,p.run(g)),),(g,))
        library=SkillLibrary(model_revision='fixed')
        evidence={'trials':[{'route':{'kind':'cell_ablation','layer':1,'cell_start':0},'accepted':True}]}
        library.remember_candidate(task,p,intervention_evidence=evidence)
        changed=((2,3),(1,4));new=ArcTask('new',(Example(changed,p.run(changed)),),(changed,))
        self.assertEqual(len(library.compatible_interventions(new)),1)
        self.assertEqual(library.compatible_interventions(task),[])
        self.assertFalse(library.memory.records[0].verified)

@unittest.skipUnless(HAS,'Optional PyTorch')
class StrengthNeuralGateTests(unittest.TestCase):
    def test_every_latent_slot_receives_separate_gate(self):
        import torch
        from test_bedrock_neural import TinyLM
        from leviathan.bedrock.neural import FrozenExecutor
        from leviathan.strength.neural import NeuralFabric,TaskWorkspace
        fabric=NeuralFabric(FrozenExecutor(TinyLM(),model_id='one',revision='fixed'))
        ids=torch.tensor([[1,2,3]])
        workspace=TaskWorkspace('fixed',1,'source',torch.randn(3,16))
        def nll(ids,pos,route,workspace=None,slot=0):
            return 1. if route.kind!='task_state' else {0:1.2,1:.8,2:.9}[slot]
        with patch.object(fabric,'nll',side_effect=nll):
            route,report=fabric.select_on_demonstrations([(ids,1),(ids,1)],workspace=workspace)
        self.assertEqual(route.kind,'task_state');self.assertEqual(report['selected_slot'],1)
        self.assertEqual(len([t for t in report['trials'] if t['route']['kind']=='task_state']),3)
        self.assertFalse(report['accuracy_gain_proven'])
    def test_direct_grid_format_has_no_guessing(self):
        from leviathan.strength.proposer import parse_grid_completion
        self.assertEqual(parse_grid_completion('12\n34'),((1,2),(3,4)))
        self.assertEqual(parse_grid_completion('[[1,2],[3,4]]'),((1,2),(3,4)))
        for bad in ('the answer is 12\n34','123\n45','[[true]]','[[-1]]'):
            with self.assertRaises(ValueError):parse_grid_completion(bad)

if __name__=='__main__':unittest.main()

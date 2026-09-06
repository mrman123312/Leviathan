"""New controls/cells/world integration. Test fixtures are not pretrained scores."""
import unittest
import importlib.util
from dataclasses import replace
from unittest.mock import patch
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from test_bedrock_neural import TinyLM,FFN
    from leviathan.bedrock.decisions import (StopPolicy,summarize,compare,initial_stop,stable_stop,
        ChoicePolicy,choose_next)
    from leviathan.bedrock.stable_neural import StableFrozenExecutor,StableFrozenPolicy,branch_target
    from leviathan.bedrock.activation_cells import ActivationCellBank,ActivationPolicy
    from leviathan.bedrock.neural_discovery import parse_expression,discover_neural
    from leviathan.bedrock.runtime import BedrockRuntime

@unittest.skipUnless(HAS_TORCH,'PyTorch optional')
class ExpressiveControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(186);torch.set_num_threads(1)
        self.model=TinyLM().eval()
        self.engine=StableFrozenExecutor(self.model,model_id='one',revision='test')
        self.ids=torch.tensor([[1,3,4,9]])
    def test_identical_predictions_are_stable(self):
        logits=torch.randn(3,32);prior=summarize(logits)
        _,signals=compare(prior,logits)
        self.assertTrue(stable_stop(signals,torch.zeros(3),StopPolicy()).all())
    def test_softmax_offset_is_not_fake_novelty(self):
        logits=torch.randn(3,32);prior=summarize(logits)
        _,signals=compare(prior,logits+5)
        self.assertLess(float(signals['coarse_js'].abs().max()),1e-6)
        self.assertLess(float(signals['max_top_logprob_change'].max()),1e-5)
    def test_changed_winner_does_not_halt(self):
        logits=torch.tensor([[8.,0.,-2.]])
        _,signals=compare(summarize(logits),torch.tensor([[0.,8.,-2.]]))
        self.assertFalse(stable_stop(signals,torch.zeros(1),StopPolicy()).any())
    def test_confident_versus_uncertain_initial_exit(self):
        p=StopPolicy()
        self.assertTrue(initial_stop(summarize(torch.tensor([[20.,0.,0.]])),p).all())
        self.assertFalse(initial_stop(summarize(torch.zeros(1,3)),p).any())
    def test_prediction_stop_can_remove_actual_layer_calls(self):
        # Exact equality fixture: zero update radius makes each attempted innovation neutral.
        policy=StableFrozenPolicy(start=1,end=3,passes=4,gain=.1,relative_radius=0.,
            prediction_stop=StopPolicy(initial_pmax=1.,initial_margin=1e6))
        out=self.engine.run(self.ids,policy=policy).logits
        trace=self.engine.last_trace
        self.assertEqual(trace['passes_executed'],2)
        self.assertEqual(trace['extra_layer_calls'],3)
        self.assertTrue(trace['prediction_steps'][0]['halted_positions']>0)
        self.assertTrue(torch.equal(out,self.engine.run(self.ids).logits))
    def test_predictive_stop_is_prefix_causal(self):
        p=StableFrozenPolicy(start=1,end=3,passes=4,gain=.04,prediction_stop=StopPolicy())
        full=self.engine.run(self.ids,policy=p).logits
        early=self.engine.run(self.ids[:,:2],policy=p).logits
        torch.testing.assert_close(full[:,:2],early,atol=2e-6,rtol=2e-6)
    def test_middle_layer_logit_lens_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.run(self.ids,policy=StableFrozenPolicy(end=2,gain=.1,passes=2,prediction_stop=StopPolicy()))
    def test_opposite_latent_branches_change_routes_without_new_weights(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        a=StableFrozenPolicy(start=1,end=3,passes=2,gain=.1,branch_direction='orthogonal_context',branch_mix=1.)
        out1=self.engine.run(self.ids,policy=a).logits
        out2=self.engine.run(self.ids,policy=replace(a,branch_sign=-1)).logits
        self.assertFalse(torch.equal(out1,out2))
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in before.items()))
    def test_context_branch_has_no_future_input_dependency(self):
        entry=torch.randn(1,5,16);state=entry+.1*torch.randn_like(entry)
        p=StableFrozenPolicy(branch_direction='causal_context',branch_mix=.5)
        torch.testing.assert_close(branch_target(entry,state,p)[:,:3],
            branch_target(entry[:,:3],state[:,:3],p),atol=0,rtol=0)
    def test_task_controller_has_direct_refine_explore_paths(self):
        self.assertEqual(choose_next([0.,-8.,-9.])['action'],'DIRECT')
        self.assertEqual(choose_next([0.,-.1,-.2])['action'],'REFINE')
        self.assertEqual(choose_next([0.,-.1,-.2],previous=[0.,-.1,-.2])['action'],'STOP_REFINE')
        self.assertEqual(choose_next([-.2,0.,-.1],previous=[0.,-.1,-.2])['action'],'EXPLORE')
    def test_task_controller_never_accepts_gold_parameter(self):
        with self.assertRaises(TypeError):choose_next([1.,2.],answerKey='B')
    def test_nan_signals_rejected(self):
        with self.assertRaises(ValueError):choose_next([float('nan'),1.])
        with self.assertRaises(FloatingPointError):summarize(torch.tensor([[float('inf'),1.]]))
    def test_half_precision_branches_finite_in_synthetic_fixture(self):
        model=TinyLM().half().eval();engine=StableFrozenExecutor(model,model_id='half',revision='test')
        p=StableFrozenPolicy(gain=.06,passes=4,prediction_stop=StopPolicy(),
                             branch_direction='causal_context',branch_mix=.4)
        out=engine.run(self.ids,policy=p).logits
        self.assertTrue(torch.isfinite(out).all())

@unittest.skipUnless(HAS_TORCH,'PyTorch optional')
class ActivationRelevanceTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(718);torch.set_num_threads(1)
    def test_observe_uses_actual_activations_and_preserves_donor(self):
        donor=FFN().eval();bank=ActivationCellBank(donor,8);x=torch.randn(6,16)
        with torch.inference_mode():
            out,trace=bank.analyze(x,ActivationPolicy(width=8,seed=2,max_cells=2))
            torch.testing.assert_close(out,donor(x),atol=0,rtol=0)
        self.assertGreater(trace['median_bound_tightening_factor'],1)
        self.assertTrue(trace['dense_gate_up_computed'])
        self.assertTrue(trace['dense_down_audit_computed'])
        self.assertFalse(trace['speedup_claim'])
    def test_bound_contains_actual_excluded_cells(self):
        for seed in range(10):
            torch.manual_seed(seed);donor=FFN();bank=ActivationCellBank(donor,8)
            with torch.inference_mode():
                _,t=bank.analyze(torch.randn(5,16),ActivationPolicy(width=8,seed=1,max_cells=1))
            self.assertLessEqual(t['max_measured_excluded_output_l2'],t['max_activation_tail_bound']+1e-5)
    def test_unmet_bound_falls_back_without_forcing_sparsity(self):
        donor=FFN();bank=ActivationCellBank(donor,8);x=torch.randn(2,16)
        with torch.inference_mode():
            out,t=bank.analyze(x,ActivationPolicy(width=8,seed=1,max_cells=1,mode='bounded'))
            torch.testing.assert_close(out,donor(x),atol=0,rtol=0)
        self.assertEqual(t['dense_fallback_tokens'],2)
    def test_all_cells_zero_tail_with_bias_once(self):
        donor=FFN();bank=ActivationCellBank(donor,8);x=torch.randn(4,16)
        with torch.inference_mode():
            out,t=bank.analyze(x,ActivationPolicy(width=8,seed=4,max_cells=4,mode='bounded'))
            torch.testing.assert_close(out,donor(x),atol=1e-6,rtol=1e-6)
        self.assertEqual(t['max_activation_tail_bound'],0)
    def test_analysis_cannot_silently_sever_gradients(self):
        with self.assertRaises(RuntimeError):ActivationCellBank(FFN(),8).analyze(torch.randn(2,16),ActivationPolicy(width=8))

@unittest.skipUnless(HAS_TORCH,'PyTorch optional')
class NeuralWorldBridgeTests(unittest.TestCase):
    def test_parser_rejects_code_execution_and_foreign_names(self):
        for text in ("__import__('os').system('echo no')",'y+1','x.__class__','open(x)','2**1000000','lambda x:x'):
            with self.assertRaises((ValueError,SyntaxError)):parse_expression(text,tuple(range(7)))
    def test_safe_expression_compiles_to_existing_ast(self):
        rule=parse_expression('(3*x+2)%11',tuple(range(11)))
        self.assertEqual(rule.predict(4),3)
    def test_neural_proposal_counterexample_revision_compilation_and_transfer(self):
        runtime=BedrockRuntime(model_id='one');seen=[]
        def proposer(domain,observations,rejections,task):
            self.assertIsInstance(observations,tuple)
            self.assertNotIn('hidden',str(domain))
            seen.append(observations)
            return [{'text':'6*x' if len(seen)==1 else 'x*x'}]
        result=discover_neural(runtime,scope='square',problem='infer device',domain=tuple(range(7)),
                               observe=lambda a:a*a,proposer=proposer)
        self.assertEqual(result['status'],'empirically_validated_candidate')
        self.assertEqual(len(seen),2)
        self.assertIn((1,1),[(a,b) for a,b,_ in seen[1]])
        self.assertFalse(result['catalogue_fallback_used'])
        self.assertEqual(result['proposal_source'],'injected_test_proposer')
        self.assertEqual(result['neural_model_calls'],0)
        self.assertEqual(runtime.transfer(scope='square',action=5)['prediction'],25)
        self.assertTrue(result['graph_complete'])
    def test_malformed_output_does_not_become_fake_success(self):
        runtime=BedrockRuntime(model_id='one')
        result=discover_neural(runtime,scope='bad',problem='device',domain=tuple(range(7)),
            observe=lambda a:a*a,proposer=lambda *a:[{'text':'hello world'}],max_rounds=1)
        self.assertEqual(result['status'],'not_solved')
        with self.assertRaises(KeyError):runtime.transfer(scope='bad',action=4)
    def test_invalid_modulus_rejected_on_domain(self):
        with self.assertRaises(ValueError):parse_expression('x % 0',tuple(range(7)))
    def test_requires_one_bound_model_for_actual_proposer(self):
        with self.assertRaises(ValueError):discover_neural(BedrockRuntime(model_id='one'),scope='a',
            problem='x',domain=tuple(range(7)),observe=lambda a:a)


@unittest.skipUnless(HAS_TORCH,'PyTorch optional')
class EvaluationBoundaryTests(unittest.TestCase):
    def test_adaptive_stages_execute_and_charge_every_call(self):
        from leviathan.bedrock.evaluation import adaptive_choice
        calls=[]
        def evaluate(name):
            calls.append(name)
            scores={'donor':[0.,-.1,-.2],'refine_2':[-.2,0.,-.1],'explore_4':[-.4,0.,-.5]}[name]
            return {'scores':scores,'model_seconds':2.,'model_calls':3,'extra_layer_calls':4,
                    'nonfinite_replay_fallbacks':0,'peak_allocated_gib':1.}
        r=adaptive_choice(evaluate)
        self.assertEqual(calls,['donor','refine_2','explore_4'])
        self.assertEqual(r['model_seconds'],6.);self.assertEqual(r['model_calls'],9)
        self.assertTrue(r['standalone_compute_fully_charged'])
    def test_confident_adaptive_does_not_execute_extra_model_pass(self):
        from leviathan.bedrock.evaluation import adaptive_choice
        calls=[]
        def evaluate(name):
            calls.append(name)
            return dict(scores=[0.,-8.],model_seconds=1.,model_calls=2,extra_layer_calls=0,
                        nonfinite_replay_fallbacks=0,peak_allocated_gib=None)
        self.assertEqual(adaptive_choice(evaluate)['selected_regime'],'donor')
        self.assertEqual(calls,['donor'])
    def test_grade_and_normalizations_separate(self):
        from leviathan.bedrock.evaluation import grade
        s=dict(scores=[-2.,-3.],token_counts=[1,4],character_counts=[10,2])
        a=grade(s,['A','B'],'A')
        self.assertTrue(a['correct']);self.assertFalse(a['token_normalized_correct'])
        self.assertTrue(a['character_normalized_correct'])
    def test_failed_examples_never_look_like_full_score(self):
        from leviathan.bedrock.evaluation import summarize_arc
        s=summarize_arc([{'status':'error','id':'x'}])
        self.assertIsNone(s['accuracy']);self.assertEqual(s['errors'],1)
        self.assertFalse(s['automatic_promotion'])

if __name__=='__main__':unittest.main()

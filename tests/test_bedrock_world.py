from __future__ import annotations
from dataclasses import asdict
import tempfile
import unittest
from leviathan.bedrock.contracts import Budget,Meter,BudgetExceeded,CausalTrace,Competence,Outcome,stable_hash
from leviathan.bedrock.world import Expr,Rule,VersionSpace,Contradiction,catalogue,structural_counterfactual
from leviathan.bedrock.runtime import BedrockRuntime,Pulse

class BedrockWorldTests(unittest.TestCase):
    def test_ast_roundtrip_and_code_execution_rejected(self):
        rule=catalogue("affine_mod11")[47]
        other=Rule.parse(rule.as_dict())
        self.assertEqual([other.predict(x) for x in other.domain],[rule.predict(x) for x in rule.domain])
        with self.assertRaises(ValueError):Expr.parse({"op":"eval","args":["__import__('os')"]})
        with self.assertRaises(ValueError):Expr.parse({"op":"const","args":[float("nan")]})
    def test_domain_is_a_real_precondition(self):
        with self.assertRaises(ValueError):catalogue("affine_mod11")[0].predict(100)
    def test_information_gain_never_uses_unknown_labels(self):
        space=VersionSpace(catalogue("affine_mod11"))
        before=space.entropy
        a=space.choose(space.domain)
        truth=catalogue("affine_mod11")[17]
        space.observe(a,truth.predict(a),evidence_id="obs1")
        self.assertLess(space.entropy,before)
        self.assertIn(truth.as_dict(),[r.as_dict() for r in space.rules])
    def test_duplicate_evidence_does_not_multiply_certainty(self):
        space=VersionSpace(catalogue("affine_mod11"))
        space.observe(0,3,evidence_id="e")
        state=space.report()
        self.assertFalse(space.observe(0,3,evidence_id="e"))
        self.assertEqual(space.report(),state)
        with self.assertRaises(ValueError):space.observe(0,4,evidence_id="e")
    def test_out_of_grammar_is_explicit_failure(self):
        space=VersionSpace(catalogue("affine_mod11"))
        with self.assertRaises(Contradiction):space.observe(0,99,evidence_id="surprise")
        self.assertTrue(space.misspecified)
        with self.assertRaises(Contradiction):space.choose(space.domain)
    def test_three_declared_families_are_identifiable(self):
        for family in ("affine_mod11","bit_permutation","boolean_circuit"):
            rules=catalogue(family)
            for index in (0,len(rules)//2,len(rules)-1):
                truth=rules[index];space=VersionSpace(rules)
                while len(space.rules)>1:
                    a=space.choose(space.domain)
                    space.observe(a,truth.predict(a),evidence_id=f"{family}-{index}-{a}")
                self.assertEqual([space.predict(a) for a in truth.domain],[truth.predict(a) for a in truth.domain])
    def test_discovery_executes_kernel_and_persistent_transfer(self):
        truth=catalogue("affine_mod11")[47]
        with tempfile.TemporaryDirectory() as tmp:
            journal=tmp+"/memory.jsonl"
            first=BedrockRuntime(model_id="one",memory_journal=journal)
            result=first.discover(problem="infer unfamiliar device",scope="device:v1",
                rules=catalogue("affine_mod11"),observe=truth.predict)
            self.assertEqual(result["status"],"validated_in_declared_domain")
            self.assertTrue(result["graph_complete"])
            self.assertEqual(result["training_steps"],0)
            self.assertEqual(result["neural_calls"],0)
            task=first.tasks[result["session_id"]]
            kinds=[r["kind"] for r in task.trace.records.values()]
            self.assertIn("hypothesize",kinds);self.assertIn("execute",kinds);self.assertIn("update_belief",kinds)
            self.assertLessEqual(result["budget"]["used"]["environment_steps"],6)
            second=BedrockRuntime(model_id="one",memory_journal=journal)
            prediction=second.transfer(scope="device:v1",action=10,observe=truth.predict)
            self.assertTrue(prediction["verified_now"])
            self.assertEqual(prediction["prediction"],truth.predict(10))
            self.assertEqual(prediction["neural_calls"],0)
    def test_transfer_unknown_scope_and_counterexample_invalidation(self):
        core=BedrockRuntime(model_id="one")
        truth=catalogue("affine_mod11")[3]
        result=core.discover(problem="infer device",scope="v1",rules=catalogue("affine_mod11"),observe=truth.predict)
        with self.assertRaises(KeyError):core.transfer(scope="v2",action=5)
        wrong=core.transfer(scope="v1",action=5,observe=lambda a:999)
        self.assertTrue(wrong["skill_deprecated"])
        with self.assertRaises(KeyError):core.transfer(scope="v1",action=5)
    def test_prediction_before_each_actual_environment_call(self):
        core=BedrockRuntime(model_id="one");truth=catalogue("affine_mod11")[10]
        def observe(a):
            task=list(core.tasks.values())[-1]
            self.assertEqual(len(task.session.prediction_ids),len(task.session.action_ids))
            self.assertGreater(len(task.session.prediction_ids),0)
            return truth.predict(a)
        result=core.discover(problem="device",scope="v1",rules=catalogue("affine_mod11"),observe=observe)
        self.assertEqual(result["status"],"validated_in_declared_domain")
    def test_failure_is_not_promoted_to_trusted_skill(self):
        core=BedrockRuntime(model_id="one")
        result=core.discover(problem="novel impossible device",scope="bad",rules=catalogue("affine_mod11"),observe=lambda x:999)
        self.assertEqual(result["status"],"not_solved")
        with self.assertRaises(KeyError):core.transfer(scope="bad",action=0)
    def test_budget_is_enforced_before_environment_side_effect(self):
        core=BedrockRuntime(model_id="one");truth=catalogue("affine_mod11")[50];calls=[]
        def observe(x):calls.append(x);return truth.predict(x)
        result=core.discover(problem="device",scope="short",rules=catalogue("affine_mod11"),
                            observe=observe,budget=Budget(environment_steps=1))
        self.assertEqual(len(calls),1);self.assertEqual(result["status"],"not_solved")
    def test_unknown_outcome_is_not_success(self):
        stats=Competence();receipt=Outcome(stable_hash("x"),"host",None,"e","test",True)
        stats.record("test","method",receipt)
        self.assertEqual(stats.estimate("test","method")["successes"],0)
        with self.assertRaises(ValueError):Outcome("x","host","false","e","test",True)
    def test_competence_deduplicates_external_evidence(self):
        stats=Competence();receipt=Outcome(stable_hash("x"),"host",True,"e","test",True)
        stats.record("test","method",receipt);stats.record("test","method",receipt)
        self.assertEqual(stats.estimate("test","method")["successes"],1)
        self.assertFalse(stats.estimate("test","method")["calibrated"])
    def test_causal_dependency_invalidation_is_transitive(self):
        trace=CausalTrace();a=trace.add("belief",{});b=trace.add("plan",{},(a,));c=trace.add("act",{},(b,))
        self.assertEqual(trace.invalidate(a),(a,b,c))
        snapshot=trace.records;snapshot[a]["valid"]=True
        self.assertFalse(trace.records[a]["valid"])
    def test_structural_counterfactual_uses_supplied_equations(self):
        out=structural_counterfactual({"belief":(("sensor",),lambda x:x),
            "action":(("belief",),lambda x:2*x),"outcome":(("action",),lambda x:int(x==4))},
            {"sensor":1},{"sensor":2})
        self.assertEqual(out["actual"]["outcome"],0);self.assertEqual(out["counterfactual"]["outcome"],1)
        with self.assertRaises(ValueError):
            structural_counterfactual({"a":(("b",),lambda x:x),"b":(("a",),lambda x:x)},{},{})
    def test_pulses_preserve_epistemic_labels(self):
        core=BedrockRuntime(model_id="one");task=core.open("reason")
        core.add_pulse(task,Pulse("hypothesis","x might control y"))
        self.assertIn('"verified": false',core.pulse_text(task))
        with self.assertRaises(ValueError):Pulse("observation","x",verified=True)
    def test_frozen_and_cognitive_identity_must_match(self):
        class Wrong: model_id="other"
        with self.assertRaises(ValueError):BedrockRuntime(model_id="one",neural=Wrong())
    def test_no_neural_executor_not_silently_replaced(self):
        with self.assertRaises(RuntimeError):BedrockRuntime(model_id="one").respond("hello",None)

if __name__=="__main__":unittest.main()

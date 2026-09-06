"""Connect grid search to the existing one-identity cognitive runtime and memory."""
from __future__ import annotations
from ..bedrock.runtime import BedrockRuntime
from ..bedrock.contracts import Budget
from ..cognitive_kernel import CognitiveOperator
from ..types import Provenance, ProvenanceKind
from .contracts import ArcTask, SearchConfig, digest
from .memory import SkillLibrary
from .search import StrengthSearch
from .programs import Program

class StrengthRuntime(BedrockRuntime):
    def __init__(self,*,model_id,neural=None,memory_journal=None,tokenizer=None,config=SearchConfig(),progress=None):
        super().__init__(model_id=model_id,neural=neural,memory_journal=memory_journal)
        self.config=config;self.progress=progress or (lambda s:None)
        self.library=SkillLibrary(model_revision=neural.revision if neural else 'symbolic',memory=self.memory)
        if neural is not None:
            if tokenizer is None:raise ValueError('Tokenizer for the one neural model is required')
            from .proposer import QwenProposer
            self.proposer=QwenProposer(neural,tokenizer,seed=config.seed,activation_trials=config.activation_trials,progress=progress)
        else:self.proposer=None
    def solve_arc(self,task:ArcTask,*,remember_support=False):
        if remember_support and task.split=='evaluation':raise ValueError('Cross-task evaluation memory updates are disabled')
        if self.proposer:
            self.proposer.memory_states=self.library.compatible_neural_states(task)
            self.proposer.memory_routes=self.library.compatible_interventions(task)
        session=self.open('Infer a grid transformation from visible demonstrations',task_type='arc_grid',
            scope=task.id,budget=Budget(model_calls=2048,layer_calls=1000000,generated_tokens=8192,
                                     branches=64,wall_seconds=self.config.max_seconds+180.))
        self.step(session,CognitiveOperator.ENCODE,lambda:task.view(),summary='Read demonstrations and query inputs; labels excluded')
        self.step(session,CognitiveOperator.ABSTRACT,lambda:None,summary='Grid, color, object, panel and topology representation beams')
        result=self.step(session,CognitiveOperator.SEARCH,
            lambda:StrengthSearch(self.config,self.proposer,self.library,self.progress).solve(task),
            summary='Counterexample-constrained proposal repair and compositional search')
        prediction='arc-prediction-'+digest([task.support_hash,result['attempts']])
        self.step(session,CognitiveOperator.PREDICT,lambda:self.record_prediction(session.session.id,prediction),
                  summary='Commit at most two distinct output grids per query before any external scoring')
        # Demonstration fit is not independent verification of query outputs.
        evidence='support-only-'+task.support_hash
        self.step(session,CognitiveOperator.VERIFY,lambda:self.record_verification(session.session.id,evidence),
                  summary='Demonstration consistency only; test outcome remains unknown')
        if remember_support:
            if task.split=='evaluation':raise ValueError('Cross-task evaluation memory updates are disabled')
            for p in result['programs']:
                self.library.remember_candidate(task,Program.from_dict(p),
                    neural_state=self.proposer.workspace.serialize() if self.proposer and self.proposer.workspace else None,
                    intervention_evidence=self.proposer.activation_report if self.proposer else None)
        self.complete_task(session.session.id,outcome_ref=evidence,verified_success=False,
            provenance=Provenance(ProvenanceKind.SELF_INFERENCE,self.model_id,0.),truth_quality=0.,novelty=.5,
            transfer_value=0.,independent_verification=False,rollback_available=True,
            episode_payload={'type':'arc_submission','task_id':task.id,'query_outcome':'unknown','prediction_ref':prediction})
        result['model_id']=self.model_id;result['graph_complete']=session.session.graph.complete
        result['activation_trials']=self.proposer.activation_report if self.proposer else {'status':'no_neural_model'}
        result['neural_total_calls']=self.proposer.total_calls if self.proposer else 0
        return result

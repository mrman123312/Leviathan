"""Same frozen Qwen supplies distinct, bounded proposals and structured revisions.

No true-rule catalogue, hidden output, external LLM, optimizer, or arbitrary code
execution. Malformed proposals remain recorded failures; symbolic search is an
explicit second *algorithm*, not silently attributed to Qwen's neural reasoning.
"""
from __future__ import annotations
from contextlib import nullcontext, ExitStack
from dataclasses import asdict
import json
import torch
from .contracts import digest
from .grid import summarize
from .neural import NeuralFabric, NeuralRoute, TaskWorkspace


def rows(g):return '\n'.join(''.join(map(str,row)) for row in g)

def render_task(task,view='rows'):
    parts=[]
    for i,ex in enumerate(task.examples):
        parts.append(f'Example {i+1}\nInput:\n{rows(ex.input)}\nOutput:\n{rows(ex.output)}')
        if view=='objects':
            parts.append('Input object description: '+json.dumps(summarize(ex.input),separators=(',',':')))
            parts.append('Output object description: '+json.dumps(summarize(ex.output),separators=(',',':')))
        elif view=='differences':
            same=(len(ex.input),len(ex.input[0]))==(len(ex.output),len(ex.output[0]))
            if same:
                changes=[(r,c,x,y) for r,(a,b) in enumerate(zip(ex.input,ex.output)) for c,(x,y) in enumerate(zip(a,b)) if x!=y]
                parts.append('Changed cells (row,column,before,after): '+json.dumps(changes[:64]))
                if len(changes)>64:parts.append(f'{len(changes)-64} further changes; full grids above remain authoritative.')
    parts.append('Test inputs, outputs unknown:\n'+'\n---\n'.join(rows(g) for g in task.queries))
    return '\n\n'.join(parts)

GRAMMAR="""Infer one Grid->Grid transformation from ALL examples. The same rule must generalize.
Write a nested expression using x for the input grid. Return a single expression.
Functions: rot90(x), rot180(x), rot270(x), flip_lr(x), flip_ud(x), transpose(x),
crop(x, background), crop_color(x, color), recolor(x, from_color, to_color),
scale(x, factor), tile(x, vertical_count, horizontal_count), subsample(x, factor), compress(x),
largest(x, background, 4, 0, 0), smallest(x, background, 4, 0, 0),
fill_holes(x, color, background), connect(x, color, 'both', background),
gravity(x, 'down', background), complete_symmetry(x, 'h', background),
panel(x, 'h', 0), overlay(x, 'h', 'xor', color, background),
mirror(x, 'h', 0), border(x, color, 1), trim(x, 1).
Colors are integers 0..9. Background -1 means most frequent input color.
Functions may be composed. Examples of SYNTAX ONLY: rot90(crop(x, 0)); recolor(flip_lr(x), 1, 2).
Do not use task IDs, filenames, memorized solutions, imports, Python statements or prose.
"""

class QwenProposer:
    is_neural=True
    def __init__(self,executor,tokenizer,*,seed=1607,max_new_tokens=256,activation_trials=False,progress=None):
        self.executor=executor;self.model=executor.model;self.tokenizer=tokenizer
        self.seed=seed;self.max_new_tokens=max_new_tokens;self.activation_trials=activation_trials
        self.progress=progress or (lambda s:None);self.last_calls=0;self.total_calls=0
        self.fabric=NeuralFabric(executor);self.workspace=None;self.selected_route=NeuralRoute()
        self.activation_report={'status':'disabled_until_demonstration_gate'}
        self.prepared=None;self.records=[];self.memory_states=[];self.memory_routes=[];self.selected_slot=0
    @property
    def device(self):return self.model.get_input_embeddings().weight.device
    def encode(self,text):return self.tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids.to(self.device)
    def prepare(self,task):
        if self.prepared==task.support_hash:return
        self.prepared=task.support_hash;self.workspace=None;self.selected_route=NeuralRoute();self.selected_slot=0
        if not self.activation_trials:return
        # One source demo is disjoint from two held-out SUPPORT demo outputs. Never
        # extract from all examples then claim an independent support validation.
        if len(task.examples)<3:
            self.activation_report={'status':'insufficient_disjoint_support','accepted':False};return
        self.progress('  Testing optional neural interventions on disjoint visible demonstrations only')
        source=task.examples[:-2];validation_examples=task.examples[-2:]
        layer=max(0,len(self.fabric.layers)//2-2)
        positive=[];control=[]
        for ex in source[:4]:
            prefix=f'Input:\n{rows(ex.input)}\nOutput:\n'
            positive.append(self.encode(prefix+rows(ex.output)+'\nTransformation:'))
            control.append(self.encode(prefix+rows(ex.input)+'\nTransformation:'))
        try:
            self.workspace=self.fabric.contrast_workspace(positive,control,layer=layer,
                support_hash=digest([(e.input,e.output) for e in source]))
            context='\n\n'.join(f'Input:\n{rows(e.input)}\nOutput:\n{rows(e.output)}' for e in source)
            validation=[]
            for ex in validation_examples:
                prompt=context+f'\n\nInput:\n{rows(ex.input)}\nOutput:\n'
                prefix=self.encode(prompt);ids=self.encode(prompt+rows(ex.output))
                # Tokenizer boundary changes are not allowed to corrupt label scoring.
                if not torch.equal(ids[:,:prefix.shape[1]],prefix):
                    raise ValueError('Support NLL token boundary changed')
                validation.append((ids,prefix.shape[1]))
            self.selected_route,self.activation_report=self.fabric.select_on_demonstrations(validation,workspace=self.workspace,prior_routes=self.memory_routes)
            # Transferred dual-form states must pass the same disjoint-support gate.
            # They never bypass present evidence because a prior record sounded confident.
            for stored in self.memory_states:
                if stored.get('revision')!=self.executor.revision:continue
                candidate=TaskWorkspace(stored['revision'],stored['layer'],stored['support_hash'],
                                        torch.tensor(stored['vectors'],dtype=torch.float32))
                route,trial=self.fabric.select_on_demonstrations(validation,workspace=candidate)
                if route.kind=='task_state' and trial['selected_validation_loss']<self.activation_report.get('selected_validation_loss',float('inf')):
                    self.workspace=candidate;self.selected_route=route;self.activation_report=trial
                    self.activation_report['transferred_from_memory']=True
                    break
            self.selected_slot=self.activation_report.get('selected_slot',0)
            self.activation_report['workspace']=self.workspace.metadata()
            self.activation_report['extraction_and_validation_support_disjoint']=True
        except (ValueError,FloatingPointError,RuntimeError) as exc:
            self.selected_route=NeuralRoute();self.workspace=None
            self.activation_report={'status':'rejected','error':str(exc),'accepted':False}
        self.total_calls+=self.fabric.forward_calls;self.last_calls+=self.fabric.forward_calls;self.fabric.forward_calls=0
    def propose(self,task,*,rejected,view,count,round_index):
        self.last_calls=0
        self.prepare(task)
        witnesses=json.dumps(rejected,separators=(',',':'))
        prompt=GRAMMAR+'\n'+render_task(task,view)+('\nRejected complete programs and specific failing cells:\n'+witnesses if rejected else '')
        prompt+='\nChange the rule structure to satisfy every observed example.\nExpression:'
        ids=self.encode(prompt)
        if ids.shape[1]>6144:
            # Re-render as lossless raw rows; never silently truncate demonstrations.
            prompt=GRAMMAR+'\n'+render_task(task,'rows')+'\nExpression:';ids=self.encode(prompt)
        if ids.shape[1]>6144:
            return [{'text':'','error':'Full visible task exceeds 6144-token local experiment cap'}]
        generated=[];seen=set();cuda_devices=[self.device.index or 0] if self.device.type=='cuda' else []
        for i in range(count):
            self.progress(f'    Same-Qwen proposal {i+1}/{count}, view={view}, round={round_index+1}')
            seed=self.seed+int(task.support_hash[:8],16)+round_index*97+i
            # Branch 0 is the actual donor. Activation routes never replace all
            # unmodified proposals even after a support-NLL gain.
            route=NeuralRoute() if i==0 else self.selected_route
            # A second representation language avoids forcing all hypotheses into
            # our hand-written grid primitive names. It is interpreted, never exec'd.
            python_mode=i>=2
            active_ids=ids
            if python_mode:
                py_prompt=('Infer a grid transformation consistent with every example. Write only def transform(x). '
                  'x is a list of rows of integers. Allowed: assignments, for, if, return, list comprehensions, '
                  'integer arithmetic, indexing, len/range/min/max/sum/abs/sorted/enumerate/zip/list/reversed/all/any, '
                  'list append/extend/copy/count/index/reverse. No imports, arbitrary attributes, recursion, while loops or external operations.\n'+render_task(task,view)+
                  '\nCounterexamples to previous attempts: '+witnesses+'\n\ndef transform(x):\n')
                active_ids=self.encode(py_prompt)
                if active_ids.shape[1]>6144:
                    generated.append({'text':'','error':'Python-view prompt exceeds token cap'});continue
            slot=self.selected_slot
            if i>1 and route.kind!='donor':
                eligible=[t for t in self.activation_report.get('trials',[]) if t.get('accepted')]
                if eligible:
                    trial=eligible[(i-1)%len(eligible)];route=NeuralRoute(**trial['route']);slot=trial.get('slot',0)
            ctx=self.fabric.use(route,workspace=self.workspace,slot=slot)
            with self.executor._lock,torch.inference_mode(),torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                kwargs={'input_ids':active_ids,'max_new_tokens':self.max_new_tokens,'pad_token_id':self.tokenizer.eos_token_id,
                        'eos_token_id':self.tokenizer.eos_token_id,'do_sample':i!=0 or round_index>0,
                        'use_cache':route.kind!='damped_band'}
                if kwargs['do_sample']:kwargs.update(temperature=.65+.15*(i%3),top_p=.95)
                def count_forward(module,args):
                    self.last_calls+=1;self.total_calls+=1
                with ExitStack() as handles:
                    handle=self.model.register_forward_pre_hook(count_forward);handles.callback(handle.remove)
                    with ctx:out=self.model.generate(**kwargs)
            text=self.tokenizer.decode(out[0,active_ids.shape[1]:],skip_special_tokens=True)
            if python_mode:text='def transform(x):\n'+text
            else:text=text.strip()
            if text.startswith('```'):text='\n'.join(text.splitlines()[1:]).split('```')[0].strip()
            text=(text.split('```')[0].rstrip() if python_mode else (text.splitlines()[0] if text else ''))
            record={'text':text,'route':asdict(route),'task_slot':slot,'duplicate_generation':text in seen,
                    'generated_tokens':int(out.shape[1]-active_ids.shape[1]),'seed':seed,'view':view}
            seen.add(text);generated.append(record);self.records.append(record)
        if not self.executor.unchanged():raise RuntimeError('Frozen weight tripwire failed after proposals')
        return generated


def parse_grid_completion(text):
    """Strict published grid formats. No guessed repairs or hidden-label matching."""
    from .contracts import grid
    text=text.strip()
    if text.startswith('```'):
        text='\n'.join(text.splitlines()[1:]).split('```')[0].strip()
    if text.startswith('['):
        try:return grid(json.loads(text))
        except (ValueError,TypeError) as exc:raise ValueError('Invalid JSON grid') from exc
    lines=text.splitlines()
    if not lines:raise ValueError('Empty grid response')
    rows_out=[]
    for line in lines:
        line=line.strip()
        if not line:break
        if not all(c in '0123456789 ' for c in line):raise ValueError('Expected only digit rows, not prose')
        values=line.split() if ' ' in line else list(line)
        if any(len(v)!=1 for v in values):raise ValueError('Grid colors must be single digits')
        rows_out.append([int(x) for x in values])
    return grid(rows_out)


def direct_grid_control(executor,tokenizer,task,*,progress=None,max_new_tokens=1400):
    """Plain frozen-Qwen direct-grid baseline. Same evidence, no search or patches.

    Two fixed seeded generations per query. Reports only parseable grids; malformed
    generations remain incorrect/abstained. This is independent of program search.
    """
    progress=progress or (lambda s:None)
    device=executor.model.get_input_embeddings().weight.device
    prefix='Complete the Output grid for the final Input. Digits 0-9 represent colors. Return only rows of digits.\n\n'
    prefix+='\n\n'.join(f'Input:\n{rows(e.input)}\nOutput:\n{rows(e.output)}' for e in task.examples)
    attempts=[];records=[];calls=0
    cuda_devices=[device.index or 0] if device.type=='cuda' else []
    for qi,query in enumerate(task.queries):
        ids=tokenizer(prefix+f'\n\nInput:\n{rows(query)}\nOutput:\n',return_tensors='pt',add_special_tokens=False).input_ids.to(device)
        predictions=[]
        if ids.shape[1]>6144:
            attempts.append([]);records.append({'query':qi,'error':'Full demonstration prompt exceeds 6144 tokens'});continue
        for k in range(2):
            progress(f'  Plain donor grid {qi+1}/{len(task.queries)}, attempt {k+1}/2')
            kwargs=dict(input_ids=ids,max_new_tokens=max_new_tokens,do_sample=k>0,use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id)
            if k:kwargs.update(temperature=.7,top_p=.95)
            def count(module,args):
                nonlocal calls
                calls+=1
            with executor._lock,torch.inference_mode(),torch.random.fork_rng(devices=cuda_devices),ExitStack() as stack:
                torch.manual_seed(1607+int(task.support_hash[:8],16)+qi*7+k)
                handle=executor.model.register_forward_pre_hook(count);stack.callback(handle.remove)
                out=executor.model.generate(**kwargs)
            text=tokenizer.decode(out[0,ids.shape[1]:],skip_special_tokens=True)
            record={'query':qi,'attempt':k,'text':text,'generated_tokens':int(out.shape[1]-ids.shape[1])}
            try:
                prediction=parse_grid_completion(text);predictions.append(prediction);record['valid_grid']=True
            except ValueError as exc:
                predictions.append(None);record.update(valid_grid=False,error=str(exc))
            records.append(record)
        attempts.append(predictions)
    if not executor.unchanged():raise RuntimeError('Donor weights changed during direct control')
    return {'attempts':attempts,'status':'generated','records':records,'neural_total_calls':calls,
            'training_steps':0,'new_parameters':0,'protocol':'direct digit-row few-shot donor; two attempts; no search or interventions'}

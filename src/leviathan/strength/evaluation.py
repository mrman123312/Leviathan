"""Evaluator only. The solver never imports labels or calls these functions.

Scores are exact whole-grid pass@1/pass@2 and whole-task all-query pass@1/pass@2.
Two attempts is stricter than the original ARC repository's three-trial interface.
"""
from __future__ import annotations
import hashlib
from .contracts import grid


def select_ids(tasks,limit=50):
    if not 1<=limit<=len(tasks):raise ValueError('Selection limit outside dataset')
    return sorted(tasks,key=lambda x:hashlib.sha256(('leviathan-strength-v1:'+x).encode()).hexdigest())[:limit]


def score(predictions:dict,labels:dict,task_ids):
    if not task_ids:raise ValueError('At least one task must be evaluated')
    records=[];task1=task2=query1=query2=queries=0
    for identifier in task_ids:
        if identifier not in labels:raise ValueError('Missing evaluator answer')
        gold=[grid(x) for x in labels[identifier]]
        if not gold:raise ValueError('A task must contain at least one query answer')
        offered=predictions.get(identifier,{}).get('attempts',[])
        q=[]
        for i,target in enumerate(gold):
            guesses=offered[i][:2] if i<len(offered) else []
            valid=[]
            for x in guesses:
                try:valid.append(grid(x))
                except ValueError:valid.append(None)
            first=bool(valid and valid[0]==target);second=target in valid
            q.append({'query':i,'pass1':first,'pass2':second})
            query1+=first;query2+=second;queries+=1
        p1=all(x['pass1'] for x in q);p2=all(x['pass2'] for x in q)
        task1+=p1;task2+=p2
        records.append({'task_id':identifier,'pass1':p1,'pass2':p2,'queries':q})
    n=len(task_ids)
    return {'tasks':n,'task_pass1':task1,'task_pass2':task2,'task_accuracy1':task1/n,'task_accuracy2':task2/n,
            'query_count':queries,'query_pass1':query1,'query_pass2':query2,'records':records,
            'exact_dimensions_and_every_cell_required':True,'max_attempts_per_query':2}

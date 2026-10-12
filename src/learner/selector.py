"""Demonstration-resampled learned/native attempt arbitration. MIT-0."""
import time
from anytime import solve as learned_solve


def candidates(row):
    return row['baseline']['candidates'] + (row.get('extension') or {}).get('candidates', [])


def proposals(row, nq):
    result = [[] for _ in range(nq)]
    for c in candidates(row):
        for qi in range(nq):
            if len(result[qi]) < 2 and all(c['predictions'][qi] != x['grid'] for x in result[qi]):
                result[qi].append(dict(grid=c['predictions'][qi],program=c['program']))
    return result


def certify(task, full=None):
    start=time.process_time()
    blind={'train':[{'input':p['input'],'output':p['output']} for p in task['train']],
           'test':[{'input':q['input']} for q in task['test']]}
    full_fresh=full is None
    if full_fresh: full=learned_solve(blind)
    n=len(blind['train']); nq=len(blind['test']); choices=proposals(full,nq)
    folds=[]
    if n >= 2 and any(choices):
        for omitted, pair in enumerate(blind['train']):
            fold={'train':blind['train'][:omitted]+blind['train'][omitted+1:],
                  'test':[{'input':pair['input']}]+blind['test']}
            row=learned_solve(fold)
            # Canonical candidates are distinct as whole prediction tuples.
            selected=[]
            for c in candidates(row):
                if all(c['predictions']!=s['predictions'] for s in selected):selected.append(c)
                if len(selected)==2:break
            folds.append(dict(omitted=omitted,row=row,selected=selected,
                              correct=[c['predictions'][0]==pair['output'] for c in selected]))
    chosen=[]
    for qi, options in enumerate(choices):
        for rank,option in enumerate(options):
            joint=agreement=exact=0
            for fold in folds:
                cs=fold['selected']; flags=fold['correct']
                joint+=any(ok and c['predictions'][qi+1]==option['grid'] for c,ok in zip(cs,flags))
                agreement+=any(c['predictions'][qi+1]==option['grid'] for c in cs)
                exact+=any(flags)
            option.update(joint=joint,agreement=agreement,fold_exact=exact,demonstrations=n,rank=rank)
        options.sort(key=lambda c:(-c['joint'],-c['agreement'],c['rank']))
        chosen.append(options[0] if options else None)
    rows=[f['row'] for f in folds]
    return dict(chosen=chosen,choices=choices,folds=folds,full=full,full_fresh=full_fresh,
                certificate_cpu_seconds=time.process_time()-start,
                fold_cpu_seconds=sum(r['cpu_seconds'] for r in rows),
                fold_primitive_calls=sum(r['work_counts']['primitive_grid_evaluations'] for r in rows),
                fold_constructor_cpu_seconds=sum(r['baseline']['constructor_cpu_seconds'] for r in rows))


def accept(proposal, gate):
    return bool(proposal and proposal['demonstrations']>=gate['min_demonstrations']
                and proposal['joint']/proposal['demonstrations']>=gate['min_joint_fraction'])


def merge(baseline,certificate,gate):
    answers=[]
    for old,proposal in zip(baseline,certificate['chosen']):
        new=dict(old)
        if accept(proposal,gate) and proposal['grid']!=old['attempt_1']:
            new['attempt_2']=proposal['grid']
        answers.append(new)
    return answers

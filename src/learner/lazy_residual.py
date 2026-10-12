"""Exact answer stopping for a frozen linear residual ranker. MIT-0.

Bounds include all future fold votes, not just votes for current winners.
No model, grammar, candidate or tie-breaking changes are made.
"""
import math
import time
from selector import candidates, learned_solve
from residual import features, predict_from_features, confidence


def bounds(entries, n, remaining, model):
    """Conservative per-candidate logit intervals for arbitrary future folds.

Every vote can gain 0..remaining. Joint votes cannot exceed agreement votes.
Joint counts also determine two Boolean features and competitor margins.
    Exact-held-demo terms (15,18) are identical for every candidate. Cancel
    their exact real contributions, retaining their maximum magnitudes in
    the floating-point error budget for each original complete score.
"""
    result=[]
    for e in entries:
        x=e['x']; lo=x.copy(); hi=x.copy()
        for i in (13,14,15,16,17,18):hi[i]=min(n,round(x[i]*n)+remaining)/n
        hi[19]=x[19]+remaining;hi[20]=x[20]+remaining
        lo[21]=float(x[19]+remaining==0);hi[21]=float(x[19]==0)
        lo[22]=float(x[20]==n);hi[22]=float(x[20]+remaining==n)
        others=[z for z in entries if z is not e]
        tj=max((z['two_joint'] for z in others),default=0)
        aj=max((z['all_joint'] for z in others),default=0)
        lo[35]=(x[19]-tj-(remaining if others else 0))/n
        hi[35]=(x[19]+remaining-tj)/n
        lo[36]=(x[20]-aj-(remaining if others else 0))/n
        hi[36]=(x[20]+remaining-aj)/n
        lows=[];highs=[];maxpad=0.0
        for dt in range(remaining+1):
            for da in range(dt,remaining+1):
                a=lo.copy();b=hi.copy();t=int(x[19])+dt;u=int(x[20])+da
                a[13]=b[13]=t/n;a[16]=b[16]=u/n
                a[19]=b[19]=t;a[20]=b[20]=u
                a[21]=b[21]=float(t==0);a[22]=b[22]=float(u==n)
                a[14]=(round(x[14]*n)+dt)/n
                a[17]=(round(x[17]*n)+da)/n
                a[35]=(t-tj-(remaining if others else 0))/n;b[35]=(t-tj)/n
                a[36]=(u-aj-(remaining if others else 0))/n;b[36]=(u-aj)/n
                lower=[];upper=[]
                for av,bv,c,m,s in zip(a,b,model['coefficients'],model['mean'],model['scale']):
                    assert av<=bv and s>0 and all(map(math.isfinite,(av,bv,c,m,s)))
                    v=c*(av-m)/s;w=c*(bv-m)/s
                    lower.append(min(v,w));upper.append(max(v,w))
                magnitude=abs(model['intercept'])+sum(max(abs(v),abs(w)) for v,w in zip(lower,upper))
                assert math.isfinite(magnitude) and magnitude<1e300
                pad=512*math.ulp(max(1.0,4*magnitude));maxpad=max(maxpad,pad)
                lows.append(model['intercept']+sum(v for i,v in enumerate(lower) if i not in (15,18))-pad)
                highs.append(model['intercept']+sum(v for i,v in enumerate(upper) if i not in (15,18))+pad)
        result.append(dict(lower=min(lows),upper=max(highs),feature_lower=lo,feature_upper=hi,padding=maxpad))
    return result


def decision(entries,n,remaining,model):
    if len(entries)==1:return 0,[]
    if remaining==0:
        winner=min(range(len(entries)),key=lambda i:(-confidence(entries[i]['x'],model),
                   not entries[i]['native_second'],entries[i]['order'],
                   tuple(map(tuple,entries[i]['grid']))))
        return winner,[]
    intervals=bounds(entries,n,remaining,model)
    for i,b in enumerate(intervals):
        if all(i==j or b['lower']>other['upper'] for j,other in enumerate(intervals)):
            return i,intervals
    return None,intervals


def certify(task,baseline,model,full=None,fold_provider=None):
    start=time.process_time()
    blind={'train':[dict(input=p['input'],output=p['output']) for p in task['train']],
           'test':[dict(input=q['input']) for q in task['test']]}
    if full is None:full=learned_solve(blind)
    n=len(blind['train']);nq=len(blind['test']);folds=[];trace=[]
    count=n if n>=2 and candidates(full) else 0
    for completed in range(count+1):
        entries=features(blind,dict(full=full,folds=folds),baseline)
        decisions=[decision(es,n,count-completed,model) for es in entries]
        trace.append(dict(completed=completed,remaining=count-completed,
                          winners=[w for w,_ in decisions],intervals=[b for _,b in decisions]))
        if all(w is not None for w,_ in decisions):break
        assert completed<count
        pair=blind['train'][completed]
        if fold_provider is None:
            ft=dict(train=blind['train'][:completed]+blind['train'][completed+1:],
                    test=[dict(input=pair['input'])]+blind['test'])
            row=learned_solve(ft)
        else:row=fold_provider(completed)
        selected=[]
        for c in candidates(row):
            if all(c['predictions']!=s['predictions'] for s in selected):selected.append(c)
            if len(selected)==2:break
        folds.append(dict(omitted=completed,row=row,selected=selected,
                          correct=[c['predictions'][0]==pair['output'] for c in selected]))
    answers=[dict(attempt_1=old['attempt_1'],attempt_2=es[w]['grid'])
             for old,es,(w,_) in zip(baseline,entries,decisions)]
    return dict(full=full,folds=folds,entries=entries,bound_trace=trace,
                selected_attempts=answers,completed_folds=len(folds),
                original_fold_count=count,skipped_folds=count-len(folds),
                certificate_cpu_seconds=time.process_time()-start,
                fold_primitive_calls=sum(f['row']['work_counts']['primitive_grid_evaluations'] for f in folds),
                fold_cpu_seconds=sum(f['row']['cpu_seconds'] for f in folds))

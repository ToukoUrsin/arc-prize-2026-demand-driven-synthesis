"""Skip validation only when each residual pool has exactly one grid. MIT-0."""
import time
from selector import certify as original,learned_solve,candidates
from residual import features,predict_from_features
def certify(task,baseline,model):
    start=time.process_time()
    blind=dict(train=[dict(input=p['input'],output=p['output']) for p in task['train']],
               test=[dict(input=q['input']) for q in task['test']])
    full=learned_solve(blind)
    empty=dict(full=full,folds=[])
    entries=features(blind,empty,baseline)
    n=len(blind['train']);count=n if n>=2 and candidates(full) else 0
    singleton=all(len(es)==1 for es in entries)
    if singleton:
        row=dict(full=full,folds=[],fold_cpu_seconds=0.0,fold_primitive_calls=0)
        answer=[dict(attempt_1=b['attempt_1'],attempt_2=es[0]['grid']) for es,b in zip(entries,baseline)]
    else:
        row=original(blind,full=full)
        entries=features(blind,row,baseline)
        answer=predict_from_features(baseline,row,entries,model)
    row.update(entries=entries,selected_attempts=answer,singleton_pool=singleton,
               original_fold_count=count,completed_folds=len(row['folds']),
               skipped_folds=count-len(row['folds']),certificate_cpu_seconds=time.process_time()-start)
    return row

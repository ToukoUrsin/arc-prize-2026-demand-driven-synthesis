"""Standalone lazy residual runtime; explicit native input, frozen weights."""
import argparse,gzip,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from lazy_residual import certify
from inference_guard import blind_view,inference_access
def one(item):
    name,task,baseline,model=item
    DecisionTreeClassifier(max_depth=1,random_state=0).fit(np.array([[0.],[1.]]),[0,1])
    start=time.process_time()
    with inference_access():row=certify(blind_view(task),baseline,model)
    row['selector_cpu_seconds']=time.process_time()-start;row['task_id']=name
    return name,row['selected_attempts'],row
def main():
    p=argparse.ArgumentParser();p.add_argument('--challenges',required=True);p.add_argument('--baseline',required=True);p.add_argument('--gate',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    tasks=json.loads(Path(a.challenges).read_text());base=json.loads(Path(a.baseline).read_text());model=json.loads(Path(a.gate).read_text());assert set(tasks)==set(base)
    items=[(n,dict(train=[dict(input=p['input'],output=p['output']) for p in t['train']],test=[dict(input=q['input']) for q in t['test']]),base[n],model) for n,t in sorted(tasks.items())]
    out=Path(a.out);out.mkdir(exist_ok=False);submission={};cpu=fold_cpu=primitive=skipped=0;start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool,gzip.open(out/'certificates.jsonl.gz','wt') as f:
        for n,ans,row in pool.map(one,items,chunksize=1):
            submission[n]=ans;cpu+=row['selector_cpu_seconds'];fold_cpu+=row['fold_cpu_seconds'];primitive+=row['fold_primitive_calls']+row['full']['work_counts']['primitive_grid_evaluations'];skipped+=row['skipped_folds']
            f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
            if len(submission)%20==0:print('lazy residual',len(submission),len(tasks),flush=True)
    (out/'submission.json').write_text(json.dumps(submission,separators=(',',':'))+'\n')
    receipt=dict(complete=True,tasks=len(tasks),stage_cpu_seconds=cpu,fold_cpu_seconds=fold_cpu,primitive_calls=primitive,skipped_folds=skipped,wall_seconds=time.perf_counter()-start,model_name=model['name'],workers=a.workers,baseline_recomputed=False)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

"""Offline two-attempt adapter for the pinned/reset MIT icecuber baseline.
Authored adapter SPDX-License-Identifier: MIT-0; embedded solver remains MIT.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


def valid(grid):
    return (isinstance(grid,list) and 1<=len(grid)<=30 and isinstance(grid[0],list)
            and 1<=len(grid[0])<=30 and all(isinstance(row,list) and len(row)==len(grid[0]) for row in grid)
            and all(type(c) is int and 0<=c<=9 for row in grid for c in row))


def predict(challenges,root,depth=2,timeout=1500):
    """Inference has no query outputs. Returns attempts plus execution receipt."""
    if not challenges:raise ValueError('Empty challenge dictionary')
    root=Path(root).resolve();target=root/'dataset/test'
    indices=[];source_hashes={}
    # Serialize only demonstrations and test inputs, including when labels supplied.
    blind={name:{'train':task['train'],'test':[{'input':q['input']} for q in task['test']]}
           for name,task in challenges.items()}
    for name,task in blind.items():
        if not re.fullmatch('[a-z0-9]{8}',name):raise ValueError('Expected official eight-character task ID')
        for pair in task['train']:
            assert valid(pair['input']) and valid(pair['output'])
        for pair in task['test']:assert valid(pair['input'])
    if target.exists():shutil.rmtree(target)
    target.mkdir(parents=True)
    for name,task in sorted(blind.items()):
        payload=json.dumps(task,separators=(',',':'))+'\n'
        (target/(name+'.json')).write_text(payload)
        source_hashes[name]=hashlib.sha256(payload.encode()).hexdigest()
        indices.extend((name,i) for i in range(len(task['test'])))
    output=root/'output';output.mkdir(exist_ok=True)
    for i in range(len(indices)):(output/f'answer_{i}_{depth}.csv').unlink(missing_ok=True)
    env=dict(os.environ)
    env.pop('ICECUBER_PROBE_MAXAREA',None);env.pop('ICECUBER_RESET_NORMALIZATION',None)
    started=time.perf_counter()
    with (output/'adapter.stdout').open('w') as stdout,(output/'adapter.stderr').open('w') as stderr:
        process=subprocess.Popen([str(root/'run'),'-1',str(depth)],cwd=root,env=env,
                                 stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();code='timeout'
    submission={name:[] for name in sorted(blind)};missing=[]
    for index,(name,qi) in enumerate(indices):
        csv=output/f'answer_{index}_{depth}.csv';grids=[]
        if csv.exists():
            lines=csv.read_text().strip().splitlines()
            assert lines and lines[0]==f'{name}_{qi}'
            for line in lines[1:]:
                grid=[[int(c) for c in row] for row in line.split()[0].strip('|').split('|')]
                assert valid(grid)
                grids.append(grid)
                if len(grids)==2:break
        else:missing.append(f'{name}_{qi}')
        while len(grids)<2:grids.append(blind[name]['test'][qi]['input'])
        submission[name].append({'attempt_1':grids[0],'attempt_2':grids[1]})
    return submission,{'binary_sha256':hashlib.sha256((root/'run').read_bytes()).hexdigest(),
                       'depth_argument':depth,'exit_code':code,'missing_queries':missing,
                       'tasks':len(blind),'queries':len(indices),'wall_seconds':time.perf_counter()-started,
                       'blind_input_hashes':source_hashes,'test_outputs_serialized':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',default='submission.json')
    p.add_argument('--root',default='data/icecuber-reset');p.add_argument('--depth',type=int,default=2)
    p.add_argument('--timeout',type=int,default=1500);a=p.parse_args()
    submission,receipt=predict(json.loads(Path(a.input).read_text()),a.root,a.depth,a.timeout)
    Path(a.output).write_text(json.dumps(submission,separators=(',',':'))+'\n')
    Path(a.output+'.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='blind_input_hashes'},indent=2))


if __name__=='__main__':main()

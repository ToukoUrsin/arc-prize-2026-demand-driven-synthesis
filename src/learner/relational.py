"""Demonstration-fitted relational cell rules; no query labels or task IDs.
SPDX-License-Identifier: MIT-0
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import time
import numpy as np
from scipy import ndimage as ndi
from sklearn.tree import DecisionTreeClassifier

VERSION = 'relational-rules-v3'
DEPTHS = (2, 4, 6, 10)
GROUPS = ('local', 'objects', 'rays', 'all')


def clean_task(task):
    return {'train': [{'input': p['input'], 'output': p['output']} for p in task['train']],
            'test': [{'input': p['input']} for p in task['test']]}


def mode(a):
    return int(np.bincount(a.ravel(), minlength=10).argmax())


def transform(a, name):
    if name == 'identity': return a
    bg = 0 if name == 'crop_zero' else mode(a)
    y, x = np.nonzero(a != bg)
    if len(y) == 0: return a
    return a[y.min():y.max()+1, x.min():x.max()+1]


def features(a):
    """Fixed input-only features. Each feature is an integer per grid cell."""
    a = np.asarray(a, dtype=np.int16)
    h, w = a.shape
    yy, xx = np.indices((h, w))
    cols, names, tags = [], [], []
    bg = mode(a)
    def add(name, v, tag):
        if np.isscalar(v): v = np.full_like(a, v)
        cols.append(np.array(v, dtype=np.int16, copy=True).ravel()); names.append(name); tags.append(tag)
    add('color', a, 'base')
    add('nonzero', a != 0, 'base'); add('nonbackground', a != bg, 'base')
    add('background', bg, 'base')
    # Absolute geometry is separate: simpler feature subsets can generalize away
    # from accidental coordinate correlations.
    for name, v in [('row',yy),('column',xx),('bottom',h-1-yy),('right',w-1-xx),
                    ('row_parity',yy%2),('col_parity',xx%2),
                    ('border',np.minimum.reduce([yy,xx,h-1-yy,w-1-xx])),
                    ('diagonal',yy-xx),('antidiagonal',yy+xx),
                    ('h',h),('w',w)]: add(name,v,'position')
    pad = np.pad(a,2,constant_values=10)
    for dy in range(-2,3):
        for dx in range(-2,3):
            if dy == dx == 0: continue
            add(f'neighbor_{dy}_{dx}',pad[2+dy:2+dy+h,2+dx:2+dx+w], 'local')
    add('mirror_ud',a[::-1], 'local'); add('mirror_lr',a[:,::-1], 'local')
    add('mirror_both',a[::-1,::-1], 'local')
    if h == w:
        add('transpose',a.T, 'local'); add('antitranspose',a[::-1,::-1].T,'local')
    else:
        add('transpose',10,'local'); add('antitranspose',10,'local')
    for c in range(10):
        mask = a == c
        rows = mask.sum(1)[:,None] + np.zeros_like(a)
        columns = mask.sum(0)[None,:] + np.zeros_like(a)
        add(f'color{c}_count',mask.sum(),'color')
        add(f'color{c}_rowcount',rows,'rays'); add(f'color{c}_colcount',columns,'rays')
        y,x=np.nonzero(mask)
        if len(y):
            y0,y1,x0,x1=y.min(),y.max(),x.min(),x.max()
            add(f'color{c}_inbbox',(yy>=y0)&(yy<=y1)&(xx>=x0)&(xx<=x1),'color')
            add(f'color{c}_bboxdy',np.maximum(y0-yy,0)+np.maximum(yy-y1,0),'color')
            add(f'color{c}_bboxdx',np.maximum(x0-xx,0)+np.maximum(xx-x1,0),'color')
            dist = ndi.distance_transform_cdt(~mask,metric='taxicab')
            add(f'color{c}_distance',dist,'color')
        else:
            for suffix in ('inbbox','bboxdy','bboxdx','distance'): add(f'color{c}_{suffix}',-1,'color')
        # Counts on each side support intervals, parity, rays, and crossings.
        for axis, label in [(0,'vertical'),(1,'horizontal')]:
            forward=np.cumsum(mask,axis=axis)-mask
            backward=np.flip(np.cumsum(np.flip(mask,axis=axis),axis=axis),axis=axis)-mask
            add(f'color{c}_{label}_before',forward,'rays')
            add(f'color{c}_{label}_after',backward,'rays')
    # Component geometry for both monochrome regions and foreground objects.
    for conn in (1,2):
        structure=ndi.generate_binary_structure(2,conn)
        maps={name:np.zeros_like(a) for name in (
            'area','height','width','top','left','bottom','right','row','col',
            'edge','holes','colors','color','fill','perimeter','rankarea',
            'nobjects','rankrow','rankcol','sym_ud','sym_lr')}
        groups=[]
        for c in range(10):
            labels,num=ndi.label(a==c,structure)
            for label,sl in enumerate(ndi.find_objects(labels),1):
                if sl is None: continue
                sy,sx=sl; sublabels=labels[sl]
                mask=sublabels==label
                groups.append((sy,sx,mask,c))
        def component_maps(groups, prefix):
            areas=[int(m.sum()) for _,_,m,_ in groups]
            rows=sorted(set(sy.start for sy,_,_,_ in groups))
            xs=sorted(set(sx.start for _,sx,_,_ in groups))
            for sy,sx,mask,c in groups:
                ah,aw=mask.shape; area=int(mask.sum())
                gy,gx=np.indices(mask.shape); filled=ndi.binary_fill_holes(mask)
                holes=int((filled&~mask).sum())
                erosion=ndi.binary_erosion(mask)
                properties={'area':area,'height':ah,'width':aw,
                    'top':sy.start,'left':sx.start,'bottom':h-sy.stop,'right':w-sx.stop,
                    'row':gy,'col':gx,'edge':int(sy.start==0 or sx.start==0 or sy.stop==h or sx.stop==w),
                    'holes':holes,'colors':len(np.unique(a[sy,sx][mask])),
                    'color':c,'fill':area==ah*aw,'perimeter':int((mask&~erosion).sum()),
                    'rankarea':sum(z<area for z in areas),'nobjects':len(groups),
                    'rankrow':rows.index(sy.start),'rankcol':xs.index(sx.start),
                    'sym_ud':int(np.array_equal(mask,mask[::-1])),
                    'sym_lr':int(np.array_equal(mask,mask[:,::-1]))}
                for k,v in properties.items():
                    view=maps[k][sy,sx]
                    view[mask]=v[mask] if isinstance(v,np.ndarray) else v
            for name,v in maps.items(): add(prefix+name,v,'objects')
        component_maps(groups,f'same{conn}_')
        for bgval,bgname in [(0,'zero'),(bg,'mode')]:
            for v in maps.values(): v.fill(0)
            labels,num=ndi.label(a!=bgval,structure)
            groups=[]
            for i,sl in enumerate(ndi.find_objects(labels),1):
                if sl is None: continue
                sy,sx=sl; mask=labels[sl]==i
                vals=a[sl][mask]
                c=int(np.bincount(vals,minlength=10).argmax())
                groups.append((sy,sx,mask,c))
            component_maps(groups,f'fg_{bgname}{conn}_')
    # Nearest non-background and nearest differing cell in eight directions.
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        raycolor=np.full_like(a,10); raydistance=np.full_like(a,31)
        diffcolor=np.full_like(a,10); diffdistance=np.full_like(a,31)
        for y in range(h):
            for x in range(w):
                for dist in range(1,31):
                    ny,nx=y+dy*dist,x+dx*dist
                    if not (0<=ny<h and 0<=nx<w): break
                    if raycolor[y,x]==10 and a[ny,nx]!=bg:
                        raycolor[y,x]=a[ny,nx];raydistance[y,x]=dist
                    if diffcolor[y,x]==10 and a[ny,nx]!=a[y,x]:
                        diffcolor[y,x]=a[ny,nx];diffdistance[y,x]=dist
                    if raycolor[y,x]!=10 and diffcolor[y,x]!=10: break
        for name,v in [('color',raycolor),('distance',raydistance),('diffcolor',diffcolor),('diffdistance',diffdistance)]:
            add(f'ray{dy}_{dx}_{name}',v,'rays')
    return np.stack(cols,axis=1).astype(np.float32),names,tags


def feature_indices(tags, group):
    allowed={'local':{'base','local'},'objects':{'base','objects','color'},
             'rays':{'base','rays','color'},'all':set(tags)}[group]
    return np.array([i for i,t in enumerate(tags) if t in allowed])


def tree_spec(tree,names,indices):
    t=tree.tree_
    def node(i):
        if t.children_left[i] < 0:
            return int(tree.classes_[np.argmax(t.value[i][0])])
        return [names[indices[t.feature[i]]],float(t.threshold[i]),node(t.children_left[i]),node(t.children_right[i])]
    return node(0)


def solve(task, execution='deferred', features_fn=None):
    startcpu,startwall=time.process_time(),time.perf_counter()
    task=clean_task(task)
    stats=Counter();candidates=[];all_proposals=[]
    cache={}
    def feat(a):
        key=(a.shape,a.tobytes())
        if key not in cache:
            stats['feature_grid_calls']+=1
            stats['feature_cells']+=a.size
            cache[key]=(features if features_fn is None else features_fn)(a)
        return cache[key]
    train=[(np.array(p['input'],dtype=np.int16),np.array(p['output'],dtype=np.int16)) for p in task['train']]
    query=[np.array(p['input'],dtype=np.int16) for p in task['test']]
    for transform_name in ('identity','crop_zero','crop_mode'):
        transformed=[transform(a,transform_name) for a,b in train]
        if any(a.shape!=b.shape for a,(_,b) in zip(transformed,train)):continue
        # Identical transform behavior may differ on a query: preserve all names.
        ftrain=[feat(a) for a in transformed]
        names,tags=ftrain[0][1:]
        y=np.concatenate([b.ravel() for a,b in train])
        boundaries=np.cumsum([0]+[a.size for a in transformed])
        for group in GROUPS:
            ix=feature_indices(tags,group)
            X=np.concatenate([f[0][:,ix] for f in ftrain])
            for depth in DEPTHS:
                model=DecisionTreeClassifier(max_depth=depth,random_state=0,criterion='entropy')
                model.fit(X,y); stats['models_fit']+=1
                predtrain=model.predict(X);stats['model_training_cells']+=len(y)
                fit=bool(np.array_equal(predtrain,y))
                preds=None
                # Eager explicitly materializes every proposal. Deferred delays
                # query features and predictions until the demonstration fit.
                if execution=='eager' or fit:
                    preds=[]
                    for q in query:
                        tq=transform(q,transform_name)
                        fq=feat(tq)[0][:,ix]
                        pq=model.predict(fq).astype(np.int16).reshape(tq.shape).tolist()
                        stats['query_model_grid_calls']+=1;stats['query_model_cells']+=tq.size
                        preds.append(pq)
                proposal={'transform':transform_name,'group':group,'depth':depth,'fits':fit,
                          'nodes':int(model.tree_.node_count)}
                all_proposals.append(proposal)
                if not fit:continue
                loo_exact=loo_cells=loo_total=0
                if len(train)>1:
                    for i in range(len(train)):
                        keep=np.ones(len(y),dtype=bool);keep[boundaries[i]:boundaries[i+1]]=False
                        cv=DecisionTreeClassifier(max_depth=depth,random_state=0,criterion='entropy')
                        cv.fit(X[keep],y[keep]);stats['models_fit']+=1
                        got=cv.predict(X[~keep]);truth=y[~keep]
                        loo_exact+=int(np.array_equal(got,truth));loo_cells+=int((got==truth).sum());loo_total+=len(truth)
                rank=[-loo_exact,-(loo_cells/max(1,loo_total)),int(model.tree_.node_count),
                      ('identity','crop_zero','crop_mode').index(transform_name),GROUPS.index(group),depth]
                candidates.append({'predictions':preds,'program':{'transform':transform_name,'tree':tree_spec(model,names,ix)},
                    'rank':rank,'loo_exact':loo_exact,'loo_cells':loo_cells,'loo_total':loo_total,'proposal':proposal})
    candidates.sort(key=lambda c:(c['rank'],json.dumps(c['predictions'],separators=(',',':'))))
    distinct=[];seen=set()
    for c in candidates:
        key=json.dumps(c['predictions'],separators=(',',':'))
        if key not in seen: seen.add(key);distinct.append(c)
    answers=[]
    for i,q in enumerate(task['test']):
        grids=[]
        for c in distinct:
            g=c['predictions'][i]
            if g not in grids:grids.append(g)
            if len(grids)==2:break
        while len(grids)<2:grids.append(q['input'])
        answers.append({'attempt_1':grids[0],'attempt_2':grids[1]})
    return {'attempts':answers,'candidates':distinct,'consistent_programs':candidates,
            'proposals':all_proposals,'metrics':dict(stats),'cpu_seconds':time.process_time()-startcpu,
            'wall_seconds':time.perf_counter()-startwall,'execution':execution}


def replay(program, grid):
    a=transform(np.asarray(grid,dtype=np.int16),program['transform'])
    X,names,tags=features(a);index={s:i for i,s in enumerate(names)}
    def node(tree, selected):
        if isinstance(tree,int):return np.full(len(selected),tree,dtype=np.int16)
        name,threshold,left,right=tree
        mask=X[selected,index[name]]<=threshold
        out=np.empty(len(selected),dtype=np.int16)
        out[mask]=node(left,selected[mask]);out[~mask]=node(right,selected[~mask])
        return out
    return node(program['tree'],np.arange(a.size)).reshape(a.shape).tolist()

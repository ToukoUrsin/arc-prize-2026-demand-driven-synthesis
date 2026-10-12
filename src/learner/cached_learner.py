"""Task-local exact fit reuse; fixed seed and all fit inputs retained. MIT-0."""
from relational import *

def solve(task, execution='deferred', features_fn=None):
    startcpu,startwall=time.process_time(),time.perf_counter()
    task=clean_task(task)
    stats=Counter();candidates=[];all_proposals=[]
    cache={}
    fitted_models={}; fit_counts=Counter()
    def fitted(X,y,key):
        fit_counts['logical_fit_requests']+=1
        if key not in fitted_models:
            model=DecisionTreeClassifier(max_depth=key[2],random_state=0,criterion='entropy')
            model.fit(X,y)
            fitted_models[key]=model
            fit_counts['actual_model_fits']+=1
        else:
            fit_counts['reused_model_fits']+=1
        return fitted_models[key]
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
        signature=tuple((a.shape,a.tobytes()) for a in transformed)
        ftrain=[feat(a) for a in transformed]
        names,tags=ftrain[0][1:]
        y=np.concatenate([b.ravel() for a,b in train])
        boundaries=np.cumsum([0]+[a.size for a in transformed])
        for group in GROUPS:
            ix=feature_indices(tags,group)
            X=np.concatenate([f[0][:,ix] for f in ftrain])
            for depth in DEPTHS:
                model=fitted(X,y,(signature,group,depth,-1)); stats['models_fit']+=1
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
                        cv=fitted(X[keep],y[keep],(signature,group,depth,i));stats['models_fit']+=1
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
            'wall_seconds':time.perf_counter()-startwall,'execution':execution,'fit_counts':dict(fit_counts)}

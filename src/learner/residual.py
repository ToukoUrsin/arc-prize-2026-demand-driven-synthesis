"""Training-calibrated second-slot arbitration. Pure inference, MIT-0."""
import math

FEATURES=['native_second','learned','base','extra','log_rank','min_depth','log_programs',
    'tree','learned_map','relational','geometric','demonstrations','log_pool',
    'two_joint','two_agree','two_exact','all_joint','all_agree','all_exact',
    'two_joint_count','all_joint_count','no_two_joint','all_joint_perfect',
    'same_shape_input','same_shape_all_targets','ratio_shape_consistent',
    'same_color_set_input','colors_seen_in_targets','input_copy',
    'height','width','log_area','color_count','uniform','native_distinct',
    'two_joint_margin','all_joint_margin']

def key(grid):return tuple(tuple(row) for row in grid)
def candidates(row):return row['baseline']['candidates']+(row.get('extension') or {}).get('candidates',[])
def colors(grid):return set(x for row in grid for x in row)

def features(task, certificate, native):
    """Uses only demonstrations, unlabeled inputs, and label-blind certificates."""
    full=certificate['full'];base=full['baseline']['candidates'];extra=(full.get('extension') or {}).get('candidates',[])
    train=task['train'];n=len(train);folds=certificate['folds'];answers=[]
    target_colors=set().union(*(colors(p['output']) for p in train))
    for qi,q in enumerate(task['test']):
        first=key(native[qi]['attempt_1']);second=key(native[qi]['attempt_2'])
        pool={second:dict(grid=native[qi]['attempt_2'],programs=[],ranks=[],base=False,extra=False,native_second=True)}
        for rank,c in enumerate(base+extra):
            k=key(c['predictions'][qi])
            if k==first:continue
            e=pool.setdefault(k,dict(grid=c['predictions'][qi],programs=[],ranks=[],base=False,extra=False,native_second=False))
            e['programs'].append(c['program']);e['ranks'].append(rank)
            e['base']|=rank<len(base);e['extra']|=rank>=len(base)
        if len(pool)>1:pool.pop(first,None)
        entries=[]
        for k,e in pool.items():
            two_joint=two_agree=two_exact=all_joint=all_agree=all_exact=0
            for fold in folds:
                held=train[fold['omitted']]['output']; cs=candidates(fold['row']);tw=fold['selected']
                two_joint+=any(c['predictions'][0]==held and key(c['predictions'][qi+1])==k for c in tw)
                two_agree+=any(key(c['predictions'][qi+1])==k for c in tw)
                two_exact+=any(c['predictions'][0]==held for c in tw)
                all_joint+=any(c['predictions'][0]==held and key(c['predictions'][qi+1])==k for c in cs)
                all_agree+=any(key(c['predictions'][qi+1])==k for c in cs)
                all_exact+=any(c['predictions'][0]==held for c in cs)
            g=e['grid'];h,w=len(g),len(g[0]);ih,iw=len(q['input']),len(q['input'][0]);gc=colors(g)
            ops=[op for p in e['programs'] for op in p]
            same_targets=all((h,w)==(len(p['output']),len(p['output'][0])) for p in train)
            ratios=all(h*len(p['input'])==ih*len(p['output']) and w*len(p['input'][0])==iw*len(p['output'][0]) for p in train)
            x=[e['native_second'],bool(e['programs']),e['base'],e['extra'],math.log1p(min(e['ranks']) if e['ranks'] else len(base+extra)+1),
               min(map(len,e['programs'])) if e['programs'] else 4,math.log1p(len(e['programs'])),
               any('tree' in op for op in ops),any('learned' in op or 'fit_color' in op for op in ops),
               any(op.startswith('rel') for op in ops),any(op.startswith(('rotate','flip','transpose','crop','scale')) for op in ops),
               n,math.log1p(len(pool)),two_joint/n,two_agree/n,two_exact/n,all_joint/n,all_agree/n,all_exact/n,
               two_joint,all_joint,two_joint==0,all_joint==n,(h,w)==(ih,iw),same_targets,ratios,
               gc==colors(q['input']),gc<=target_colors,k==key(q['input']),h/30,w/30,math.log1p(h*w),len(gc)/10,len(gc)==1,
               first!=second,0,0]
            assert len(x)==len(FEATURES)
            entries.append(dict(grid=g,x=list(map(float,x)),native_second=e['native_second'],
                order=min(e['ranks']) if e['ranks'] else -1,programs=e['programs'],two_joint=two_joint,all_joint=all_joint))
        for e in entries:
            others=[z for z in entries if z is not e]
            e['x'][-2]=(e['two_joint']-max((z['two_joint'] for z in others),default=0))/n
            e['x'][-1]=(e['all_joint']-max((z['all_joint'] for z in others),default=0))/n
        answers.append(entries)
    return answers

def confidence(x,model):
    z=model['intercept']+sum(c*(v-m)/s for v,m,s,c in zip(x,model['mean'],model['scale'],model['coefficients']))
    return z  # Monotone log-odds; avoids needless sigmoid overflow.

def rank(entries,model):
    return sorted(entries,key=lambda e:(-confidence(e['x'],model),not e['native_second'],e['order'],key(e['grid'])))

def predict_from_features(native,certificate,entries,model):
    if model['kind']=='native':return [dict(a) for a in native]
    if model['kind']=='prior':
        from selector import merge
        return merge(native,certificate,model['gate'])
    return [dict(attempt_1=a['attempt_1'],attempt_2=rank(es,model)[0]['grid']) for a,es in zip(native,entries)]

def predict(task,certificate,native,model):
    return predict_from_features(native,certificate,features(task,certificate,native),model)

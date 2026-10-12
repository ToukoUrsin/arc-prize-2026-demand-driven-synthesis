"""Uniform selection; no task IDs, categories or labels. MIT-0."""
def select(baseline,periodic,relational,query_count):
    out=[]
    for i in range(query_count):
        if periodic:
            proposed=[c['predictions'][i] for c in periodic]+list(baseline[i].values())
        else:
            proposed=[baseline[i]['attempt_1']]
            if relational:proposed.append(relational[0]['predictions'][i])
            proposed.append(baseline[i]['attempt_2'])
        grids=[]
        for g in proposed:
            if g not in grids:grids.append(g)
            if len(grids)==2:break
        while len(grids)<2:grids.append(baseline[i]['attempt_1'])
        out.append(dict(zip(['attempt_1','attempt_2'],grids)))
    return out

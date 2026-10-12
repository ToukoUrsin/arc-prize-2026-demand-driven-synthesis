"""Compile demonstration-fitted relational trees into pure unary operations.
MIT-0. The prior relational learner is used unchanged and sees no query inputs.
All demonstration-fitting trees are retained in fixed demonstration-LOO order.
"""
from collections import OrderedDict
import json
import numpy as np
from solver import Operation,pack,unpack
from screened_grammar import ScreenedGrammar
import relational


class RelationalGrammar:
    def __init__(self,train,features_fn=None,solve_fn=None):
        self.screened=ScreenedGrammar(train)
        self.operations=list(self.screened.operations)
        self.screening_grid_evaluations=self.screened.screening_grid_evaluations
        self.proposed_operations=self.screened.proposed_operations
        self.admitted_operations=self.screened.admitted_operations
        # With an empty query list the unmodified learner constructs and ranks
        # programs using only demonstration fit and fold-local LOO predictions.
        learned=(relational.solve if solve_fn is None else solve_fn)({'train':train,'test':[]},features_fn=features_fn)
        self.constructor_metrics=learned['metrics']
        self.fit_counts=learned.get('fit_counts',{})
        self.programs={};self.feature_cache=OrderedDict();seen=set()
        for candidate in learned['consistent_programs']:
            program=candidate['program']
            encoded=json.dumps(program,sort_keys=True,separators=(',',':'))
            if encoded in seen:continue
            seen.add(encoded)
            op=Operation('relational_tree:'+encoded,None)
            self.operations.append(op);self.programs[id(op)]=program
        self.learned_operations=len(self.programs)

    def clear_cache(self):
        self.screened.clear_cache();self.feature_cache.clear()

    def primitive(self,program,s):
        a=relational.transform(unpack(s).astype(np.int16),program['transform'])
        key=pack(a)
        if key not in self.feature_cache:
            X,names,tags=relational.features(a)
            self.feature_cache[key]=(X,{name:i for i,name in enumerate(names)})
            if len(self.feature_cache)>128:self.feature_cache.popitem(last=False)
        X,index=self.feature_cache[key]
        self.feature_cache.move_to_end(key)
        out=np.empty(a.size,dtype=np.uint8)
        def node(tree,indices):
            if isinstance(tree,int):out[indices]=tree;return
            name,threshold,left,right=tree
            mask=X[indices,index[name]]<=threshold
            if mask.any():node(left,indices[mask])
            if (~mask).any():node(right,indices[~mask])
        node(program['tree'],np.arange(a.size))
        return pack(out.reshape(a.shape))

    def apply(self,op,values):
        if id(op) not in self.programs:return self.screened.apply(op,values)
        return tuple(self.primitive(self.programs[id(op)],s) for s in values)

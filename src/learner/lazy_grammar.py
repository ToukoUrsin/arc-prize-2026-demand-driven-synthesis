"""Exact lazy evaluation of learned relational trees. MIT-0."""
import numpy as np
from solver import pack,unpack
import relational
from relational_grammar import RelationalGrammar
from fast_features import FeatureGrid,features


class LazyGrammar(RelationalGrammar):
    def __init__(self,train):
        super().__init__(train,features_fn=features)
        self.feature_counts={'feature_grids':0,'requested_columns':0,'computed_columns':0}

    def primitive(self,program,s):
        a=relational.transform(unpack(s).astype(np.int16),program['transform'])
        key=pack(a)
        if key not in self.feature_cache:
            self.feature_cache[key]=FeatureGrid(a)
            self.feature_counts['feature_grids']+=1
            if len(self.feature_cache)>128:self.feature_cache.popitem(last=False)
        features=self.feature_cache[key];self.feature_cache.move_to_end(key)
        out=np.empty(a.size,dtype=np.uint8)
        def node(tree,indices):
            if isinstance(tree,int):out[indices]=tree;return
            name,threshold,left,right=tree
            self.feature_counts['requested_columns']+=1
            self.feature_counts['computed_columns']+=int(name not in features.cache)
            mask=features.column(name)[indices]<=threshold
            if mask.any():node(left,indices[mask])
            if (~mask).any():node(right,indices[~mask])
        node(program['tree'],np.arange(a.size))
        return pack(out.reshape(a.shape))

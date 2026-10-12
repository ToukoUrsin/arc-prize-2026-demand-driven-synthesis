"""Demonstration-screened unary constructors, fixed before search. MIT-0.

The unfiltered object catalog is deliberately not the searched grammar. A new
operation is admitted iff its direct application to every original input is
defined and fits all output grids, either literally or under one consistent
terminal color map. No query inputs or outputs enter this screening. There is
no semantic deduplication: distinct admitted operations remain distinct.
"""
import numpy as np
from solver import pack
from solver_extended import fit_map
from object_grammar import ObjectGrammar


class ScreenedGrammar:
    def __init__(self,train):
        self.catalog=ObjectGrammar(train)
        self.catalog.clear_cache()
        self.operations=list(self.catalog.base.operations)
        values=tuple(pack(np.asarray(p['input'],dtype=np.uint8)) for p in train)
        goals=tuple(pack(np.asarray(p['output'],dtype=np.uint8)) for p in train)
        self.screening_grid_evaluations=0
        self.proposed_operations=len(self.catalog.operations)-len(self.operations)
        for op in self.catalog.operations[len(self.operations):]:
            predictions=[]
            for value in values:
                self.screening_grid_evaluations+=1
                result=self.catalog.apply(op,(value,))
                if result is None:break
                predictions.append(result[0])
            if len(predictions)!=len(values):continue
            predictions=tuple(predictions)
            if predictions==goals or fit_map(predictions,goals) is not None:
                self.operations.append(op)
        self.admitted_operations=len(self.operations)-len(self.catalog.base.operations)

    def clear_cache(self):self.catalog.clear_cache()
    def apply(self,op,values):return self.catalog.apply(op,values)

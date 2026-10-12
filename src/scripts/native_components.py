"""Optional exact C++ component selector. No search or grammar changes.
SPDX-License-Identifier: MIT-0
"""
import ctypes
from pathlib import Path
import numpy as np


def install():
    import solver
    lib=ctypes.CDLL(str(Path(__file__).resolve().parent.parent/'data/components.so'))
    fn=lib.select_component
    fn.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_int,ctypes.c_int,
                 ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                 ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_int)]
    fn.restype=ctypes.c_int
    def select(a,bg,same,smallest,crop):
        a=np.ascontiguousarray(a,dtype=np.uint8)
        out=np.empty(900,dtype=np.uint8);dims=(ctypes.c_int*2)()
        success=fn(a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),*a.shape,bg,int(same),
                   int(smallest),int(crop),out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),dims)
        if not success:return None
        return out[:dims[0]*dims[1]].reshape(dims[0],dims[1])
    original=solver.select_object
    solver.select_object=select
    return original,select


if __name__=='__main__':
    import itertools,json,random,time
    rng=random.Random(42)
    original,native=install();checks=0;started=time.perf_counter()
    for iteration in range(200):
        h,w=rng.randrange(1,31),rng.randrange(1,31)
        a=np.array([[rng.randrange(4) for _ in range(w)] for _ in range(h)],dtype=np.uint8)
        for bg,same,small,crop in itertools.product((0,1,2,3),(False,True),(False,True),(False,True)):
            x,y=original(a,bg,same,small,crop),native(a,bg,same,small,crop)
            assert (x is None and y is None) or (x is not None and y is not None and np.array_equal(x,y))
            checks+=1
    dest=Path('results/native-checks');dest.mkdir(parents=True,exist_ok=True)
    (dest/'evidence.json').write_text(json.dumps({'semantic_equivalence_checks':checks,'seconds':time.perf_counter()-started},indent=2)+'\n')
    print(f'Passed {checks} native/Python component comparisons.')

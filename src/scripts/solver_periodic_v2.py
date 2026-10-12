"""Minimum-description periodic denoising inside delimiter-separated panels.
SPDX-License-Identifier: MIT-0
"""
import numpy as np
from scipy import ndimage as ndi


def denoise_line(line,min_repeats,penalty,max_fraction):
    n=len(line);best=None
    if n<2*min_repeats:return line
    for p in range(1,n//min_repeats+1):
        motif=np.array([np.bincount(line[i::p],minlength=10).argmax() for i in range(p)],dtype=np.uint8)
        pred=motif[np.arange(n)%p]
        errors=int(np.count_nonzero(pred!=line))
        if errors>max_fraction*n:continue
        # Literal motif symbols plus penalized correction locations.
        rank=(p+penalty*errors,errors,p,tuple(motif))
        if best is None or rank<best[0]:best=(rank,pred)
    return line if best is None else best[1]


def periodic_repair(a,separator,axis,min_repeats,penalty,max_fraction,scope):
    b=a.copy();h,w=a.shape
    if scope=='whole':regions=[(slice(0,h),slice(0,w))]
    else:
        labels,n=ndi.label(a!=separator)
        regions=[]
        for i,sl in enumerate(ndi.find_objects(labels),1):
            if sl is None:continue
            sy,sx=sl
            # Solid rectangular panels avoid cutting through arbitrary objects.
            if not np.all(labels[sl]==i):continue
            if scope=='interior' and (sy.start==0 or sx.start==0 or sy.stop==h or sx.stop==w):continue
            regions.append(sl)
    for sy,sx in regions:
        region=a[sy,sx];target=b[sy,sx]
        if axis==0:region=region.T;target=target.T
        for i,line in enumerate(region):
            target[i]=denoise_line(line,min_repeats,penalty,max_fraction)
    return b if np.any(b!=a) else None


def role_repair(a,selector,axis,min_repeats,penalty,max_fraction,scope):
    h,w=a.shape
    counts=np.bincount(a.ravel(),minlength=10)
    present=[int(c) for c in np.flatnonzero(counts)]
    if selector=='all':separators=present
    elif selector=='second':
        ranked=sorted(present,key=lambda c:(-int(counts[c]),c))
        if len(ranked)<2:return None
        separators=[ranked[1]]
    else:raise ValueError(selector)
    regions=set()
    if scope=='whole':regions.add((0,h,0,w))
    else:
        for sep in separators:
            labels,n=ndi.label(a!=sep)
            for i,sl in enumerate(ndi.find_objects(labels),1):
                if sl is None:continue
                sy,sx=sl
                if scope=='interior' and (sy.start==0 or sx.start==0 or sy.stop==h or sx.stop==w):continue
                if not np.all(labels[sl]==i):continue
                regions.add((sy.start,sy.stop,sx.start,sx.stop))
    chosen=a.copy();edits=np.zeros(a.shape,dtype=bool)
    for y0,y1,x0,x1 in sorted(regions):
        source=a[y0:y1,x0:x1];target=source.copy()
        current_axis=(1 if source.shape[1]>=source.shape[0] else 0) if axis=='long' else axis
        if current_axis==0:source=source.T;target=target.T
        for i,line in enumerate(source):target[i]=denoise_line(line,min_repeats,penalty,max_fraction)
        if current_axis==0:target=target.T
        before=a[y0:y1,x0:x1];mask=target!=before
        old=chosen[y0:y1,x0:x1];already=edits[y0:y1,x0:x1]
        if np.any(mask&already&(old!=target)):return None
        old[mask]=target[mask];already|=mask
    return chosen if edits.any() else None


def operations(train):
    from solver import Operation
    out=[]
    for scope in ('interior','panels','whole'):
     for selector in (('all','second') if scope!='whole' else ('all',)):
      for axis in ('long',1,0):
       for min_repeats in (3,2,4):
        for penalty in (2,3,4):
         for max_fraction in (0.2,0.1,0.3):
          name=f'role_periodic_{scope}_{selector}_axis{axis}_reps{min_repeats}_penalty{penalty}_fraction{max_fraction}'
          out.append(Operation(name,lambda a,selector=selector,axis=axis,min_repeats=min_repeats,penalty=penalty,max_fraction=max_fraction,scope=scope:
            role_repair(a,selector,axis,min_repeats,penalty,max_fraction,scope)))
    return out

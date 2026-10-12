"""Exact demand-driven implementation of the frozen relational features. MIT-0.

Input-only; names, int16 conversion and column order match relational.features.
An instance belongs to one immutable grid. No task identity or output is used.
"""
import numpy as np
from scipy import ndimage as ndi

PROPERTIES = ('area','height','width','top','left','bottom','right','row','col',
              'edge','holes','colors','color','fill','perimeter','rankarea',
              'nobjects','rankrow','rankcol','sym_ud','sym_lr')
DIRECTIONS = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))


def schema():
    names, tags = [], []
    def add(name, tag): names.append(name); tags.append(tag)
    for n in ('color','nonzero','nonbackground','background'): add(n,'base')
    for n in ('row','column','bottom','right','row_parity','col_parity','border',
              'diagonal','antidiagonal','h','w'): add(n,'position')
    for dy in range(-2,3):
        for dx in range(-2,3):
            if dy or dx: add(f'neighbor_{dy}_{dx}','local')
    for n in ('mirror_ud','mirror_lr','mirror_both','transpose','antitranspose'): add(n,'local')
    for c in range(10):
        for n,t in (('count','color'),('rowcount','rays'),('colcount','rays'),
                    ('inbbox','color'),('bboxdy','color'),('bboxdx','color'),('distance','color'),
                    ('vertical_before','rays'),('vertical_after','rays'),
                    ('horizontal_before','rays'),('horizontal_after','rays')): add(f'color{c}_{n}',t)
    for conn in (1,2):
        for prefix in (f'same{conn}_',f'fg_zero{conn}_',f'fg_mode{conn}_'):
            for name in PROPERTIES: add(prefix+name,'objects')
    for dy,dx in DIRECTIONS:
        for n in ('color','distance','diffcolor','diffdistance'): add(f'ray{dy}_{dx}_{n}','rays')
    return names,tags


NAMES,TAGS=schema()


class FeatureGrid:
    def __init__(self,a):
        self.a=np.asarray(a,dtype=np.int16)
        self.h,self.w=self.a.shape
        self.bg=int(np.bincount(self.a.ravel(),minlength=10).argmax())
        self.yy,self.xx=np.indices(self.a.shape)
        self.cache={};self.groups={};self.masks={}

    def column(self,name):
        if name not in self.cache:
            value=self.compute(name)
            # Original add() broadcasts scalars, copies to int16, then flattens.
            if np.isscalar(value): value=np.full_like(self.a,value)
            self.cache[name]=np.array(value,dtype=np.int16,copy=True).ravel()
        return self.cache[name]

    def compute(self,name):
        a=self.a;h,w=self.h,self.w;y,x=self.yy,self.xx
        if name=='color': return a
        if name=='nonzero': return a!=0
        if name=='nonbackground': return a!=self.bg
        if name=='background': return self.bg
        if name=='row': return y
        if name=='column': return x
        if name=='bottom': return h-1-y
        if name=='right': return w-1-x
        if name=='row_parity': return y%2
        if name=='col_parity': return x%2
        if name=='border': return np.minimum.reduce([y,x,h-1-y,w-1-x])
        if name=='diagonal': return y-x
        if name=='antidiagonal': return y+x
        if name=='h': return h
        if name=='w': return w
        if name.startswith('neighbor_'):
            _,dy,dx=name.split('_');dy,dx=int(dy),int(dx)
            if 'pad' not in self.groups:self.groups['pad']=np.pad(a,2,constant_values=10)
            return self.groups['pad'][2+dy:2+dy+h,2+dx:2+dx+w]
        if name=='mirror_ud':return a[::-1]
        if name=='mirror_lr':return a[:,::-1]
        if name=='mirror_both':return a[::-1,::-1]
        if name=='transpose':return a.T if h==w else 10
        if name=='antitranspose':return a[::-1,::-1].T if h==w else 10
        if name.startswith('color'):
            head,suffix=name.split('_',1);c=int(head[5:])
            if c not in self.masks:self.masks[c]=a==c
            mask=self.masks[c]
            if suffix=='count':return mask.sum()
            if suffix=='rowcount':return np.broadcast_to(mask.sum(1)[:,None],a.shape)
            if suffix=='colcount':return np.broadcast_to(mask.sum(0)[None,:],a.shape)
            if suffix in ('inbbox','bboxdy','bboxdx','distance'):
                yy,xx=np.nonzero(mask)
                if not len(yy):return -1
                if suffix=='distance':return ndi.distance_transform_cdt(~mask,metric='taxicab')
                y0,y1,x0,x1=yy.min(),yy.max(),xx.min(),xx.max()
                if suffix=='inbbox':return (y>=y0)&(y<=y1)&(x>=x0)&(x<=x1)
                if suffix=='bboxdy':return np.maximum(y0-y,0)+np.maximum(y-y1,0)
                return np.maximum(x0-x,0)+np.maximum(x-x1,0)
            axis,label=(0,'vertical') if suffix.startswith('vertical') else (1,'horizontal')
            if suffix==label+'_before':return np.cumsum(mask,axis=axis)-mask
            if suffix==label+'_after':return np.flip(np.cumsum(np.flip(mask,axis=axis),axis=axis),axis=axis)-mask
        if name.startswith(('same','fg_')):
            prefix,prop=name.rsplit('_',1)
            # sym_ud and sym_lr are two-token properties.
            if prop in ('ud','lr'):prefix,prop=name.rsplit('_',2)[0],'sym_'+prop
            conn=int(prefix[-1]);kind=prefix[:-1]
            bg=None if kind=='same' else (0 if kind=='fg_zero' else self.bg)
            key=('components',conn,bg)
            if key not in self.groups:self.groups[key]=self.component_maps(conn,bg)
            return self.groups[key][prop]
        if name.startswith('ray'):
            dy,dx,suffix=name[3:].split('_',2);dy,dx=int(dy),int(dx)
            key=('ray',dy,dx)
            if key not in self.groups:self.groups[key]=self.rays(dy,dx)
            return self.groups[key][suffix]
        raise KeyError(name)

    def component_maps(self,conn,bg):
        a=self.a;h,w=self.h,self.w
        structure=ndi.generate_binary_structure(2,conn)
        groups=[]
        for c in (range(10) if bg is None else (None,)):
            labels,_=ndi.label(a==c if bg is None else a!=bg,structure)
            for label,sl in enumerate(ndi.find_objects(labels),1):
                if sl is None:continue
                sy,sx=sl;mask=labels[sl]==label
                color=c if bg is None else int(np.bincount(a[sl][mask],minlength=10).argmax())
                groups.append((sy,sx,mask,color))
        maps={name:np.zeros_like(a) for name in PROPERTIES}
        areas=[int(m.sum()) for _,_,m,_ in groups]
        rows=sorted(set(sy.start for sy,_,_,_ in groups))
        cols=sorted(set(sx.start for _,sx,_,_ in groups))
        for sy,sx,mask,c in groups:
            ah,aw=mask.shape;area=int(mask.sum());gy,gx=np.indices(mask.shape)
            properties={'area':area,'height':ah,'width':aw,'top':sy.start,'left':sx.start,
                'bottom':h-sy.stop,'right':w-sx.stop,'row':gy,'col':gx,
                'edge':int(sy.start==0 or sx.start==0 or sy.stop==h or sx.stop==w),
                'holes':int((ndi.binary_fill_holes(mask)&~mask).sum()),
                'colors':len(np.unique(a[sy,sx][mask])),'color':c,'fill':area==ah*aw,
                'perimeter':int((mask&~ndi.binary_erosion(mask)).sum()),
                'rankarea':sum(z<area for z in areas),'nobjects':len(groups),
                'rankrow':rows.index(sy.start),'rankcol':cols.index(sx.start),
                'sym_ud':int(np.array_equal(mask,mask[::-1])),
                'sym_lr':int(np.array_equal(mask,mask[:,::-1]))}
            for name,value in properties.items():
                view=maps[name][sy,sx];view[mask]=value[mask] if isinstance(value,np.ndarray) else value
        return maps

    def rays(self,dy,dx):
        a=self.a;h,w=a.shape
        rc=np.full_like(a,10);rd=np.full_like(a,31)
        dc=np.full_like(a,10);dd=np.full_like(a,31)
        # Whole-grid shifted comparisons preserve the original nearest-first
        # scan (including its distance-30 cutoff), without Python cell loops.
        for d in range(1,31):
            sy,sx=dy*d,dx*d
            if abs(sy)>=h or abs(sx)>=w:break
            dst=(slice(max(0,-sy),min(h,h-sy)),slice(max(0,-sx),min(w,w-sx)))
            src=(slice(max(0,sy),min(h,h+sy)),slice(max(0,sx),min(w,w+sx)))
            v=a[src]
            choose=(rc[dst]==10)&(v!=self.bg)
            rc[dst][choose]=v[choose];rd[dst][choose]=d
            choose=(dc[dst]==10)&(v!=a[dst])
            dc[dst][choose]=v[choose];dd[dst][choose]=d
        return dict(color=rc,distance=rd,diffcolor=dc,diffdistance=dd)


def features(a):
    grid=FeatureGrid(a)
    return np.stack([grid.column(name) for name in NAMES],axis=1).astype(np.float32),list(NAMES),list(TAGS)

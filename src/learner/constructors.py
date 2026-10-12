"""Demonstration-induced finite object-color programs. MIT-0.

Each rule maps one input-derived object property to a constant color or to
identity. Unseen property values preserve the input object's colors. Learned
rules are retained only after exact replay of every demonstration. The search
then treats each immutable rule as one deterministic unary operation.
"""
from collections import Counter
import json
import numpy as np
from solver import pack
from object_grammar import objects, background

FEATURES = ('area', 'area_rank', 'box', 'box_rank', 'holes', 'colors',
            'shape_frequency', 'row_rank', 'col_rank', 'density',
            'border_count', 'aspect', 'area_parity', 'symmetric_lr', 'symmetric_ud')


def values(s, objs, feature):
    h,w=s[0],s[1]
    counts=Counter(o[8] for o in objs)
    sizes=sorted({len(o[0]) for o in objs}); boxes=sorted({o[5] for o in objs})
    rows=sorted({o[1] for o in objs}); columns=sorted({o[3] for o in objs})
    result=[]
    for o in objs:
        cells,y0,y1,x0,x1,box,holes,ncolors,shape=o
        if feature=='area': value=len(cells)
        elif feature=='area_rank': value=sizes.index(len(cells))
        elif feature=='box': value=box
        elif feature=='box_rank': value=boxes.index(box)
        elif feature=='holes': value=holes
        elif feature=='colors': value=ncolors
        elif feature=='shape_frequency': value=counts[shape]
        elif feature=='row_rank': value=rows.index(y0)
        elif feature=='col_rank': value=columns.index(x0)
        elif feature=='density':
            from math import gcd
            g=gcd(len(cells),box); value=f'{len(cells)//g}/{box//g}'
        elif feature=='border_count': value=sum(p//w in (0,h-1) or p%w in (0,w-1) for p in cells)
        elif feature=='aspect': value=(y1-y0>x1-x0)-(y1-y0<x1-x0)
        elif feature=='area_parity': value=len(cells)%2
        elif feature=='symmetric_lr': value=int({(p//w,x0+x1-p%w) for p in cells}=={divmod(p,w) for p in cells})
        elif feature=='symmetric_ud': value=int({(y0+y1-p//w,p%w) for p in cells}=={divmod(p,w) for p in cells})
        else: raise ValueError(feature)
        result.append(str(value))
    return result


def apply_rule(s, spec):
    role,kind,feature,mapping=spec
    objs=objects(s,role,kind); out=bytearray(s[2:])
    for o,value in zip(objs,values(s,objs,feature)):
        color=mapping.get(value,-1)
        if color>=0:
            for p in o[0]: out[p]=color
    return s[:2]+bytes(out)


def construct(train):
    pairs=[(pack(np.asarray(p['input'],dtype=np.uint8)),pack(np.asarray(p['output'],dtype=np.uint8))) for p in train]
    if any(a[:2]!=b[:2] for a,b in pairs): return []
    rules=[]
    for role in ('zero','mode'):
        for kind in ('fg4','mono4','mono8','colors'):
            examples=[]; valid=True
            for a,b in pairs:
                objs=objects(a,role,kind); labels=[]
                bg=background(a,role)
                if any(x==bg and y!=x for x,y in zip(a[2:],b[2:])):
                    valid=False; break
                for o in objs:
                    # Prefer identity whenever observed colors are unchanged;
                    # otherwise require a uniform target over the entire object.
                    if all(a[2+p]==b[2+p] for p in o[0]): labels.append(-1)
                    else:
                        colors={b[2+p] for p in o[0]}
                        if len(colors)!=1:valid=False;break
                        labels.append(next(iter(colors)))
                if not valid:break
                examples.append((a,objs,labels))
            if not valid:continue
            for feature in FEATURES:
                allowed={}
                for a,objs,labels in examples:
                    for value,label,o in zip(values(a,objs,feature),labels,objs):
                        # An unchanged monochrome object permits both identity
                        # and a constant equal to its input color.
                        options={label}
                        if label==-1 and len({a[2+p] for p in o[0]})==1:
                            options.add(a[2+o[0][0]])
                        allowed[value]=allowed.get(value,options)&options
                if not allowed or any(not v for v in allowed.values()):continue
                mapping={k:min(v) for k,v in sorted(allowed.items())}
                if all(v==-1 for v in mapping.values()):continue
                spec=[role,kind,feature,mapping]
                if all(apply_rule(a,spec)==b for a,b in pairs):
                    rules.append(spec)
    return rules


def name(spec): return 'learned_object_color:'+json.dumps(spec,sort_keys=True,separators=(',',':'))

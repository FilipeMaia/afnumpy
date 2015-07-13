from . import numeric as _nx
from .numeric import asanyarray, newaxis

def atleast_1d(*arys):
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if len(ary.shape) == 0 :
            result = ary.reshape(1)
        else :
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res

def atleast_2d(*arys):
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if len(ary.shape) == 0 :
            result = ary.reshape(1, 1)
        elif len(ary.shape) == 1 :
            result = ary[newaxis,:]
        else :
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res

def vstack(tup):
    return _nx.concatenate([atleast_2d(_m) for _m in tup], 0)

    
def hstack(tup):
    arrs = [atleast_1d(_m) for _m in tup]
    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs[0].ndim == 1:
        return _nx.concatenate(arrs, 0)
    else:
        return _nx.concatenate(arrs, 1)

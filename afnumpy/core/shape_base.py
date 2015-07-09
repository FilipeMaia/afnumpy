from . import numeric as _nx

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

    

import numpy
import afnumpy
from .. import private_utils as pu

def all(a, axis=None, out=None, keepdims=False):
    if(out is not None):
        raise NotImplementedError
    if(keepdims is not False):
        raise NotImplementedError
    if(axis is None):
        for i in range(len(a.shape)-1,-1,-1):
            s = afnumpy.arrayfire.allTrue(a.d_array, pu.c2f(a.shape, i)) 
            a = afnumpy.ndarray(pu.af_shape(s), dtype=bool, af_array=s)
    else:
        s = afnumpy.arrayfire.allTrue(a.d_array, pu.c2f(a.shape, axis))
    a = afnumpy.ndarray(pu.af_shape(s), dtype=bool, af_array=s)
    if(axis == -1):
        if(keepdims):
            return numpy.array(a)
        else:
            return numpy.array(a)[0]
    else:
        return a

def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if(out is not None):
        raise NotImplementedError
    if(keepdims is not False):
        raise NotImplementedError
    if(axis is None):
        for i in range(len(a.shape)-1,-1,-1):
            s = afnumpy.arrayfire.sum(a.d_array, pu.c2f(a.shape, i)) 
            a = afnumpy.ndarray(pu.af_shape(s), dtype=a.dtype, af_array=s)
    else:
        s = afnumpy.arrayfire.sum(a.d_array, pu.c2f(a.shape, axis))
    a = afnumpy.ndarray(pu.af_shape(s), dtype=a.dtype, af_array=s)
    if(axis is None):
        if(keepdims):
            return numpy.array(a)
        else:
            return numpy.array(a)[0]
    else:
        return a


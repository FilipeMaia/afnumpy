import numpy
import afnumpy
from afnumpy import private_utils as pu
from afnumpy.decorators import *

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
    shape = pu.af_shape(s)
    if(shape == (1,) and keepdims is False):
        shape = tuple()
    return afnumpy.ndarray(shape, dtype=bool, af_array=s)

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
    shape = pu.af_shape(s)
    if(shape == (1,) and keepdims is False):
        shape = tuple()
    return afnumpy.ndarray(shape, dtype=a.dtype, af_array=s)


def round(a, decimals=0, out=None):
    try:
        return a.round(decimals, out)
    except AttributeError:
        return numpy.round(a, decimals, out)

def take(a, indices, axis=None, out=None, mode='raise'):
    try:
        return a.take(indices, axis, out, mode)
    except AttributeError:
        return numpy.take(a, indices, axis, out, mode)

def amin(a, axis=None, out=None, keepdims=False):
    try:
        return a.min(axis, out, keepdims)
    except AttributeError:
        return numpy.amin(a, axis, out, keepdims)
        
min = amin

def amax(a, axis=None, out=None, keepdims=False):
    try:
        return a.max(axis, out, keepdims)
    except AttributeError:
        return numpy.amax(a, axis, out, keepdims)
        
max = amax

def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    try:
        return a.prod(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    except AttributeError:
        return numpy.prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    try:
        return a.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    except AttributeError:
        return numpy.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


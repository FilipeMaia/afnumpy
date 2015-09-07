import numpy
import afnumpy
import arrayfire
from afnumpy import private_utils as pu
from afnumpy.decorators import *

def all(a, axis=None, out=None, keepdims=False):
    try:
        return a.all(axis, out, keepdims)
    except AttributeError:
        return numpy.all(a, axis, out, keepdims)

def any(a, axis=None, out=None, keepdims=False):
    try:
        return a.any(axis, out, keepdims)
    except AttributeError:
        return numpy.any(a, axis, out, keepdims)

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

def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    try:
        return a.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    except AttributeError:
        return numpy.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

@outufunc
def sqrt(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.sqrt(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.sqrt(x)

def transpose(a, axes=None):
    try:
        return a.transpose(axes)
    except AttributeError:
        return numpy.tranpose(a, axes)

def squeeze(a, axis=None):
    try:
        return a.squeeze(axis)
    except AttributeError:
        return numpy.squeeze(a, axis)

def argmax(a, axis=None):
    try:
        return a.argmax(axis)
    except AttributeError:
        return numpy.argmax(a, axis)

def argmin(a, axis=None):
    try:
        return a.argmin(axis)
    except AttributeError:
        return numpy.argmin(a, axis)

def argsort(a, axis=-1, kind='quicksort', order=None):
    try:
        return a.argsort(axis, kind, order)
    except AttributeError:
        return numpy.argsort(a, axis, kind, order)

import numpy
from .. import private_utils as pu
import afnumpy
from numpy import newaxis
import numbers

def concatenate(arrays, axis=0):
    if(len(arrays) < 1):
        raise ValueError('need at least one array to concatenate')
    if(axis > 3):
        raise NotImplementedError('only up to 4 axis as currently supported')
    arr = arrays[0].d_array.copy()
    axis = pu.c2f(arrays[0].shape, axis)
    for a in arrays[1:]:
        arr = afnumpy.arrayfire.join(axis, arr, a.d_array)
    return afnumpy.ndarray(pu.af_shape(arr), dtype=arrays[0].dtype, af_array=arr)

def roll(a, shift, axis=None):
    shape = a.shape
    if(axis is None):
        axis = 0
        a = a.flatten()
    axis = pu.c2f(a.shape, axis)
    if axis == 0:
        s = afnumpy.arrayfire.shift(a.d_array, shift, 0, 0, 0)
    elif axis == 1:
        s = afnumpy.arrayfire.shift(a.d_array, 0, shift, 0, 0)
    elif axis == 2:
        s = afnumpy.arrayfire.shift(a.d_array, 0, 0, shift, 0)
    elif axis == 3:
        s = afnumpy.arrayfire.shift(a.d_array, 0, 0, 0, shift)
    else:
        raise NotImplementedError
    return afnumpy.ndarray(shape, dtype=a.dtype, af_array=s)        

def ones(shape, dtype=float, order='C'):
    b = numpy.ones(shape, dtype, order)
    return afnumpy.ndarray(b.shape, b.dtype, buffer=b,order=order)

def reshape(a, newshape, order='C'):
    return a.reshape(newshape,order)

def asanyarray(a, dtype=None, order=None):
    return afnumpy.array(a, dtype, copy=False, order=order, subok=True)

def floor(x, out=None):
    s = afnumpy.arrayfire.floor(x.d_array)
    a = afnumpy.ndarray(x.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
    if out is not None:
        out[:] = a[:]
    return a

def ceil(x, out=None):
    s = afnumpy.arrayfire.ceil(x.d_array)
    a = afnumpy.ndarray(x.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
    if out is not None:
        out[:] = a[:]
    return a
            
def abs(x, out=None):
    if not isinstance(x, afnumpy.ndarray):
        return numpy.abs(x, out)
    a = x.__abs__()
    if out is not None:
        out[:] = a
    return a

def asarray(a, dtype=None, order=None):
    return afnumpy.array(a, dtype, copy=False, order=order)

def ascontiguousarray(a, dtype=None):
    return afnumpy.array(a, dtype, copy=False, order='C', ndmin=1)


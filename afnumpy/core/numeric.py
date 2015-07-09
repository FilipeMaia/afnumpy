import numpy
from .. import private_utils as pu
import afnumpy
from numpy import newaxis

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
    if(order is not 'C'):
        raise NotImplementedError
    newshape = numpy.array(pu.c2f(newshape), dtype=pu.dim_t)
    ret, handle = afnumpy.arrayfire.af_moddims(a.d_array.get(), newshape.size, newshape.ctypes.data)
    s = afnumpy.arrayfire.array_from_handle(handle)
    a = afnumpy.ndarray(pu.af_shape(s), dtype=a.dtype, af_array=s)
    return a


def asanyarray(a, dtype=None, order=None):
    return afnumpy.array(a, dtype, copy=False, order=order, subok=True)

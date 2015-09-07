import arrayfire
import numpy
from .. import private_utils as pu
import afnumpy
from numpy import newaxis
import numbers
from numpy import broadcast
from IPython.core.debugger import Tracer


def concatenate(arrays, axis=0):
    if(len(arrays) < 1):
        raise ValueError('need at least one array to concatenate')
    if(axis > 3):
        raise NotImplementedError('only up to 4 axis as currently supported')
    arr = arrays[0].d_array.copy()
    axis = pu.c2f(arrays[0].shape, axis)
    for a in arrays[1:]:
        arr = arrayfire.join(axis, arr, a.d_array)
    return afnumpy.ndarray(pu.af_shape(arr), dtype=arrays[0].dtype, af_array=arr)

def roll(a, shift, axis=None):
    shape = a.shape
    if(axis is None):
        axis = 0
        a = a.flatten()
    axis = pu.c2f(a.shape, axis)
    if axis == 0:
        s = arrayfire.shift(a.d_array, shift, 0, 0, 0)
    elif axis == 1:
        s = arrayfire.shift(a.d_array, 0, shift, 0, 0)
    elif axis == 2:
        s = arrayfire.shift(a.d_array, 0, 0, shift, 0)
    elif axis == 3:
        s = arrayfire.shift(a.d_array, 0, 0, 0, shift)
    else:
        raise NotImplementedError
    return afnumpy.ndarray(shape, dtype=a.dtype, af_array=s)        

def rollaxis(a, axis, start=0):
    n = a.ndim
    if axis < 0:
        axis += n
    if start < 0:
        start += n
    msg = 'rollaxis: %s (%d) must be >=0 and < %d'
    if not (0 <= axis < n):
        raise ValueError(msg % ('axis', axis, n))
    if not (0 <= start < n+1):
        raise ValueError(msg % ('start', start, n+1))
    if (axis < start): # it's been removed
        start -= 1
    if axis==start:
        return a
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.transpose(axes)

def ones(shape, dtype=float, order='C'):
    b = numpy.ones(shape, dtype, order)
    return afnumpy.ndarray(b.shape, b.dtype, buffer=b,order=order)

def reshape(a, newshape, order='C'):
    return a.reshape(newshape,order)

def asanyarray(a, dtype=None, order=None):
    return afnumpy.array(a, dtype, copy=False, order=order, subok=True)

def floor(x, out=None):
    s = arrayfire.floor(x.d_array)
    a = afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    if out is not None:
        out[:] = a[:]
    return a

def ceil(x, out=None):
    s = arrayfire.ceil(x.d_array)
    a = afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
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

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3
    a = asarray(a)
    b = asarray(b)
    # Move working axis to the end of the shape
    a = rollaxis(a, axisa, a.ndim)
    b = rollaxis(b, axisb, b.ndim)
    msg = ("incompatible dimensions for cross product\n"
           "(dimension must be 2 or 3)")
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(msg)

        # Create the output array
    shape = broadcast(a[..., 0], b[..., 0]).shape
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
    dtype = afnumpy.promote_types(a.dtype, b.dtype)
    cp = afnumpy.empty(shape, dtype)

    # create local aliases for readability
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]

    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # a0 * b1 - a1 * b0
            afnumpy.multiply(a0, b1, out=cp)
            cp -= a1 * b0
            if cp.ndim == 0:
                return cp
            else:
                # This works because we are moving the last axis
                return rollaxis(cp, -1, axisc)
        else:
            # cp0 = a1 * b2 - 0  (a2 = 0)
            # cp1 = 0 - a0 * b2  (a2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            afnumpy.multiply(a1, b2, out=cp0)
            afnumpy.multiply(a0, b2, out=cp1)
            negative(cp1, out=cp1)
            afnumpy.multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0
    elif a.shape[-1] == 3:
        if b.shape[-1] == 3:
            # cp0 = a1 * b2 - a2 * b1
            # cp1 = a2 * b0 - a0 * b2
            # cp2 = a0 * b1 - a1 * b0
            afnumpy.multiply(a1, b2, out=cp0)
            tmp = afnumpy.array(a2 * b1)
            cp0 -= tmp
            afnumpy.multiply(a2, b0, out=cp1)
            afnumpy.multiply(a0, b2, out=tmp)
            cp1 -= tmp
            afnumpy.multiply(a0, b1, out=cp2)
            afnumpy.multiply(a1, b0, out=tmp)
            cp2 -= tmp
        else:
            # cp0 = 0 - a2 * b1  (b2 = 0)
            # cp1 = a2 * b0 - 0  (b2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            afnumpy.multiply(a2, b1, out=cp0)
            negative(cp0, out=cp0)
            afnumpy.multiply(a2, b0, out=cp1)
            afnumpy.multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0

    if cp.ndim == 1:
        return cp
    else:
        # This works because we are moving the last axis
        return rollaxis(cp, -1, axisc)

import numpy
import afnumpy
import arrayfire
from .. import private_utils as pu
from ..decorators import *


def fromstring(string, dtype=float, count=-1, sep=''):
    return array(numpy.fromstring(string, dtype, count, sep))

def array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0):
    # We're going to ignore this for now
    # if(subok is not False):
    #     raise NotImplementedError
    if(order is not None and order is not 'K' and order is not 'C'):
        raise NotImplementedError

    # If it's not a numpy or afnumpy array first create a numpy array from it
    if(not isinstance(object, afnumpy.ndarray) and
       not isinstance(object, numpy.ndarray) and
       not isinstance(object, arrayfire.array.Array)):
        object = numpy.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    if isinstance(object, arrayfire.array.Array):
        shape = pu.c2f(object.dims())
    else:
        shape = object.shape
    while(ndmin > len(shape)):
        shape = (1,)+shape
    if(dtype is None):
        if isinstance(object, arrayfire.array.Array):
            dtype = pu.typemap(object.dtype())
        else:
            dtype = object.dtype
    if(isinstance(object, afnumpy.ndarray)):
        if(copy):
            s = arrayfire.cast(object.d_array.copy(), pu.typemap(dtype))
        else:
            s = arrayfire.cast(object.d_array, pu.typemap(dtype))
        a = afnumpy.ndarray(shape, dtype=dtype, af_array=s)
        a._eval()
        return a
    elif(isinstance(object, arrayfire.array.Array)):
        if(copy):
            s = arrayfire.cast(object.copy(), pu.typemap(dtype))
        else:
            s = arrayfire.cast(object, pu.typemap(dtype))
        a = afnumpy.ndarray(shape, dtype=dtype, af_array=s)
        a._eval()
        return a
    elif(isinstance(object, numpy.ndarray)):
        return afnumpy.ndarray(shape, dtype=dtype, buffer=numpy.ascontiguousarray(object.astype(dtype, copy=copy)))
    else:
        raise AssertionError

def arange(start, stop = None, step = None, dtype=None):
    return afnumpy.array(numpy.arange(start,stop,step,dtype))

def empty(shape, dtype=float, order='C'):
    return afnumpy.ndarray(shape, dtype=dtype, order=order)

def zeros(shape, dtype=float, order='C'):
    b = numpy.zeros(shape, dtype, order)
    return afnumpy.ndarray(b.shape, b.dtype, buffer=b,order=order)

def where(condition, x=pu.dummy, y=pu.dummy):
    a = condition
    s = arrayfire.where(a.d_array)
    # numpy uses int64 while arrayfire uses uint32
    s = afnumpy.ndarray(pu.af_shape(s), dtype=numpy.uint32, af_array=s).astype(numpy.int64)
    # Looks like where goes through the JIT??
    s.eval()
    if(x is pu.dummy and y is pu.dummy):
        idx = []
        mult = 1
        for i in a.shape[::-1]:
            mult = i
            idx = [s % mult] + idx
            s //= mult
        idx = tuple(idx)
        return idx
    elif(x is not pu.dummy and y is not pu.dummy):
        if(x.dtype != y.dtype):
            raise TypeError('x and y must have same dtype')
        if(x.shape != y.shape):
            raise ValueError('x and y must have same shape')
        ret = afnumpy.array(y)
        if(len(ret.shape) > 1):
            ret = ret.flatten()
            ret[s] = x.flatten()[s]
            ret = ret.reshape(x.shape)
        else:
            ret[s] = x[s]
        return ret;
    else:
        raise ValueError('either both or neither of x and y should be given')

def concatenate(arrays, axis=0):
    arrays = tuple(arrays)
    if len(arrays) == 0:
        raise ValueError('need at least one array to concatenate')
    base = arrays[0]
    if len(arrays) == 1:
        return base.copy()
    # arrayfire accepts at most 4 arrays to concatenate at once so we'll have
    # to chunk the arrays
    # The first case is special as we don't want to create unnecessary copies
    i = 0
    a = arrays[i].d_array
    if i+1 < len(arrays):
        b = arrays[i+1].d_array
    else:
        b = None
    if i+2 < len(arrays):
        c = arrays[i+2].d_array
    else:
        c = None
    if i+3 < len(arrays):
        d = arrays[i+3].d_array
    else:
        d = None            
    ret = arrayfire.join(pu.c2f(arrays[0].shape, axis), a, b, c, d)

    for i in range(4,len(arrays),4):
        a = ret.d_array
        if i < len(arrays):
            b = arrays[i].d_array
        else:
            b = None
        if i+1 < len(arrays):
            c = arrays[i+1].d_array
        else:
            c = None
        if i+2 < len(arrays):
            d = arrays[i+2].d_array
        else:
            d = None
            
        ret = arrayfire.join(pu.c2f(arrays[0].shape, axis), a, b, c, d)
    
    return ret

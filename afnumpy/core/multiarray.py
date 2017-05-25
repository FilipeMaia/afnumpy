import numpy
import afnumpy
import arrayfire
from .. import private_utils as pu
from ..decorators import *

def array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0):
    # We're going to ignore this for now
    # if(subok is not False):
    #     raise NotImplementedError
    if(order is not None and order is not 'K' and order is not 'C'):
        raise NotImplementedError

    # If it's not a numpy or afnumpy array first create a numpy array from it
    if(not isinstance(object, ndarray) and
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

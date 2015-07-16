from IPython.core.debugger import Tracer
import afnumpy
import private_utils as pu

def outufunc(func):
    def wrapper(*args, **kws):
        out = kws.pop('out', None)
        ret = func(*args, **kws)
        if out is not None:
            if(out.ndim):
                out[:] = ret
            else:
                out[()] = ret
        return ret
    return wrapper

def iufunc(func):
    def wrapper(*args, **kws):
        if all(isinstance(A, afnumpy.ndarray) for A in args):
            bcast_args = afnumpy.broadcast_arrays(*args)
            if(bcast_args[0].shape is not args[0].shape):
                raise ValueError("non-broadcastable output operand with"
                                 " shape %s doesn't match the broadcast"
                                 " shape %s" % (args[0].shape, bcast_args[0].shape))
            args = bcast_args
            
        return func(*args, **kws)
    return wrapper

def ufunc(func):
    def wrapper(*args, **kws):
        if all(isinstance(A, afnumpy.ndarray) for A in args):
            args = afnumpy.broadcast_arrays(*args)            
        return func(*args, **kws)
    return wrapper

def reductufunc(func):
    def wrapper(a, axis=None, dtype=None, keepdims=False):        
        if not isinstance(axis, tuple):
            axis = (axis,)
        if(a.d_array):
            if axis[0] is None:
                a = a.flat
                axis = (0,)
            shape = list(a.shape)
            s = a.d_array
            # Do in decreasing axis order to avoid problems with the pop
            for ax in sorted(axis)[::-1]:
                s = func(a, s, pu.c2f(a.shape, ax))
                if keepdims:
                    shape[ax] = 1
                else:
                    shape.pop(ax)
            ret = afnumpy.ndarray(tuple(shape), dtype=pu.typemap(s.type()), 
                                  af_array=s)
            if(dtype is not None):
                ret = ret.astype(dtype)
            if(len(shape) == 0):
                ret = ret[()]
            return ret
        else:
            # Call the host array function instead
            return getattr(a.h_array,func.__name__)(axis=None, dtype=dtype, keepdims=keepdims)
    return wrapper

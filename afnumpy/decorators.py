from IPython.core.debugger import Tracer
import afnumpy
import private_utils as pu

def outufunc(func):
    # This could use some optimization
    def wrapper(*args, **kws):
        out = kws.pop('out', None)
        ret = func(*args, **kws)
        if out is not None:
            if(out.ndim):
                out[:] = ret
            else:
                out[()] = ret
            return out
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
            
        ret = func(*args, **kws)
        if len(ret.shape) == 0:
            return ret[()]
        return ret
    return wrapper

def ufunc(func):
    def wrapper(*args, **kws):
        if all(isinstance(A, afnumpy.ndarray) for A in args):
            args = afnumpy.broadcast_arrays(*args)            
        ret = func(*args, **kws)
        if len(ret.shape) == 0:
            return ret[()]
        return ret
    return wrapper

def reductufunc(func):
    def wrapper(a, axis=None, dtype=None, keepdims=False):        
        if not isinstance(axis, tuple):
            axis = (axis,)
        if axis[0] is None:
            # Special case for speed
            a_flat = a.flat
            s = func(a_flat, a_flat.d_array, pu.c2f(a_flat.shape, 0))
            if keepdims:
                shape = (1,)*a.ndim
            else:
                shape = ()
            ret = afnumpy.ndarray(tuple(shape), dtype=pu.typemap(s.dtype()), 
                                  af_array=s)
        else:
            shape = list(a.shape)
            s = a.d_array
            # Do in decreasing axis order to avoid problems with the pop
            for ax in sorted(axis)[::-1]:
                # Optimization
                if(a.shape[ax] > 1):
                    s = func(a, s, pu.c2f(a.shape, ax))
                if keepdims:
                    shape[ax] = 1
                else:
                    shape.pop(ax)
            ret = afnumpy.ndarray(tuple(shape), dtype=pu.typemap(s.dtype()), 
                                  af_array=s)
        if(dtype is not None):
            ret = ret.astype(dtype)
        if(len(shape) == 0):
            ret = ret[()]
        return ret
    return wrapper

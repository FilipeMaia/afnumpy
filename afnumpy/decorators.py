import afnumpy

def outufunc(func):
    def wrapper(*args, **kws):
        out = kws.pop('out', None)
        ret = func(*args, **kws)
        if out is not None:
            out[:] = ret
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

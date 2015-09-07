import afnumpy
import arrayfire
import numpy
from IPython.core.debugger import Tracer
from .. import private_utils as pu

def tile(A, reps):
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    if(len(tup) > 4):
        raise NotImplementedError('Only up to 4 dimensions are supported')

    d = len(tup)
    shape = list(A.shape)
    n = max(A.size, 1)
    if (d < A.ndim):
        tup = (1,)*(A.ndim-d) + tup
    
    # Calculate final shape
    while len(shape) < len(tup):
        shape.insert(0,1)
    shape = tuple(numpy.array(shape) * numpy.array(tup))

    # Prepend ones to simplify calling
    if (d < 4):
        tup = (1,)*(4-d) + tup
    tup = pu.c2f(tup)
    s = arrayfire.tile(A.d_array, int(tup[0]), int(tup[1]), int(tup[2]), int(tup[3]))
    return afnumpy.ndarray(shape, dtype=pu.typemap(s.dtype()), af_array=s)



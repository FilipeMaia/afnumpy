import numpy.core.numeric as _nx
from numpy.core.numeric import obj2sctype
import afnumpy

def asfarray(a, dtype=_nx.float_):
    dtype = _nx.obj2sctype(dtype)
    if not issubclass(dtype, _nx.inexact):
        dtype = _nx.float_
    return afnumpy.asarray(a, dtype=dtype)

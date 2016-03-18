import arrayfire
import numpy
from .multiarray import ndarray, zeros, where, array, fromstring, arange, empty
from . import random
from .core import *
from .lib import *
from . import linalg
from .linalg import vdot, dot
import ctypes

def arrayfire_version(numeric = False):
    major = ctypes.c_int(0)
    minor = ctypes.c_int(0)
    patch = ctypes.c_int(0)
    arrayfire.backend.get().af_get_version(ctypes.pointer(major),
                                           ctypes.pointer(minor),
                                           ctypes.pointer(patch));
    if(numeric):
        return major.value * 1000000 + minor.value*1000 + patch.value
    return '%d.%d.%d' % (major.value, minor.value, patch.value)

def inplace_setitem(self, key, val):
    try:
        n_dims = self.numdims()
        if (arrayfire.util._is_number(val)):
            tdims = arrayfire.array._get_assign_dims(key, self.dims())
            other_arr = arrayfire.array.constant_array(val, tdims[0], tdims[1], tdims[2], tdims[3], self.type())
            del_other = True
        else:
            other_arr = val.arr
            del_other = False

        inds  = arrayfire.array._get_indices(key)

        # In place assignment. Notice passing a pointer to self.arr as output
        arrayfire.util.safe_call(arrayfire.backend.get().af_assign_gen(ctypes.pointer(self.arr),
                                                                            self.arr, ctypes.c_longlong(n_dims),
                                                                            inds.pointer,
                                                                            other_arr))
        if del_other:
            arrayfire.safe_call(arrayfire.backend.get().af_release_array(other_arr))
    except RuntimeError as e:
        raise IndexError(str(e))


def raw_ptr(self):
    """
    Return the device pointer held by the array.

    Returns
    ------
    ptr : int
          Contains location of the device pointer

    Note
    ----
    - This can be used to integrate with custom C code and / or PyCUDA or PyOpenCL.
    - No mem copy is peformed, this function returns the raw device pointer.
    """
    ptr = ctypes.c_void_p(0)
    arrayfire.backend.get().af_get_raw_ptr(ctypes.pointer(ptr), self.arr)
    return ptr.value

arrayfire.Array.__setitem__ = inplace_setitem

if arrayfire_version(numeric=True) >= 3003000:
    arrayfire.Array.device_ptr = raw_ptr
elif arrayfire_version(numeric=True) >= 3002000:
    raise RuntimeError('afnumpy is incompatible with arrayfire 3.2. Please upgrade.')


# This defines if we'll try to force JIT evals
# after every instructions.
# We we do not we risk having certain operations out of order
force_eval = True

# Check arrays for out of bounds indexing
# Also properly handle negative indices
safe_indexing = True

# The version number of afnumpy
__version__ = "1.0"

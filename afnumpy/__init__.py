import arrayfire_python
import numpy
import arrayfire
from multiarray import ndarray, zeros, where, array, fromstring, arange, empty
import random
from core import *
from lib import *
import linalg
from linalg import vdot, dot
import ctypes


def inplace_setitem(self, key, val):
    try:
        n_dims = self.numdims()

        if (arrayfire_python.util.is_number(val)):
            tdims = arrayfire_python.index.get_assign_dims(key, self.dims())
            other_arr = arrayfire_python.array.constant_array(val, tdims[0], tdims[1], tdims[2], tdims[3], self.type())
        else:
            other_arr = val.arr

        inds  = arrayfire_python.index.get_indices(key)

        # In place assignment. Notice passing a pointer to self.arr as output
        arrayfire_python.util.safe_call(arrayfire_python.backend.get().af_assign_gen(ctypes.pointer(self.arr),
                                                                            self.arr, ctypes.c_longlong(n_dims), 
                                                                            ctypes.pointer(inds),
                                                                            other_arr))
    except RuntimeError as e:
        raise IndexError(str(e))  

arrayfire_python.Array.__setitem__ = inplace_setitem


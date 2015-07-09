from afnumpy import ndarray
import afnumpy.private_utils as pu
import afnumpy.arrayfire as arrayfire

def concatenate(arrays, axis=0):
    if(len(arrays) < 1):
        raise ValueError('need at least one array to concatenate')
    if(axis > 3):
        raise NotImplementedError('only up to 4 axis as currently supported')
    arr = arrays[0].d_array.copy()
    axis = pu.c2f(arrays[0].shape, axis)
    for a in arrays[1:]:
        arr = arrayfire.join(axis, arr, a.d_array)
    return ndarray(pu.af_shape(arr), dtype=arrays[0].dtype, af_array=arr)

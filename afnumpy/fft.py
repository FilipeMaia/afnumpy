import numpy
from afnumpy.multiarray import ndarray
import arrayfire
import private_utils as pu

def fft2(a, s=None, axes=(-2, -1)):
    if s is not None:
        raise NotImplementedError
    ft_axes = list(axes)
    for i in range(0,2):
        if axes[i] < 0:
            ft_axes[i] = len(a.shape)+axes[i]
        ft_axes[i] = len(a.shape)-ft_axes[i]-1
    s = arrayfire.fft2(a.d_array, ft_axes[0], ft_axes[1])
    return ndarray(a.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)


    

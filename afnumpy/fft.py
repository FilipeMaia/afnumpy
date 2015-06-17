import numpy
from afnumpy.multiarray import ndarray
import arrayfire
import private_utils as pu

def fft(a, s=None, axes=None):
    if(s is None):
        s = a.shape[-1:]
    return __fftn__(a, s=s, axes=axes, direction='forward')

def ifft(a, s=None, axes=None):
    if(s is None):
        s = a.shape[-1:]
    return __fftn__(a, s=s, axes=axes, direction='inverse')

def fft2(a, s=None, axes=None):
    if(s is None):
        s = a.shape[-2:]
    return __fftn__(a, s=s, axes=axes, direction='forward')

def ifft2(a, s=None, axes=None):
    if(s is None):
        s = a.shape[-2:]
    return __fftn__(a, s=s, axes=axes, direction='inverse')

def fftn(a, s=None, axes=None):
    if(s is None):
        s = a.shape
    return __fftn__(a, s=s, axes=axes, direction='forward')

def ifftn(a, s=None, axes=None):
    if(s is None):
        s = a.shape
    return __fftn__(a, s=s, axes=axes, direction='inverse')

def __fftn__(a, s, axes, direction='forward'):
    if len(s) != 3 and len(s) != 2 and len(s) != 1:
        raise NotImplementedError
    if axes is not None:
        raise NotImplementedError
    if(direction == 'forward'):
        if len(s) == 3:
            fa = arrayfire.fft3(a.d_array, s[2], s[1], s[0])
        elif len(s) == 2:
            fa = arrayfire.fft2(a.d_array, s[1], s[0])
        elif len(s) == 1:
            fa = arrayfire.fft(a.d_array, s[0])
    elif direction == 'inverse':
        if len(s) == 3:
            fa = arrayfire.ifft3(a.d_array, s[2], s[1], s[0])
        elif len(s) == 2:
            fa = arrayfire.ifft2(a.d_array, s[1], s[0])
        elif len(s) == 1:
            fa = arrayfire.ifft(a.d_array, s[0])
    else:
        raise ValueError('Wrong FFT direction')
    return ndarray(a.shape, dtype=pu.InvTypeMap[fa.type()], af_array=fa)


def fftshift(a, axes=None):
    raise NotImplemented

    

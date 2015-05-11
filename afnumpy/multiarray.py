import numpy
import arrayfire
from IPython.core.debugger import Tracer


__TypeMap__ = { float: arrayfire.f64,
                numpy.float32: arrayfire.f32,
                numpy.float64: arrayfire.f64,
                numpy.dtype('float64'): arrayfire.f64,
                numpy.int8: arrayfire.b8,
                numpy.uint8: arrayfire.u8,
                numpy.bool: arrayfire.b8,
            }

__dummy__ = object()

def _raw(x):
    if(isinstance(x,ndarray)):
        return x.d_array
    else:
        return x

def zeros(shape, dtype=float, order='C'):
    b = numpy.zeros(shape, dtype, order)
    return ndarray(shape, dtype, buffer=b,order=order)

def ones(shape, dtype=float, order='C'):
    b = numpy.ones(shape, dtype, order)
    return ndarray(shape, dtype, buffer=b,order=order)

def where(condition, x=__dummy__, y=__dummy__):
    if(len(args
    raise

class ndarray(object):
    def __init__(self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, af_array=None):
        self.shape = shape
        self.dtype = dtype
        s_a = numpy.array(shape)
        if(s_a.size < 4):
            if(af_array is not None):
                self.handle = af_array.get()
                # We need to make sure to keep a copy of af_array
                # Otherwise python will free it and havoc ensues
                self.d_array = af_array
            else:
                if(buffer is not None):
                    ret, self.handle = arrayfire.af_create_array(buffer.ctypes.data, s_a.size, s_a.ctypes.data, __TypeMap__[dtype])
                else:
                    ret, self.handle = arrayfire.af_create_handle(s_a.size, s_a.ctypes.data, __TypeMap__[dtype])
                self.d_array = arrayfire.array(self.handle)
        else:
            raise NotImplementedError('Only up to 4 dimensions are supported')
        self.h_array = numpy.ndarray(shape,dtype,buffer,offset,strides,order)
        
    def __repr__(self):
        self.d_array.host(self.h_array.ctypes.data)
        return self.h_array.__repr__()        

    def __str__(self):
        self.d_array.host(self.h_array.ctypes.data)
        return self.h_array.__str__()        

    def __add__(self, other):
        s = self.d_array + _raw(other)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __iadd__(self, other):
        self.d_array += _raw(other)
        return self

    def __radd__(self, other):
        s = arrayfire.__add__(_raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __sub__(self, other):
        s = self.d_array - _raw(other)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __isub__(self, other):
        self.d_array -= _raw(other)
        return self

    def __rsub__(self, other):
        s = arrayfire.__sub__(_raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __mul__(self, other):
        s = self.d_array * _raw(other)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __imul__(self, other):
        self.d_array *= _raw(other)
        return self

    def __rmul__(self, other):
        s = arrayfire.__mul__(_raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __div__(self, other):
        s = self.d_array / _raw(other)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __idiv__(self, other):
        self.d_array /= _raw(other)
        return self

    def __rdiv__(self, other):
        s = arrayfire.__div__(_raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)
        
    def __pow__(self, other):
        s = arrayfire.pow(self.d_array, _raw(other))
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __lt__(self, other):
        s = self.d_array < _raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __le__(self, other):
        s = self.d_array <= _raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __gt__(self, other):
        s = self.d_array > _raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __ge__(self, other):
        s = self.d_array >= _raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __eq__(self, other):
        s = self.d_array == _raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __ne__(self, other):
        s = self.d_array != _raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __nonzero__(self):
        s = self.d_array != zeros(self.shape, self.dtype).d_array
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __len__(self):
        return len(self.shape)

    def __getitem__(self, key):
        

        


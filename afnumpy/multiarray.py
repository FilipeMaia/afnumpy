import numpy
import arrayfire
from IPython.core.debugger import Tracer

dim_t = numpy.int64

__TypeMap__ = { float: arrayfire.f64,
                numpy.float32: arrayfire.f32,
                numpy.dtype('float32'): arrayfire.f32,
                numpy.float64: arrayfire.f64,
                numpy.dtype('float64'): arrayfire.f64,
                numpy.int8: arrayfire.b8,
                numpy.dtype('int8'): arrayfire.b8,
                numpy.uint8: arrayfire.u8,
                numpy.dtype('uint8'): arrayfire.u8,
                numpy.bool: arrayfire.b8,
                numpy.dtype('bool'): arrayfire.b8,
                numpy.int64: arrayfire.s64,
                numpy.dtype('int64'): arrayfire.s64,
                numpy.uint64: arrayfire.u64,
                numpy.dtype('uint64'): arrayfire.u64,
                numpy.uint32: arrayfire.u32,
                numpy.dtype('uint32'): arrayfire.u32,
                numpy.int32: arrayfire.s32,
                numpy.dtype('int32'): arrayfire.s32,
            }

__dummy__ = object()

def _raw(x):
    if(isinstance(x,ndarray)):
        return x.d_array
    else:
        return x

def _af_shape(af_array):
    shape = ()
    for i in range(0,af_array.numdims()):
        shape = (af_array.dims(i),)+shape
    return shape

def zeros(shape, dtype=float, order='C'):
    b = numpy.zeros(shape, dtype, order)
    return ndarray(b.shape, b.dtype, buffer=b,order=order)

def ones(shape, dtype=float, order='C'):
    b = numpy.ones(shape, dtype, order)
    return ndarray(b.shape, b.dtype, buffer=b,order=order)

def array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0):
    if(order is not None):
        raise NotImplementedError
    if(subok is not False):
        raise NotImplementedError
    # If it's not a numpy or afnumpy array first create a numpy array from it
    if(not isinstance(object, ndarray) and
       not isinstance(object, numpy.ndarray)):
        object = numpy.array(object, dtype=dtype, copy=copy, subok=subok, ndmin=ndmin)
#        return ndarray(a.shape, dtype=a.dtype, buffer=a)       

    shape = object.shape
    while(ndmin > len(shape)):
        shape = (1,)+shape
    if(dtype is None):
        dtype = object.dtype
    if(isinstance(object, ndarray)):
        if(copy):
            s = object.d_array.copy().astype(__TypeMap__[dtype])
        else:
            s = object.d_array.astype(__TypeMap__[dtype])
        return ndarray(shape, dtype=dtype, af_array=s)
    elif(isinstance(object, numpy.ndarray)):
        return ndarray(shape, dtype=dtype, buffer=object.astype(dtype, copy=False))
    else:
        raise AssertionError
        

def where(condition, x=__dummy__, y=__dummy__):
    a = condition
    s = arrayfire.where(a.d_array)
    if(x is __dummy__ and y is __dummy__):
        return ndarray(_af_shape(s), dtype=numpy.uint32, af_array=s)
    elif(x is not __dummy__ and y is not __dummy__):
        if(x.dtype != y.dtype):
            raise TypeError('x and y must have same dtype')
        if(x.shape != y.shape):
            raise ValueError('x and y must have same shape')
        idx = ndarray(_af_shape(s), dtype=numpy.uint32, af_array=s)
        ret = array(y)
        ret[idx] = x[idx]
        return ret;
    else:
        raise ValueError('either both or neither of x and y should be given')

def all(a, axis=None, out=None, keepdims=False):
    if(out is not None):
        raise NotImplementedError
    if(keepdims is not False):
        raise NotImplementedError
    if(axis is None):
        for i in range(len(a.shape)-1,-1,-1):
            s = arrayfire.alltrue(a.d_array, c2f(a.shape, i)) 
            a = ndarray(_af_shape(s), dtype=bool, af_array=s)
    else:
        s = arrayfire.alltrue(a.d_array, axis)
    a = ndarray(_af_shape(s), dtype=bool, af_array=s)
    if(axis == -1):
        if(keepdims):
            return numpy.array(a)
        else:
            return numpy.array(a)[0]
    else:
        return a

def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if(out is not None):
        raise NotImplementedError
    if(keepdims is not False):
        raise NotImplementedError
    if(axis is None):
        for i in range(len(a.shape)-1,-1,-1):
            s = arrayfire.sum(a.d_array, c2f(a.shape, i)) 
            a = ndarray(_af_shape(s), dtype=a.dtype, af_array=s)
    else:
        s = arrayfire.sum(a.d_array, c2f(a.shape, axis))
    a = ndarray(_af_shape(s), dtype=a.dtype, af_array=s)
    if(axis is None):
        if(keepdims):
            return numpy.array(a)
        else:
            return numpy.array(a)[0]
    else:
        return a


def reshape(a, newshape, order='C'):
    if(order is not 'C'):
        raise NotImplementedError
    newshape = numpy.array(c2f(newshape), dtype=dim_t)
    ret, handle = arrayfire.af_moddims(a.d_array.get(), newshape.size, newshape.ctypes.data)
    s = arrayfire.array_from_handle(handle)
    a = ndarray(_af_shape(s), dtype=a.dtype, af_array=s)
    return a

def c2f(shape, dim = None):
    if(dim is None):
        return shape[::-1]
    else:
        return len(shape)-dim-1

class ndarray(object):
    def __init__(self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, af_array=None):
        if(offset != 0):
            raise NotImplementedError('offset must be 0')
        if(strides is not None):
            raise NotImplementedError('strides must be None')
        if(order is not None and order != 'C'):
            raise NotImplementedError('order must be None')
        self.shape = shape
        self.dtype = dtype
        s_a = numpy.array(c2f(shape),dtype=dim_t)
        if(s_a.size < 1):
            raise NotImplementedError('0 dimension arrays are not yet supported')
        elif(s_a.size < 4):
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
                self.d_array = arrayfire.array_from_handle(self.handle)
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
        s = arrayfire.__add__(self.d_array, _raw(other))
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __iadd__(self, other):
        self.d_array += _raw(other)
        return self

    def __radd__(self, other):
        s = arrayfire.__add__(_raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __sub__(self, other):
        s = arrayfire.__sub__(self.d_array, _raw(other))
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __isub__(self, other):
        self.d_array -= _raw(other)
        return self

    def __rsub__(self, other):
        s = arrayfire.__sub__(_raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __mul__(self, other):
        s = arrayfire.__mul__(self.d_array, _raw(other))
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __imul__(self, other):
        self.d_array *= _raw(other)
        return self

    def __rmul__(self, other):
        s = arrayfire.__mul__(_raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __div__(self, other):
        s = arrayfire.__div__(self.d_array, _raw(other))
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

    def __rpow__(self, other):
        s = arrayfire.pow(_raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __lt__(self, other):
        s = arrayfire.__lt__(self.d_array, _raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __le__(self, other):
        s = arrayfire.__le__(self.d_array, _raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __gt__(self, other):
        s = arrayfire.__gt__(self.d_array, _raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __ge__(self, other):
        s = arrayfire.__ge__(self.d_array, _raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __eq__(self, other):
        s = arrayfire.__eq__(self.d_array, _raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __ne__(self, other):
        s = arrayfire.__ne__(self.d_array, _raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __nonzero__(self):
        return numpy.array(self).__nonzero__()

    def __len__(self):
        return len(h_array)

    @property
    def size(self):
        return len(h_array)

    def _convert_dim(self, args):
        if args < 0:
            args = len(self) + args
        return args

    def __getitem__(self, args):
        if(isinstance(args, ndarray)):
            s = (self.d_array[arrayfire.index(args.d_array)])
            return ndarray(_af_shape(s), dtype=self.dtype, af_array=s)
        elif(isinstance(args, tuple)):
            args = list(args)
            args[0] = self._convert_dim(args[0])
            return self._data[tuple(args)]
        else:
            idx = arrayfire.index(self._convert_dim(args))
            s = self.d_array.__getitem__(arrayfire.index(self._convert_dim(args)))
            return ndarray(_af_shape(s), dtype=self.dtype, af_array=s)


    def __setitem__(self, idx, value):
        if(isinstance(idx, ndarray)):
            idx = arrayfire.index(idx.d_array)
        else:
            raise NotImplementedError('indices must be a afnumpy.ndarray')
        sel = self.d_array[idx]
        if(isinstance(value, ndarray)):
            if(value.dtype != self.dtype):
                raise TypeError('left hand side must have same dtype as right hand side')
            sel.copy_on_write(value.d_array)
        else:
            raise NotImplementedError('values must be a afnumpy.ndarray')

    def __array__(self):
        self.d_array.host(self.h_array.ctypes.data)
        return numpy.copy(self.h_array)

    def reshape(self, shape, order = 'C'):
        return reshape(self, shape, order)
#    def __getattr__(self,name):
#        print name
#        raise AttributeError
        

    

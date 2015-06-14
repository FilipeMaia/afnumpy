import numpy
import arrayfire
import numbers
from IPython.core.debugger import Tracer
import private_utils as pu


def vdot(a, b):
    s = arrayfire.dot(arrayfire.conjg(a.d_array), b.d_array)
    return ndarray(pu.af_shape(s), dtype=a.dtype, af_array=s)

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
            s = object.d_array.copy().astype(pu.TypeMap[dtype])
        else:
            s = object.d_array.astype(pu.TypeMap[dtype])
        return ndarray(shape, dtype=dtype, af_array=s)
    elif(isinstance(object, numpy.ndarray)):
        return ndarray(shape, dtype=dtype, buffer=object.astype(dtype, copy=False))
    else:
        raise AssertionError
        

def where(condition, x=pu.dummy, y=pu.dummy):
    a = condition
    s = arrayfire.where(a.d_array)
    if(x is pu.dummy and y is pu.dummy):
        return ndarray(pu.af_shape(s), dtype=numpy.uint32, af_array=s)
    elif(x is not pu.dummy and y is not pu.dummy):
        if(x.dtype != y.dtype):
            raise TypeError('x and y must have same dtype')
        if(x.shape != y.shape):
            raise ValueError('x and y must have same shape')
        idx = ndarray(pu.af_shape(s), dtype=numpy.uint32, af_array=s)
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
            s = arrayfire.allTrue(a.d_array, pu.c2f(a.shape, i)) 
            a = ndarray(pu.af_shape(s), dtype=bool, af_array=s)
    else:
        s = arrayfire.allTrue(a.d_array, pu.c2f(a.shape, axis))
    a = ndarray(pu.af_shape(s), dtype=bool, af_array=s)
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
            s = arrayfire.sum(a.d_array, pu.c2f(a.shape, i)) 
            a = ndarray(pu.af_shape(s), dtype=a.dtype, af_array=s)
    else:
        s = arrayfire.sum(a.d_array, pu.c2f(a.shape, axis))
    a = ndarray(pu.af_shape(s), dtype=a.dtype, af_array=s)
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
    newshape = numpy.array(pu.c2f(newshape), dtype=pu.dim_t)
    ret, handle = arrayfire.af_moddims(a.d_array.get(), newshape.size, newshape.ctypes.data)
    s = arrayfire.array_from_handle(handle)
    a = ndarray(pu.af_shape(s), dtype=a.dtype, af_array=s)
    return a

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
        s_a = numpy.array(pu.c2f(shape),dtype=pu.dim_t)
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
                    ret, self.handle = arrayfire.af_create_array(buffer.ctypes.data, s_a.size, s_a.ctypes.data, pu.TypeMap[dtype])
                else:
                    ret, self.handle = arrayfire.af_create_handle(s_a.size, s_a.ctypes.data, pu.TypeMap[dtype])
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
        s = arrayfire.__add__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __iadd__(self, other):
        self.d_array += pu.raw(other)
        return self

    def __radd__(self, other):
        s = arrayfire.__add__(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __sub__(self, other):
        s = arrayfire.__sub__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __isub__(self, other):
        self.d_array -= pu.raw(other)
        return self

    def __rsub__(self, other):
        s = arrayfire.__sub__(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __mul__(self, other):
        s = arrayfire.__mul__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __imul__(self, other):
        self.d_array *= pu.raw(other)
        return self

    def __rmul__(self, other):
        s = arrayfire.__mul__(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __div__(self, other):
        s = arrayfire.__div__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __idiv__(self, other):
        self.d_array /= pu.raw(other)
        return self

    def __rdiv__(self, other):
        s = arrayfire.__div__(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)
        
    def __pow__(self, other):
        s = arrayfire.pow(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __rpow__(self, other):
        s = arrayfire.pow(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=self.dtype, af_array=s)

    def __lt__(self, other):
        s = arrayfire.__lt__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __le__(self, other):
        s = arrayfire.__le__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __gt__(self, other):
        s = arrayfire.__gt__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __ge__(self, other):
        s = arrayfire.__ge__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __eq__(self, other):
        s = arrayfire.__eq__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __ne__(self, other):
        s = arrayfire.__ne__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __abs__(self):
        s = arrayfire.abs(self.d_array)
        # dtype is wrong for complex types
        return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)

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
            return ndarray(pu.af_shape(s), dtype=self.dtype, af_array=s)
        elif(isinstance(args, tuple)):
            args = list(args)
            args[0] = self._convert_dim(args[0])
            return self._data[tuple(args)]
        else:
            idx = arrayfire.index(self._convert_dim(args))
            s = self.d_array.__getitem__(arrayfire.index(self._convert_dim(args)))
            return ndarray(pu.af_shape(s), dtype=self.dtype, af_array=s)

    def __convert_dim__(self, idx, maxlen):
        if(isinstance(idx, slice)):
            if idx.step is None:
                step = 1
            else:
                step = idx.step
            if idx.start is None:
                if step < 0:
                    start = maxlen-1
                else:
                    start = 0
            else:
                start = idx.start
            if idx.stop is None:
                if step < 0:
                    end = 0
                else:
                    end = maxlen-1
            else:
                end = idx.stop
            return slice(start,end,step)
        else:
            if idx < 0:
                return maxlen+idx
            else:
                return idx            

    def __setitem__(self, idx, value):
        if(isinstance(idx, ndarray)):
            idx = arrayfire.index(idx.d_array)
        elif(isinstance(idx, slice)):
            idx = self.__convert_dim__(idx,self.shape[0])
            idx = arrayfire.index(arrayfire.seq(float(idx.start),float(idx.stop),float(idx.step)))
        else:
            raise NotImplementedError('indices must be a afnumpy.ndarray')
        if(isinstance(value, ndarray)):
            if(value.dtype != self.dtype):
                raise TypeError('left hand side must have same dtype as right hand side')
            self.d_array.setValue(idx, value.d_array)
        elif(isinstance(value, numbers.Number)):
            self.d_array.setValue(idx, value)
        else:
            raise NotImplementedError('values must be a afnumpy.ndarray')

    def __array__(self):
        self.d_array.host(self.h_array.ctypes.data)
        return numpy.copy(self.h_array)

    def reshape(self, shape, order = 'C'):
        return reshape(self, shape, order)

    def max(self):
        type_max = getattr(arrayfire, 'max_'+pu.TypeToString[self.d_array.type()])
        return type_max(self.d_array)

    def min(self):
        type_max = getattr(arrayfire, 'min_'+pu.TypeToString[self.d_array.type()])
        return type_max(self.d_array)

#    def __getattr__(self,name):
#        print name
#        raise AttributeError
        

    

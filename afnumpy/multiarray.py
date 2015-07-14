import numpy
#import arrayfire
import numbers
from IPython.core.debugger import Tracer
import private_utils as pu
import afnumpy
import indexing

def iufunc(func):
    def wrapper(*args, **kws):
        if all(isinstance(A, ndarray) for A in args):
            bcast_args = afnumpy.broadcast_arrays(*args)
            if(bcast_args[0].shape is not args[0].shape):
                raise ValueError("non-broadcastable output operand with"
                                 " shape %s doesn't match the broadcast"
                                 " shape %s" % (args[0].shape, bcast_args[0].shape))
            
        return func(*bcast_args, **kws)
    return wrapper

def ufunc(func):
    def wrapper(*args, **kws):
        if all(isinstance(A, ndarray) for A in args):
            args = afnumpy.broadcast_arrays(*args)            
        return func(*args, **kws)
    return wrapper

def fromstring(string, dtype=float, count=-1, sep=''):
    return array(numpy.fromstring(string, dtype, count, sep))

def vdot(a, b):
    s = afnumpy.arrayfire.dot(afnumpy.arrayfire.conjg(a.d_array), b.d_array)
    return ndarray((), dtype=a.dtype, af_array=s)

def zeros(shape, dtype=float, order='C'):
    b = numpy.zeros(shape, dtype, order)
    return ndarray(b.shape, b.dtype, buffer=b,order=order)

def array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0):
    if(order is not None and order is not 'K'):
        raise NotImplementedError
    # We're going to ignore this for now
    # if(subok is not False):
    #     raise NotImplementedError

    # If it's not a numpy or afnumpy array first create a numpy array from it
    if(not isinstance(object, ndarray) and
       not isinstance(object, numpy.ndarray)):
        object = numpy.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)
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
        
def arange(start, stop = None, step = None, dtype=None):
    return afnumpy.array(numpy.arange(start,stop,step,dtype))        
 
def where(condition, x=pu.dummy, y=pu.dummy):
    a = condition
    s = afnumpy.arrayfire.where(a.d_array)
    # numpy uses int64 while arrayfire uses uint32
    s = ndarray(pu.af_shape(s), dtype=numpy.uint32, af_array=s).astype(numpy.int64)
    if(x is pu.dummy and y is pu.dummy):
        idx = []
        mult = 1
        for i in a.shape[::-1]:
            mult *= i
            idx = [s % mult] + idx 
            s /= mult
        idx = tuple(idx)
        return idx
    elif(x is not pu.dummy and y is not pu.dummy):
        if(x.dtype != y.dtype):
            raise TypeError('x and y must have same dtype')
        if(x.shape != y.shape):
            raise ValueError('x and y must have same shape')
        ret = array(y)
        idx = afnumpy.arrayfire.index(s.d_array)
        if(len(ret.shape) > 1):
            ret = ret.flatten()
            ret[s] = x.flatten()[s]
            ret = ret.reshape(x.shape)
        else:
            ret[s] = x[s]
        return ret;
    else:
        raise ValueError('either both or neither of x and y should be given')


class ndarray(object):
    def __init__(self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, af_array=None):
        if(offset != 0):
            raise NotImplementedError('offset must be 0')
        if(strides is not None):
            raise NotImplementedError('strides must be None')
        if(order is not None and order != 'C'):
            raise NotImplementedError('order must be None')
        if isinstance(shape, numbers.Number):
            self._shape = (shape,)
        else:
            self._shape = tuple(shape)
        self.dtype = dtype
        s_a = numpy.array(pu.c2f(shape),dtype=pu.dim_t)
        if(s_a.size < 1):
            self.d_array = None
            if buffer is None:
                if af_array is None:
                    raise ValueError
                buffer = numpy.array(1, dtype=dtype)
                af_array.host(buffer.ctypes.data)                
        elif(s_a.size <= 4):
            if(af_array is not None):
                # We need to make sure to keep a copy of af_array
                # Otherwise python will free it and havoc ensues
                self.d_array = af_array
            else:
                if(buffer is not None):
                    ret, self.handle = afnumpy.arrayfire.af_create_array(buffer.ctypes.data, s_a.size, s_a.ctypes.data, pu.TypeMap[dtype])
                else:
                    ret, self.handle = afnumpy.arrayfire.af_create_handle(s_a.size, s_a.ctypes.data, pu.TypeMap[dtype])
                self.d_array = afnumpy.arrayfire.array_from_handle(self.handle)
        else:
            raise NotImplementedError('Only up to 4 dimensions are supported')
        self.h_array = numpy.ndarray(shape,dtype,buffer,offset,strides,order)
        
    def __repr__(self):
        if(self.d_array is not None):
            self.d_array.host(self.h_array.ctypes.data)
        return self.h_array.__repr__()        

    def __str__(self):
        if(self.d_array is not None):
            self.d_array.host(self.h_array.ctypes.data)
        return self.h_array.__str__()        

    @ufunc
    def __add__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__add__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(self.h_array + other, dtype=self.dtype)

    @iufunc
    def __iadd__(self, other):
        if(self.d_array):
            self[:] = self[:] + pu.raw(other)
        else:
            self.h_array += other
        return self

    def __radd__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__add__(pu.raw(other), self.d_array)
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(other + self.h_array, dtype=self.dtype)

    @ufunc
    def __sub__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__sub__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(self.h_array - other, dtype=self.dtype)

    @iufunc
    def __isub__(self, other):
        if(self.d_array):
            self[:] = self[:] - pu.raw(other)
        else:
            self.h_array -= other
        return self

    def __rsub__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__sub__(pu.raw(other), self.d_array)
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(other - self.h_array, dtype=self.dtype)

    @ufunc
    def __mul__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__mul__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(self.h_array * other, dtype=self.dtype)

    @iufunc
    def __imul__(self, other):
        if(self.d_array):
            self[:] = self[:] * pu.raw(other)
        else:
            self.h_array *= other
        return self

    def __rmul__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__mul__(pu.raw(other), self.d_array)
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(other * self.h_array, dtype=self.dtype)

    @ufunc
    def __div__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__div__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(self.h_array / other, dtype=self.dtype)

    @iufunc
    def __idiv__(self, other):
        if(self.d_array):
            self[:] = self[:] / pu.raw(other)
        else:
            self.h_array /= other
        return self

    def __rdiv__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__div__(pu.raw(other), self.d_array)
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(other / self.h_array, dtype=self.dtype)
        
    def __pow__(self, other):
        if(self.d_array):
            if(isinstance(other, numbers.Number) and numpy.issubdtype(type(other), numpy.float) and
               numpy.issubdtype(self.dtype, numpy.integer)):
                # AF does not automatically upconvert A**0.5 to float for integer arrays
                s = afnumpy.arrayfire.pow(self.astype(type(other)).d_array, pu.raw(other))
            else:
                s = afnumpy.arrayfire.pow(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(self.h_array ** other, dtype=self.dtype)

    def __rpow__(self, other):
        if(self.d_array):
            if(isinstance(other, numbers.Number) and numpy.issubdtype(type(other), numpy.float) and
               numpy.issubdtype(self.dtype, numpy.integer)):
                # AF does not automatically upconvert A**0.5 to float for integer arrays
                s = afnumpy.arrayfire.pow(pu.raw(other), self.astype(type(other)).d_array)
            else:
                s = afnumpy.arrayfire.pow(pu.raw(other), self.d_array)
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(other ** self.h_array, dtype=self.dtype)

    def __lt__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__lt__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=numpy.bool, af_array=s)
        else:
            return array(self.h_array < other, dtype=self.dtype)

    def __le__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__le__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=numpy.bool, af_array=s)
        else:
            return array(self.h_array <= other, dtype=self.dtype)

    def __gt__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__gt__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=numpy.bool, af_array=s)
        else:
            return array(self.h_array > other, dtype=self.dtype)

    def __ge__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__ge__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=numpy.bool, af_array=s)
        else:
            return array(self.h_array >= other, dtype=self.dtype)

    def __eq__(self, other):
        if(other is None):
            return False
        if(self.d_array):
            s = afnumpy.arrayfire.__eq__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=numpy.bool, af_array=s)
        else:
            return array(self.h_array == other, dtype=self.dtype)

    def __ne__(self, other):
        if(other is None):
            return True
        if(self.d_array):
            s = afnumpy.arrayfire.__ne__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=numpy.bool, af_array=s)
        else:
            return array(self.h_array != other, dtype=self.dtype)

    def __abs__(self):
        if(self.d_array):
            s = afnumpy.arrayfire.abs(self.d_array)
            # dtype is wrong for complex types
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(abs(self.h_array), dtype=self.dtype)

    def __neg__(self):
        return 0 - self;

    def __pos__(self):
        return self;

    def __invert__(self):
        raise NotImplementedError

    def __nonzero__(self):
        return numpy.array(self).__nonzero__()

    def __len__(self):
        return self.shape[0]

    def __mod__(self, other):
        a = self / other
        if numpy.issubdtype(a.dtype, numpy.float):
            out = self - afnumpy.floor(self / other) * other
        else:
            out = self - a * other
        out.d_array.eval()
        return out

    def __rmod__(self, other):
        return other - afnumpy.floor(other / self) * self

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return numpy.prod(self.shape)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self.__reshape__(value)

    @property
    def strides(self):
        return self.h_array.strides

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        if(self.d_array is None):
            raise IndexError('too many indices for array')
        idx, new_shape = indexing.__convert_dim__(self.shape, args)
        if None in idx:
            # one of the indices is empty
            return ndarray(indexing.__index_shape__(self.shape, idx), dtype=self.dtype)

        if(isinstance(idx,list)):
            # There must be a better way to do this!
            if(len(idx) == 1):
                s = self.d_array.__getitem__(idx[0])
            if(len(idx) == 2):
                s = self.d_array.__getitem__(idx[0],idx[1])
            if(len(idx) == 3):
                s = self.d_array.__getitem__(idx[0],idx[1],idx[2])
            if(len(idx) == 4):
                s = self.d_array.__getitem__(idx[0],idx[1],idx[2],idx[3])
        else:
            s = self.d_array.__getitem__(idx)
        shape = pu.af_shape(s)
        array = ndarray(shape, dtype=self.dtype, af_array=s)
        if(shape != new_shape):
            array = array.reshape(new_shape)
        return array

    def __setitem__(self, idx, value):
        if(self.d_array is None):
            raise IndexError('too many indices for array')
        idx, idx_shape = indexing.__convert_dim__(self.shape, idx)
        if None in idx:
            # one of the indices is empty
            return
            
        if(isinstance(value, ndarray)):
            if(value.dtype != self.dtype):
                raise TypeError('left hand side must have same dtype as right hand side')
            if(isinstance(idx,list)):
                # There must be a better way to do this!
#                if(idx_shape != value.shape):
#                    value = value.reshape(idx_shape)
                value = indexing.__expand_dim__(self.shape, value, idx)
                if(len(idx) == 1):
                    self.d_array.setValue(idx[0], value.d_array)
                if(len(idx) == 2):
                    self.d_array.setValue(idx[0], idx[1], value.d_array)
                if(len(idx) == 3):
                    self.d_array.setValue(idx[0], idx[1], idx[2], value.d_array)
                if(len(idx) == 4):
                    self.d_array.setValue(idx[0], idx[1], idx[2], idx[3], value.d_array)
            else:
                self.d_array.setValue(idx, value.d_array)
        elif(isinstance(value, numbers.Number)):
            self.d_array.setValue(idx[0], value)
        else:
            raise NotImplementedError('values must be a afnumpy.ndarray')

    def __array__(self):
        if(self.d_array is not None):
            self.d_array.host(self.h_array.ctypes.data)
        return numpy.copy(self.h_array)

    def transpose(self, *axes):
        if(self.d_array):
            s = afnumpy.arrayfire.transpose(self.d_array)
            return ndarray(pu.af_shape(s), dtype=self.dtype, af_array=s)
        else:
            return array(self.h_array.transpose(axes), dtype=self.dtype)

    def reshape(self, shape, order = 'C'):
        ret =  afnumpy.copy(self)
        ret.__reshape__(shape, order)
        return ret
        
    # In place reshape    
    def __reshape__(self, newshape, order = 'C'):
        if(order is not 'C'):
            raise NotImplementedError
        if isinstance(newshape,numbers.Number):
            newshape = (newshape,)
        # Replace a possible -1 with the 
        if -1 in newshape:
            newshape = list(newshape)
            i = newshape.index(-1)
            newshape[i] = 1
            if -1 in newshape:
                raise ValueError('Only one -1 allowed in shape')
            newshape[i] = self.size/numpy.prod(newshape)
        if self.size != numpy.prod(newshape):
            raise ValueError('total size of new array must be unchanged')
        if len(newshape) == 0:
            # Deal with empty shapes
            self.d_array.host(self.h_array.ctypes.data)
            self._shape = tuple()
            self.d_array = None
            return

        af_shape = numpy.array(pu.c2f(newshape), dtype=pu.dim_t)
        ret, handle = afnumpy.arrayfire.af_moddims(self.d_array.get(), af_shape.size, af_shape.ctypes.data)
        s = afnumpy.arrayfire.array_from_handle(handle)
        self.d_array = s
        self.h_array.shape = newshape
        self._shape = tuple(newshape)
        
    def flatten(self):
        return afnumpy.reshape(self, self.size)

    def max(self):
        if(self.d_array):
            type_max = getattr(afnumpy.arrayfire, 'max_'+pu.TypeToString[self.d_array.type()])
            return type_max(self.d_array)
        else:
            return self.h_array.max()

    def min(self):
        if(self.d_array):
            type_min = getattr(afnumpy.arrayfire, 'min_'+pu.TypeToString[self.d_array.type()])
            return type_min(self.d_array)
        else:
            return self.h_array.min()

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        if(self.d_array):
            if(order != 'K'):
                raise NotImplementedError('only order=K implemented')
            if(casting != 'unsafe'):
                raise NotImplementedError('only casting=unsafe implemented')
            if(copy == False and order == 'K' and dtype == self.dtype):
                return self
            s = self.d_array.astype(pu.TypeMap[dtype])
            return ndarray(self.shape, dtype=dtype, af_array=s)
        else:
            return array(self.h_array.astype(dtype, order, casting, subok, copy), dtype=dtype)


    def round(self, decimals=0, out=None):
        if decimals != 0:
            raise NotImplementedError('only supports decimals=0')
        s = afnumpy.arrayfire.round(self.d_array)
        ret = ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        if(out):
            out[:] = ret[:]
        return ret

        

    


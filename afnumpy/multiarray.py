import numpy
import numbers
from IPython.core.debugger import Tracer
import private_utils as pu
import afnumpy
import indexing
from decorators import *
import collections

def fromstring(string, dtype=float, count=-1, sep=''):
    return array(numpy.fromstring(string, dtype, count, sep))

def empty(shape, dtype=float, order='C'):
    return ndarray(shape, dtype=dtype, order=order)

def zeros(shape, dtype=float, order='C'):
    b = numpy.zeros(shape, dtype, order)
    return ndarray(b.shape, b.dtype, buffer=b,order=order)

def array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0):
    # We're going to ignore this for now
    # if(subok is not False):
    #     raise NotImplementedError
    if(order is not None and order is not 'K' and order is not 'C'):
        raise NotImplementedError

    # If it's not a numpy or afnumpy array first create a numpy array from it
    if(not isinstance(object, ndarray) and
       not isinstance(object, numpy.ndarray)):
        object = numpy.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    shape = object.shape
    while(ndmin > len(shape)):
        shape = (1,)+shape
    if(dtype is None):
        dtype = object.dtype
    if(isinstance(object, ndarray)):
        if(copy):
            s = object.d_array.copy().astype(pu.typemap(dtype))
        else:
            s = object.d_array.astype(pu.typemap(dtype))
        return ndarray(shape, dtype=dtype, af_array=s)
    elif(isinstance(object, numpy.ndarray)):
        return ndarray(shape, dtype=dtype, buffer=object.astype(dtype, copy=copy))
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
        self._base = None
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
            # We'll use af_arrays of size (1) to keep scalars
            s_a = numpy.array((1),dtype=pu.dim_t)
        if(s_a.size <= 4):
            if(af_array is not None):
                # We need to make sure to keep a copy of af_array
                # Otherwise python will free it and havoc ensues
                self.d_array = af_array
            else:
                if(buffer is not None):
                    ret, self.handle = afnumpy.arrayfire.af_create_array(buffer.ctypes.data, s_a.size, s_a.ctypes.data, pu.typemap(dtype))
                else:
                    ret, self.handle = afnumpy.arrayfire.af_create_handle(s_a.size, s_a.ctypes.data, pu.typemap(dtype))
                self.d_array = afnumpy.arrayfire.array_from_handle(self.handle)
        else:
            raise NotImplementedError('Only up to 4 dimensions are supported')
        self.h_array = numpy.ndarray(shape,dtype,buffer,offset,strides,order)
        
    def __repr__(self):
        self.d_array.host(self.h_array.ctypes.data)
        return self.h_array.__repr__()        

    def __str__(self):
        self.d_array.host(self.h_array.ctypes.data)
        return self.h_array.__str__()        

    @ufunc
    def __add__(self, other):
        s = afnumpy.arrayfire.__add__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)

    @iufunc
    def __iadd__(self, other):
        self[:] = self[:] + pu.raw(other)
        return self

    def __radd__(self, other):
        s = afnumpy.arrayfire.__add__(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)

    @ufunc
    def __sub__(self, other):
        s = afnumpy.arrayfire.__sub__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)

    @iufunc
    def __isub__(self, other):
        afnumpy.subtract(self, pu.raw(other), out=self)
        return self

    def __rsub__(self, other):
        s = afnumpy.arrayfire.__sub__(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)

    @ufunc
    def __mul__(self, other):
        s = afnumpy.arrayfire.__mul__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)

    @iufunc
    def __imul__(self, other):
        afnumpy.multiply(self, pu.raw(other), out=self)
        return self

    def __rmul__(self, other):
        s = afnumpy.arrayfire.__mul__(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)

    @ufunc
    def __div__(self, other):
        s = afnumpy.arrayfire.__div__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)

    @iufunc
    def __idiv__(self, other):
        afnumpy.divide(self, pu.raw(other), out=self)
        return self

    def __rdiv__(self, other):
        s = afnumpy.arrayfire.__div__(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)
        
    def __pow__(self, other):
        if(isinstance(other, numbers.Number) and numpy.issubdtype(type(other), numpy.float) and
           numpy.issubdtype(self.dtype, numpy.integer)):
            # AF does not automatically upconvert A**0.5 to float for integer arrays
            s = afnumpy.arrayfire.pow(self.astype(type(other)).d_array, pu.raw(other))
        else:
            s = afnumpy.arrayfire.pow(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)

    def __rpow__(self, other):
        if(isinstance(other, numbers.Number) and numpy.issubdtype(type(other), numpy.float) and
           numpy.issubdtype(self.dtype, numpy.integer)):
            # AF does not automatically upconvert A**0.5 to float for integer arrays
            s = afnumpy.arrayfire.pow(pu.raw(other), self.astype(type(other)).d_array)
        else:
            s = afnumpy.arrayfire.pow(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)

    def __lt__(self, other):
        s = afnumpy.arrayfire.__lt__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __le__(self, other):
        s = afnumpy.arrayfire.__le__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __gt__(self, other):
        s = afnumpy.arrayfire.__gt__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __ge__(self, other):
        s = afnumpy.arrayfire.__ge__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __eq__(self, other):
        if(other is None):
            return False
        s = afnumpy.arrayfire.__eq__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __ne__(self, other):
        if(other is None):
            return True
        s = afnumpy.arrayfire.__ne__(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __abs__(self):
        s = afnumpy.arrayfire.abs(self.d_array)
        # dtype is wrong for complex types
        return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)

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

    @property
    def flat(self):        
        ret = ndarray(self.size, dtype=self.dtype, af_array=afnumpy.arrayfire.flat(self.d_array))
        ret._base = self
        return ret

    @property
    def real(self):
        ret_type = numpy.real(numpy.zeros((),dtype=self.dtype)).dtype
        shape = list(self.shape)
        shape[0] *= 2
        dims = numpy.array(pu.c2f(shape),dtype=pu.dim_t)
        ret, handle = afnumpy.arrayfire.af_device_array(self.d_array.device_f32(),
                                                        self.ndim,
                                                        dims.ctypes.data,
                                                        pu.typemap(ret_type))
        afnumpy.arrayfire.af_retain_array(handle)
        s = afnumpy.arrayfire.array_from_handle(handle)
        a = ndarray(shape, dtype=ret_type, af_array=s)
        a._base = self
        return a[::2,]

    @property
    def imag(self):
        ret_type = numpy.real(numpy.zeros((),dtype=self.dtype)).dtype
        shape = list(self.shape)
        shape[0] *= 2
        dims = numpy.array(pu.c2f(shape),dtype=pu.dim_t)
        ret, handle = afnumpy.arrayfire.af_device_array(self.d_array.device_f32(),
                                                        self.ndim,
                                                        dims.ctypes.data,
                                                        pu.typemap(ret_type))
        afnumpy.arrayfire.af_retain_array(handle)
        s = afnumpy.arrayfire.array_from_handle(handle)
        a = ndarray(shape, dtype=ret_type, af_array=s)
        a._base = self
        return a[1::2,]

    def ravel(self, order=None):
        if(order != None and order != 'K' and order != 'C'):
            raise NotImplementedError('order %s not supported' % (order))
        return self.flat

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        idx, new_shape = indexing.__convert_dim__(self.shape, args)
        if None in idx:
            # one of the indices is empty
            return ndarray(indexing.__index_shape__(self.shape, idx), dtype=self.dtype)
        if(isinstance(idx,list)):
            # There must be a better way to do this!
            if(len(idx) == 0):
                s = self.d_array.__getitem__(afnumpy.arrayfire.index(0))
            elif(len(idx) == 1):
                s = self.d_array.__getitem__(idx[0])
            elif(len(idx) == 2):
                s = self.d_array.__getitem__(idx[0],idx[1])
            elif(len(idx) == 3):
                s = self.d_array.__getitem__(idx[0],idx[1],idx[2])
            elif(len(idx) == 4):
                s = self.d_array.__getitem__(idx[0],idx[1],idx[2],idx[3])
        else:
            raise ValueError
            s = self.d_array.__getitem__(idx)
        shape = pu.af_shape(s)
        array = ndarray(shape, dtype=self.dtype, af_array=s)
        if(shape != new_shape):
            array = array.reshape(new_shape)

        if new_shape == () and Ellipsis not in args:
            # Return the actual scalar
            return numpy.array(array)[()]

        return array

    def __setitem__(self, idx, value):
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
                if(len(idx) == 0):
                    self.d_array.setValue(afnumpy.arrayfire.index(0), value.d_array)
                elif(len(idx) == 1):
                    self.d_array.setValue(idx[0], value.d_array)
                elif(len(idx) == 2):
                    self.d_array.setValue(idx[0], idx[1], value.d_array)
                elif(len(idx) == 3):
                    self.d_array.setValue(idx[0], idx[1], idx[2], value.d_array)
                elif(len(idx) == 4):
                    self.d_array.setValue(idx[0], idx[1], idx[2], idx[3], value.d_array)
            else:
                self.d_array.setValue(idx, value.d_array)
        elif(isinstance(value, numbers.Number)):
            if(len(idx) == 0):
                self.d_array.setValue(afnumpy.arrayfire.index(0), value)
            elif(len(idx) == 1):
                self.d_array.setValue(idx[0], value)
            elif(len(idx) == 2):
                self.d_array.setValue(idx[0], idx[1], value)
            elif(len(idx) == 3):
                self.d_array.setValue(idx[0], idx[1], idx[2], value)
            elif(len(idx) == 4):
                self.d_array.setValue(idx[0], idx[1], idx[2], idx[3], value)
#            self.d_array.setValue(idx[0], value)
        else:
            raise NotImplementedError('values must be a afnumpy.ndarray')

    def __array__(self):
        self.d_array.host(self.h_array.ctypes.data)
        return numpy.copy(self.h_array)

    def transpose(self, *axes):
        if(self.ndim == 1):
            return self
        if len(axes) == 0 and self.ndim == 2:
            s = afnumpy.arrayfire.transpose(self.d_array)
        else:
            order = [0,1,2,3]
            if len(axes) == 0 or axes[0] is None:
                order[:self.ndim] = order[:self.ndim][::-1]
            else:
                if isinstance(axes[0], collections.Iterable):
                    axes = axes[0]
                for i,ax in enumerate(axes):
                    order[i] = pu.c2f(self.shape, ax)
                # We have to do this gymnastic due to the fact that arrayfire
                # uses Fortran order
                order[:len(axes)] = order[:len(axes)][::-1]

            #print order
            s = afnumpy.arrayfire.reorder(self.d_array, order[0],order[1],order[2],order[3])
        return ndarray(pu.af_shape(s), dtype=self.dtype, af_array=s)

    def reshape(self, shape, order = 'C'):
        a = array(self, copy=False)
        a.__reshape__(shape, order)
        return a
        
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
        if len(newshape) != 0:
            # No need to modify the af_array for empty shapes
            af_shape = numpy.array(pu.c2f(newshape), dtype=pu.dim_t)
            ret, handle = afnumpy.arrayfire.af_moddims(self.d_array.get(), af_shape.size, af_shape.ctypes.data)
            s = afnumpy.arrayfire.array_from_handle(handle)
            self.d_array = s

        self.h_array.shape = newshape
        self._shape = tuple(newshape)
        
    def flatten(self):
        return afnumpy.reshape(self, self.size)

    @reductufunc
    def max(self, s, axis):
        return afnumpy.arrayfire.max(s, axis)

    @reductufunc
    def min(self, s, axis):
        return afnumpy.arrayfire.min(s, axis)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        if(self.d_array):
            if(order != 'K'):
                raise NotImplementedError('only order=K implemented')
            if(casting != 'unsafe'):
                raise NotImplementedError('only casting=unsafe implemented')
            if(copy == False and order == 'K' and dtype == self.dtype):
                return self
            s = self.d_array.astype(pu.typemap(dtype))
            return ndarray(self.shape, dtype=dtype, af_array=s)
        else:
            return array(self.h_array.astype(dtype, order, casting, subok, copy), dtype=dtype)


    def round(self, decimals=0, out=None):
        if decimals != 0:
            raise NotImplementedError('only supports decimals=0')
        s = afnumpy.arrayfire.round(self.d_array)
        ret = ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)
        if(out):
            out[:] = ret[:]
        return ret

        
    def take(self, indices, axis=None, out=None, mode='raise'):
        if mode != 'raise':
            raise NotImplementedError('only supports mode=raise')
        if axis is None:
            ret = self.flat[indices]
        else:
            ret = self[(slice(None),)*axis+(indices,)]
        if out:
            out[:] = ret[:]
        return ret

    @outufunc
    @reductufunc
    def sum(self, s, axis):
        if self.dtype == numpy.bool:
            s = s.astype(pu.typemap(numpy.int64))
        return afnumpy.arrayfire.sum(s, axis)

    @outufunc
    @reductufunc
    def mean(self, s, axis):
        if self.dtype == numpy.bool:
            s = s.astype(pu.typemap(numpy.float64))
        return afnumpy.arrayfire.mean(s, axis)

    @outufunc
    @reductufunc
    def prod(self, s, axis):
        if self.dtype == numpy.bool:
            s = s.astype(pu.typemap(numpy.int64))
        return afnumpy.arrayfire.product(s, axis)

    product = prod

    @outufunc
    @reductufunc
    def all(self, s, axis):
        return afnumpy.arrayfire.allTrue(s, axis)

    @outufunc
    @reductufunc
    def any(self, s, axis):
        return afnumpy.arrayfire.anyTrue(s, axis)


    def conj(self):
        if not numpy.issubdtype(self.dtype, numpy.complex):
            return afnumpy.copy(self)
        if(self.d_array):
            s = afnumpy.arrayfire.conjg(self.d_array)
            return ndarray(self.shape, dtype=pu.typemap(s.type()), af_array=s)
        else:
            return self.h_array.conj()

    # Convert to float
    def __float__(self):
        if self.size != 1:
            raise TypeError('only length-1 arrays can be converted to Python scalars')
        ret = self[(0,)*self.ndim]
        return ret

    def squeeze(self, axis=None):
        if axis is None:
            axis = tuple(i for i, x in enumerate(self.shape) if x == 1)
        if not isinstance(axis, tuple):
            axis = (axis,)
        newshape = list(self.shape)
        for a in sorted(axis)[::-1]:
            newshape.pop(a)
        return self.reshape(newshape)

    @property            
    def T(self):
        if self.ndim < 2:
            return self
        return self.transpose()

    def argmax(self, axis=None):
        if axis is None:
            return self.flat.argmax(axis=0)
        if not isinstance(axis, numbers.Number):
            raise TypeError('an integer is required for the axis')
        val = afnumpy.arrayfire.array()
        idx = afnumpy.arrayfire.array()
        afnumpy.arrayfire.max(val, idx, self.d_array, pu.c2f(self.shape, axis))
        shape = list(self.shape)
        shape.pop(axis)
        if(len(shape)):
            return ndarray(shape, dtype=pu.typemap(idx.type()), af_array=idx)
        else:
            return ndarray(shape, dtype=pu.typemap(idx.type()), af_array=idx)[()]

    def argmin(self, axis=None):
        if axis is None:
            return self.flat.argmin(axis=0)
        if not isinstance(axis, numbers.Number):
            raise TypeError('an integer is required for the axis')
        val = afnumpy.arrayfire.array()
        idx = afnumpy.arrayfire.array()
        afnumpy.arrayfire.min(val, idx, self.d_array, pu.c2f(self.shape, axis))
        shape = list(self.shape)
        shape.pop(axis)
        if(len(shape)):
            return ndarray(shape, dtype=pu.typemap(idx.type()), af_array=idx)
        else:
            return ndarray(shape, dtype=pu.typemap(idx.type()), af_array=idx)[()]
            
        
    def argsort(self, axis=-1, kind='quicksort', order=None):
        if kind != 'quicksort':
            print "argsort 'kind' argument ignored"
        if order is not None:
            raise ValueError('order argument is not supported')
        if(axis < 0):
            axis = self.ndim+axis
        val = afnumpy.arrayfire.array()
        idx = afnumpy.arrayfire.array()
        afnumpy.arrayfire.sort(val, idx, self.d_array, pu.c2f(self.shape, axis))
        return ndarray(self.shape, dtype=pu.typemap(idx.type()), af_array=idx)

    @property            
    def base(self):
        return self._base

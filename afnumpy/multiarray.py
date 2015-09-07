import ctypes
import arrayfire
import sys
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
            s = arrayfire.cast(object.d_array.copy(), pu.typemap(dtype))
        else:
            s = arrayfire.cast(object.d_array, pu.typemap(dtype))
        return ndarray(shape, dtype=dtype, af_array=s)
    elif(isinstance(object, numpy.ndarray)):
        return ndarray(shape, dtype=dtype, buffer=object.astype(dtype, copy=copy))
    else:
        raise AssertionError
        
def arange(start, stop = None, step = None, dtype=None):
    return afnumpy.array(numpy.arange(start,stop,step,dtype))        
 
def where(condition, x=pu.dummy, y=pu.dummy):
    a = condition
    s = arrayfire.where(a.d_array)
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
                out_arr = ctypes.c_void_p(0)
                if(buffer is not None):
                    arrayfire.backend.get().af_create_array(ctypes.pointer(out_arr), ctypes.c_void_p(buffer.ctypes.data),
                                                            s_a.size, ctypes.c_void_p(s_a.ctypes.data), pu.typemap(dtype).value)
                else:
                    arrayfire.backend.get().af_create_handle(ctypes.pointer(out_arr), s_a.size, ctypes.c_void_p(s_a.ctypes.data), pu.typemap(dtype).value)
                self.d_array = arrayfire.Array()
                self.d_array.arr = out_arr
        else:
            raise NotImplementedError('Only up to 4 dimensions are supported')
        self.h_array = numpy.ndarray(shape,dtype,buffer,offset,strides,order)
        
    def __repr__(self):
        arrayfire.backend.get().af_get_data_ptr(ctypes.c_void_p(self.h_array.ctypes.data), self.d_array.arr)
        return self.h_array.__repr__()        

    def __str__(self):
        return self.__repr__()

    @ufunc
    def __add__(self, other):
        s = self.d_array + pu.raw(other)
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

    @iufunc
    def __iadd__(self, other):
        self[:] = self[:] + pu.raw(other)
        return self

    def __radd__(self, other):
        s = pu.raw(other) + self.d_array
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

    @ufunc
    def __sub__(self, other):
        s = self.d_array - pu.raw(other)
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

    @iufunc
    def __isub__(self, other):
        afnumpy.subtract(self, pu.raw(other), out=self)
        return self

    def __rsub__(self, other):
        s = pu.raw(other) - self.d_array
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

    @ufunc
    def __mul__(self, other):
        s = self.d_array * pu.raw(other)
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

    @iufunc
    def __imul__(self, other):
        afnumpy.multiply(self, pu.raw(other), out=self)
        return self

    def __rmul__(self, other):
        s = pu.raw(other) * self.d_array
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

    @ufunc
    def __div__(self, other):
        s = self.d_array / pu.raw(other)
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

    @iufunc
    def __idiv__(self, other):
        afnumpy.divide(self, pu.raw(other), out=self)
        return self

    def __rdiv__(self, other):
        s = pu.raw(other) / self.d_array
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        
    def __pow__(self, other):
        if(isinstance(other, numbers.Number) and numpy.issubdtype(type(other), numpy.float) and
           numpy.issubdtype(self.dtype, numpy.integer)):
            # AF does not automatically upconvert A**0.5 to float for integer arrays
            s = arrayfire.pow(self.astype(type(other)).d_array, pu.raw(other))
        else:
            s = arrayfire.pow(self.d_array, pu.raw(other))
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

    def __rpow__(self, other):
        if(isinstance(other, numbers.Number) and numpy.issubdtype(type(other), numpy.float) and
           numpy.issubdtype(self.dtype, numpy.integer)):
            # AF does not automatically upconvert A**0.5 to float for integer arrays
            s = arrayfire.pow(pu.raw(other), self.astype(type(other)).d_array)
        else:
            s = arrayfire.pow(pu.raw(other), self.d_array)
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

    def __lt__(self, other):
        s = self.d_array < pu.raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __le__(self, other):
        s = self.d_array <= pu.raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __gt__(self, other):
        s = self.d_array > pu.raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __ge__(self, other):
        s = self.d_array >= pu.raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __eq__(self, other):
        if(other is None):
            return False
        s = self.d_array == pu.raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __ne__(self, other):
        if(other is None):
            return True
        s = self.d_array != pu.raw(other)
        return ndarray(self.shape, dtype=numpy.bool, af_array=s)

    def __abs__(self):
        s = arrayfire.abs(self.d_array)
        # dtype is wrong for complex types
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

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
#        out.d_array.eval()
        arrayfire.backend.get().af_eval(out.d_array.arr)
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
        ret = ndarray(self.size, dtype=self.dtype, af_array=arrayfire.flat(self.d_array))
        ret._base = self
        return ret

    @property
    def real(self):
        ret_type = numpy.real(numpy.zeros((),dtype=self.dtype)).dtype
        shape = list(self.shape)
        if not numpy.issubdtype(self.dtype, numpy.complexfloating):
            return self

        shape[-1] *= 2
        dims = numpy.array(pu.c2f(shape),dtype=pu.dim_t)
        s = arrayfire.Array()
        arrayfire.backend.get().af_device_array(ctypes.pointer(s.arr),
                                                ctypes.c_void_p(self.d_array.device_ptr()),
                                                self.ndim,
                                                ctypes.c_void_p(dims.ctypes.data),
                                                pu.typemap(ret_type).value)
        arrayfire.backend.get().af_retain_array(ctypes.pointer(s.arr),s.arr)
        a = ndarray(shape, dtype=ret_type, af_array=s)
        ret = a[...,::2]
        ret._base = a
        ret._base_index = (Ellipsis, slice(None,None,2))
        return ret

    @property
    def imag(self):
        ret_type = numpy.real(numpy.zeros((),dtype=self.dtype)).dtype
        shape = list(self.shape)
        if not numpy.issubdtype(self.dtype, numpy.complexfloating):
            return afnumpy.zeros(self.shape)
        shape[-1] *= 2
        dims = numpy.array(pu.c2f(shape),dtype=pu.dim_t)
        s = arrayfire.Array()
        arrayfire.backend.get().af_device_array(ctypes.pointer(s.arr),
                                                ctypes.c_void_p(self.d_array.device_ptr()),
                                                self.ndim,
                                                ctypes.c_void_p(dims.ctypes.data),
                                                pu.typemap(ret_type).value)
        arrayfire.backend.get().af_retain_array(ctypes.pointer(s.arr),s.arr)
        a = ndarray(shape, dtype=ret_type, af_array=s)
        ret = a[...,1::2]
        ret._base = a
        ret._base_index = (Ellipsis, slice(1,None,2))
        return ret
 
    def ravel(self, order=None):
        if(order != None and order != 'K' and order != 'C'):
            raise NotImplementedError('order %s not supported' % (order))
        return self.flat

    def __iter__(self):
        ret = []
        for i in range(0,len(self)):
            ret.append(self[i])
        return iter(ret)

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        if len(args) == 1 and isinstance(args[0], afnumpy.ndarray) and args[0].dtype == numpy.dtype('bool'):
            # Special case for boolean getitem
            return self.flat[afnumpy.where(args[0].flat)]
        idx, new_shape = indexing.__convert_dim__(self.shape, args)
        if any(x is None for x in idx):
            # one of the indices is empty
            return ndarray(indexing.__index_shape__(self.shape, idx), dtype=self.dtype)
        idx = tuple(idx)
        if len(idx) == 0:
            idx = tuple([0])
        s = self.d_array[idx]
        shape = pu.af_shape(s)
        array = ndarray(shape, dtype=self.dtype, af_array=s)
        if(shape != new_shape):
            array = array.reshape(new_shape)

        if new_shape == () and Ellipsis not in args:
            # Return the actual scalar
            return numpy.array(array)[()]

        return array

    def __setitem__(self, idx, value):       
        try:
            if idx.dtype == numpy.dtype('bool') or (idx[0].dtype == 'bool' and len(idx) == 1):
                # Special case for boolean setitem
                self_flat = self.flat
                idx = afnumpy.where(idx.flat)
                self_flat[idx] = value
                return
        except AttributeError:
            pass
        except RuntimeError:
            # idx is all False
            return

        idx, idx_shape = indexing.__convert_dim__(self.shape, idx)
        if any(x is None for x in idx):
            # one of the indices is empty
            return            
        idx = tuple(idx)
        if len(idx) == 0:
            idx = tuple([0])
        if(isinstance(value, ndarray)):
            if(value.dtype != self.dtype):
                raise TypeError('left hand side must have same dtype as right hand side')
            value = indexing.__expand_dim__(self.shape, value, idx).d_array
        elif(isinstance(value, numbers.Number)):
            pass
        else:
            raise NotImplementedError('values must be a afnumpy.ndarray')
        self.d_array[idx] = value            
        # This is a hack to be able to make it look like arrays with stride[0] > 1 can still be views
        # In practise right now it only applies to ndarray.real and ndarray.imag
        try:
            self._base[self._base_index] = self
        except AttributeError:
            pass

    def __array__(self):
        arrayfire.backend.get().af_get_data_ptr(ctypes.c_void_p(self.h_array.ctypes.data), self.d_array.arr)
        return numpy.copy(self.h_array)

    def transpose(self, *axes):
        if(self.ndim == 1):
            return self
        if len(axes) == 0 and self.ndim == 2:
            s = arrayfire.transpose(self.d_array)
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
            s = arrayfire.reorder(self.d_array, order[0],order[1],order[2],order[3])
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
            s = arrayfire.Array()
#            Tracer()()
            arrayfire.backend.get().af_moddims(ctypes.pointer(s.arr), self.d_array.arr, af_shape.size, ctypes.c_void_p(af_shape.ctypes.data))
#            arrayfire.backend.get().af_moddims(ctypes.pointer(self.d_array.arr), self.d_array.arr, af_shape.size, ctypes.c_void_p(af_shape.ctypes.data))
            self.d_array = s

        self.h_array.shape = newshape
        self._shape = tuple(newshape)
        
    def flatten(self):
        return afnumpy.reshape(self, self.size)

    @reductufunc
    def max(self, s, axis):
        return arrayfire.max(s, axis)

    @reductufunc
    def min(self, s, axis):
        return arrayfire.min(s, axis)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        if(self.d_array is not None):
            if(order != 'K'):
                raise NotImplementedError('only order=K implemented')
            if(casting != 'unsafe'):
                raise NotImplementedError('only casting=unsafe implemented')
            if(copy == False and order == 'K' and dtype == self.dtype):
                return self
#            s = self.d_array.astype(pu.typemap(dtype))
            s = arrayfire.cast(self.d_array, pu.typemap(dtype))
            return ndarray(self.shape, dtype=dtype, af_array=s)
        else:
            return array(self.h_array.astype(dtype, order, casting, subok, copy), dtype=dtype)


    def round(self, decimals=0, out=None):
        if decimals != 0:
            raise NotImplementedError('only supports decimals=0')
        s = arrayfire.round(self.d_array)
        ret = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
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
            s = arrayfire.cast(s, pu.typemap(numpy.int64))
#            s = s.astype(pu.typemap(numpy.int64))
        return arrayfire.sum(s, axis)

    @outufunc
    @reductufunc
    def mean(self, s, axis):
        if self.dtype == numpy.bool:
            s = s.astype(pu.typemap(numpy.float64))
        return arrayfire.mean(s, axis)

    @outufunc
    @reductufunc
    def prod(self, s, axis):
        if self.dtype == numpy.bool:
            s = s.astype(pu.typemap(numpy.int64))
        return arrayfire.product(s, axis)

    product = prod

    @outufunc
    @reductufunc
    def all(self, s, axis):
        return arrayfire.all_true(s, axis)

    @outufunc
    @reductufunc
    def any(self, s, axis):
        return arrayfire.any_true(s, axis)


    def conj(self):
        if not numpy.issubdtype(self.dtype, numpy.complex):
            return afnumpy.copy(self)
        if(self.d_array is not None):
            s = arrayfire.conjg(self.d_array)
            return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
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
        val, idx = arrayfire.imax(self.d_array, pu.c2f(self.shape, axis))
        shape = list(self.shape)
        shape.pop(axis)
        if(len(shape)):
            return ndarray(shape, dtype=pu.typemap(idx.dtype()), af_array=idx)
        else:
            return ndarray(shape, dtype=pu.typemap(idx.dtype()), af_array=idx)[()]

    def argmin(self, axis=None):
        if axis is None:
            return self.flat.argmin(axis=0)
        if not isinstance(axis, numbers.Number):
            raise TypeError('an integer is required for the axis')
        val, idx = arrayfire.imin(self.d_array, pu.c2f(self.shape, axis))
        shape = list(self.shape)
        shape.pop(axis)
        if(len(shape)):
            return ndarray(shape, dtype=pu.typemap(idx.dtype()), af_array=idx)
        else:
            return ndarray(shape, dtype=pu.typemap(idx.dtype()), af_array=idx)[()]
            
        
    def argsort(self, axis=-1, kind='quicksort', order=None):
        if kind != 'quicksort':
            print "argsort 'kind' argument ignored"
        if order is not None:
            raise ValueError('order argument is not supported')
        if(axis < 0):
            axis = self.ndim+axis
        val, idx = arrayfire.sort_index(self.d_array, pu.c2f(self.shape, axis))
        return ndarray(self.shape, dtype=pu.typemap(idx.dtype()), af_array=idx)

    @property            
    def base(self):
        return self._base

import ctypes
import arrayfire
import sys
import numpy
import numbers
from . import private_utils as pu
import afnumpy
from . import indexing
from .decorators import *
import collections




class ndarray(object):
    # Ensures that our functions are called before numpy ones
    __array_priority__ = 20
    def __init__(self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, af_array=None, buffer_type='python'):
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
        # Make sure you transform the type to a numpy dtype
        self.dtype = numpy.dtype(dtype)
        s_a = numpy.array(pu.c2f(shape),dtype=pu.dim_t)
        if(s_a.size < 1):
            # We'll use af_arrays of size (1) to keep scalars
            s_a = numpy.array((1),dtype=pu.dim_t)
        if(s_a.size > 4):
            raise NotImplementedError('Only up to 4 dimensions are supported')
        if(af_array is not None):
            self.d_array = af_array
            # Remove leading and trailing dimensions of size 1
            af_dims = list(af_array.dims())
            while len(af_dims) and af_dims[-1] == 1:
                af_dims.pop()
            if s_a.shape == ():
                arg_dims = [1]
            else:
                arg_dims = list(s_a)
            while len(arg_dims) and arg_dims[-1] == 1:
                arg_dims.pop()
            if af_dims != arg_dims:
                raise ValueError('shape argument not consistent with the dimensions of the af_array given')
        else:
            out_arr = ctypes.c_void_p(0)
            if(buffer is not None):
                if buffer_type == 'python':
                    # normal python buffer. We copy the data to arrayfire
                    ptr = numpy.frombuffer(buffer, dtype='int8').ctypes.data
                    arrayfire.backend.get().af_create_array(ctypes.pointer(out_arr), ctypes.c_void_p(ptr),
                                                            s_a.size, ctypes.c_void_p(s_a.ctypes.data), pu.typemap(dtype).value)
                elif buffer_type == afnumpy.arrayfire.get_active_backend():
                    # in this case buffer is a device memory address. We create the array without copying
                    ptr = buffer
                    arrayfire.backend.get().af_device_array(ctypes.pointer(out_arr), ctypes.c_void_p(ptr),
                                                            s_a.size, ctypes.c_void_p(s_a.ctypes.data), pu.typemap(dtype).value)
                    # Do not release the memory on destruction
                    arrayfire.backend.get().af_retain_array(ctypes.pointer(out_arr),out_arr)
                else:
                    raise ValueError("buffer_type must match afnumpy.arrayfire.get_active_backend() or be 'python'")
            else:
                arrayfire.backend.get().af_create_handle(ctypes.pointer(out_arr), s_a.size, ctypes.c_void_p(s_a.ctypes.data), pu.typemap(dtype).value)
            self.d_array = arrayfire.Array()
            self.d_array.arr = out_arr

        # Check if array size matches the af_array size
        # This is necessary as certain operations that cause reduction in
        # dimensions in numpy do not necessarily do that in arrayfire
        if af_array is not None and self.d_array.dims() != pu.c2f(self._shape):
            self.__reshape__(self._shape)


    def __repr__(self):
        h_array = numpy.empty(shape=self.shape, dtype=self.dtype)
        if self.size:
            arrayfire.backend.get().af_get_data_ptr(ctypes.c_void_p(h_array.ctypes.data), self.d_array.arr)
        return h_array.__repr__()

    def __str__(self):
        return self.__repr__()

    def __format__(self, f):
        h_array = numpy.empty(shape=self.shape, dtype=self.dtype)
        if self.size:
            arrayfire.backend.get().af_get_data_ptr(ctypes.c_void_p(h_array.ctypes.data), self.d_array.arr)
        return h_array.__format__(f)
        
    @ufunc
    def __add__(self, other):
        s = self.d_array + pu.raw(other)
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    @iufunc
    def __iadd__(self, other):
        afnumpy.add(self, pu.raw(other), out=self)
        self._eval()
        return self

    def __radd__(self, other):
        s = pu.raw(other) + self.d_array
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    @ufunc
    def __sub__(self, other):
        s = self.d_array - pu.raw(other)
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    @iufunc
    def __isub__(self, other):
        afnumpy.subtract(self, pu.raw(other), out=self)
        self._eval()
        return self

    def __rsub__(self, other):
        s = pu.raw(other) - self.d_array
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    @ufunc
    def __mul__(self, other):
        s = self.d_array * pu.raw(other)
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    @iufunc
    def __imul__(self, other):
        afnumpy.multiply(self, pu.raw(other), out=self)
        self._eval()
        return self

    def __rmul__(self, other):
        s = pu.raw(other) * self.d_array
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    @ufunc
    def __div__(self, other):
        s = self.d_array / pu.raw(other)
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    __floordiv__ = __div__

    @ufunc
    def __truediv__(self, other):
        # Check if we need to cast input to floating point to get a true division
        if(pu.isintegertype(self) and pu.isintegertype(other)):
            s = self.astype(numpy.float32).d_array / pu.raw(other)
        else:
            s = self.d_array / pu.raw(other)
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    @iufunc
    def __idiv__(self, other):
        afnumpy.floor_divide(self, pu.raw(other), out=self)
        self._eval()
        return self

    @iufunc
    def __itruediv__(self, other):
        afnumpy.true_divide(self, pu.raw(other), out=self)
        self._eval()
        return self

    def __rdiv__(self, other):
        s = pu.raw(other) / self.d_array
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a


    def __rtruediv__(self, other):
        # Check if we need to cast input to floating point to get a true division
        if(pu.isintegertype(self) and pu.isintegertype(other)):
            s = pu.raw(other) / self.astype(numpy.float32).d_array
        else:
            s = pu.raw(other) / self.d_array
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    def __pow__(self, other):
        if(isinstance(other, numbers.Number) and numpy.issubdtype(type(other), numpy.float64) and
           numpy.issubdtype(self.dtype, numpy.integer)):
            # AF does not automatically upconvert A**0.5 to float for integer arrays
            s = arrayfire.pow(self.astype(type(other)).d_array, pu.raw(other))
        else:
            s = arrayfire.pow(self.d_array, pu.raw(other))
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    def __rpow__(self, other):
        if(isinstance(other, numbers.Number) and numpy.issubdtype(type(other), numpy.float64) and
           numpy.issubdtype(self.dtype, numpy.integer)):
            # AF does not automatically upconvert A**0.5 to float for integer arrays
            s = arrayfire.pow(pu.raw(other), self.astype(type(other)).d_array)
        else:
            s = arrayfire.pow(pu.raw(other), self.d_array)
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    def __lt__(self, other):
        s = self.d_array < pu.raw(other)
        a = ndarray(self.shape, dtype=numpy.bool, af_array=s)
        a._eval()
        return a

    def __le__(self, other):
        s = self.d_array <= pu.raw(other)
        a = ndarray(self.shape, dtype=numpy.bool, af_array=s)
        a._eval()
        return a

    def __gt__(self, other):
        s = self.d_array > pu.raw(other)
        a = ndarray(self.shape, dtype=numpy.bool, af_array=s)
        a._eval()
        return a

    def __ge__(self, other):
        s = self.d_array >= pu.raw(other)
        a = ndarray(self.shape, dtype=numpy.bool, af_array=s)
        a._eval()
        return a

    def __eq__(self, other):
        if(other is None):
            return False
        s = self.d_array == pu.raw(other)
        a = ndarray(self.shape, dtype=numpy.bool, af_array=s)
        a._eval()
        return a

    def __ne__(self, other):
        if(other is None):
            return True
        s = self.d_array != pu.raw(other)
        a = ndarray(self.shape, dtype=numpy.bool, af_array=s)
        a._eval()
        return a

    def __abs__(self):
        s = arrayfire.abs(self.d_array)
        # dtype is wrong for complex types
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

    def __neg__(self):
        if self.dtype == numpy.dtype('bool'):
            # Special case for boolean getitem
            a = afnumpy.array([True]) - self
        else:
            a = self * self.dtype.type(-1)
        a._eval()
        return a

    def __pos__(self):
        return afnumpy.array(self)

    def __invert__(self):
        raise NotImplementedError

    def __nonzero__(self):
        # This should be improved
        return numpy.array(self).__nonzero__()

    def __len__(self):
        return self.shape[0]

    def __mod__(self, other):
        s = self.d_array % pu.raw(other)
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

    def __rmod__(self, other):
        s = pu.raw(other) % self.d_array
        a = ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)
        a._eval()
        return a

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
    def flat(self):
        # Currently arrayfire.flat is doing unnecessary copies
        # ret = ndarray(self.size, dtype=self.dtype, af_array=arrayfire.flat(self.d_array))
        ret = self.reshape(-1)
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
        try:
            idx, new_shape, input_shape = indexing.__convert_dim__(self.shape, args)
        except NotImplementedError:
            # Slow indexing method for not currently implemented fancy indexing
            return afnumpy.array(numpy.array(self).__getitem__(args))
        if numpy.prod(new_shape) == 0:
            # We're gonna end up with an empty array
            # As we don't yet support empty arrays return an empty numpy array
            return numpy.empty(new_shape, dtype=self.dtype)

        if any(x is None for x in idx):
            # one of the indices is empty
            return ndarray(indexing.__index_shape__(self.shape, idx), dtype=self.dtype)
        idx = tuple(idx)
        if len(idx) == 0:
            idx = tuple([0])
        s = self.reshape(input_shape).d_array[idx]
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
            idx, idx_shape, input_shape = indexing.__convert_dim__(self.shape, idx)
        except NotImplementedError:
            # Slow indexing method for not currently implemented fancy indexing
            a = numpy.array(self)
            a.__setitem__(idx, value)
            self[...] = afnumpy.array(a)
            return

        if numpy.prod(idx_shape) == 0:
            # We've selected an empty array
            # No need to do anything
            return
        if any(x is None for x in idx):
            # one of the indices is empty
            return
        idx = tuple(idx)
        if len(idx) == 0:
            idx = tuple([0])
        if(isinstance(value, numbers.Number)):
            pass
        elif(isinstance(value, ndarray)):
            if(value.dtype != self.dtype):
                value.astype(self.dtype)
            value = indexing.__expand_dim__(self.shape, value, idx).d_array
        else:
            raise NotImplementedError('values must be a afnumpy.ndarray')
        self.reshape(input_shape).d_array[idx] = value
        # This is a hack to be able to make it look like arrays with stride[0] > 1 can still be views
        # In practise right now it only applies to ndarray.real and ndarray.imag
        try:
            self._base[self._base_index] = self
        except AttributeError:
            pass

    def __array__(self, dtype=None):
        h_array = numpy.empty(shape=self.shape,dtype=self.dtype)
        if self.size:
            arrayfire.backend.get().af_get_data_ptr(ctypes.c_void_p(h_array.ctypes.data), self.d_array.arr)
        if dtype is None:
            return h_array
        else:
            return h_array.astype(dtype)

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
        a = afnumpy.array(self, copy=False)
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
            newshape[i] = self.size//numpy.prod(newshape)
        if self.size != numpy.prod(newshape):
            raise ValueError('total size of new array must be unchanged')
        if len(newshape) != 0:
            # No need to modify the af_array for empty shapes
            # af_shape = numpy.array(pu.c2f(newshape), dtype=pu.dim_t)
            # s = arrayfire.Array()
            # arrayfire.backend.get().af_moddims(ctypes.pointer(s.arr), self.d_array.arr, af_shape.size, ctypes.c_void_p(af_shape.ctypes.data))
            # self.d_array = s
            if tuple(newshape) == self.shape:
                # No need to do anything
                return
            af_shape = numpy.array(pu.c2f(newshape), dtype=pu.dim_t)
            s = arrayfire.Array()
            arrayfire.backend.get().af_moddims(ctypes.pointer(s.arr), self.d_array.arr,
                                               af_shape.size, ctypes.c_void_p(af_shape.ctypes.data))
            self.d_array = s

        self._shape = tuple(newshape)

    def flatten(self, order='C'):
        if(order != None and order != 'K' and order != 'C' and order != 'A'):
            raise NotImplementedError('order %s not supported' % (order))
        return afnumpy.reshape(self, self.size).copy()

    @reductufunc
    def max(self, s, axis):
        return arrayfire.max(s, axis)

    @reductufunc
    def min(self, s, axis):
        return arrayfire.min(s, axis)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        if(order != 'K'):
            raise NotImplementedError('only order=K implemented')
        if(casting != 'unsafe'):
            raise NotImplementedError('only casting=unsafe implemented')
        if(copy == False and order == 'K' and dtype == self.dtype):
            return self
        s = arrayfire.cast(self.d_array, pu.typemap(dtype))
        a = ndarray(self.shape, dtype=dtype, af_array=s)
        a._eval()
        return a


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
        return arrayfire.sum(s, dim=axis)

    @outufunc
    @reductufunc
    def mean(self, s, axis):
        if self.dtype == numpy.bool:
            s = s.astype(pu.typemap(numpy.float64))
        return arrayfire.mean(s, dim=axis)

    @outufunc
    @reductufunc
    def prod(self, s, axis):
        if self.dtype == numpy.bool:
            s = s.astype(pu.typemap(numpy.int64))
        return arrayfire.product(s, dim=axis)

    product = prod

    @outufunc
    @reductufunc
    def all(self, s, axis):
        return arrayfire.all_true(s, dim=axis)

    @outufunc
    @reductufunc
    def any(self, s, axis):
        return arrayfire.any_true(s, dim=axis)


    def conj(self):
        if not numpy.issubdtype(self.dtype, numpy.complex):
            return afnumpy.copy(self)
        s = arrayfire.conjg(self.d_array)
        return ndarray(self.shape, dtype=pu.typemap(s.dtype()), af_array=s)

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
            print( "argsort 'kind' argument ignored" )
        if order is not None:
            raise ValueError('order argument is not supported')
        if(axis is None):            
            input = self.flatten()
            axis = 0
        else:
            input = self
        if(axis < 0):
            axis = self.ndim+axis
        val, idx = arrayfire.sort_index(input.d_array, pu.c2f(input.shape, axis))
        return ndarray(input.shape, dtype=pu.typemap(idx.dtype()), af_array=idx)

    @property
    def base(self):
        return self._base

    @property
    def strides(self):
        strides = ()
        # we have access to the stride functions
        if afnumpy.arrayfire_version(numeric=True) >= 3003000:
            strides = pu.c2f(self.d_array.strides()[0:self.ndim])
            if len(strides) < self.ndim and self.ndim > 1:
                strides = (strides[0]*self.shape[1],) + strides
            strides = tuple([s*self.dtype.itemsize for s in strides])
        else:
            idx = (slice(1,None),)
            base_addr = self.d_array.device_ptr()
            dims = self.d_array.dims()
            # Append any missing ones
            dims = dims + (1,)*(self.ndim-len(dims))
            for i in range(0, self.ndim):
                if(dims[i] > 1):
                    strides = (self.d_array[idx].device_ptr()-base_addr,)+strides
                else:
                    if len(strides):
                        strides = (dims[i-1]*numpy.prod(strides),)+strides
                    else:
                        strides = (self.itemsize,)+strides
                idx = (slice(None),)+idx
        return strides

    @property
    def itemsize(self):
        return self.dtype.itemsize

    def eval(self):
        return arrayfire.backend.get().af_eval(self.d_array.arr)

    def _eval(self):
        if afnumpy.force_eval:
            return self.eval()
        else:
            return 0

    def copy(self, order='C'):
        return afnumpy.array(self, copy=True, order=order)

    def nonzero(self):
        s = arrayfire.where(self.d_array)
        s = ndarray(pu.af_shape(s), dtype=numpy.uint32,
                    af_array=s).astype(numpy.int64)
        # TODO: Unexplained eval
        s.eval()
        idx = []
        mult = 1
        for i in self.shape[::-1]:
            mult = i
            idx = [s % mult] + idx
            s //= mult
        idx = tuple(idx)
        return idx

    def sort(self, axis=-1, kind='quicksort', order=None):
        if kind != 'quicksort':
            print( "sort 'kind' argument ignored" )
        if order is not None:
            raise ValueError('order argument is not supported')
        if(axis is None):            
            input = self.flatten()
            axis = 0
        else:
            input = self
        if(axis < 0):
            axis = self.ndim+axis
        s = arrayfire.sort(input.d_array, pu.c2f(input.shape, axis))
        self.d_array = s

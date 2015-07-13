import numpy
#import arrayfire
import numbers
from IPython.core.debugger import Tracer
import private_utils as pu
import afnumpy

def fromstring(string, dtype=float, count=-1, sep=''):
    return array(numpy.fromstring(string, dtype, count, sep))

def vdot(a, b):
    s = afnumpy.arrayfire.dot(afnumpy.arrayfire.conjg(a.d_array), b.d_array)
    return ndarray(pu.af_shape(s), dtype=a.dtype, af_array=s)

def zeros(shape, dtype=float, order='C'):
    b = numpy.zeros(shape, dtype, order)
    return ndarray(b.shape, b.dtype, buffer=b,order=order)

def array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0):
    if(order is not None):
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
        

def where(condition, x=pu.dummy, y=pu.dummy):
    a = condition
    s = afnumpy.arrayfire.where(a.d_array)
    idx = []
    mult = 1
    # numpy uses int64 while arrayfire uses uint32
    s = ndarray(pu.af_shape(s), dtype=numpy.uint32, af_array=s).astype(numpy.int64)
    Tracer()()
    for i in a.shape[::-1]:
        mult *= i
        idx = [s % mult] + idx 
        s /= mult
    idx = tuple(idx)
    if(x is pu.dummy and y is pu.dummy):
        return idx
    elif(x is not pu.dummy and y is not pu.dummy):
        if(x.dtype != y.dtype):
            raise TypeError('x and y must have same dtype')
        if(x.shape != y.shape):
            raise ValueError('x and y must have same shape')
        ret = array(y)
        ret[idx] = x[idx]
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
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)
        self.dtype = dtype
        s_a = numpy.array(pu.c2f(shape),dtype=pu.dim_t)
        if(s_a.size < 1):
            self.d_array = None
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

    def __add__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__add__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(self.h_array + other, dtype=self.dtype)

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

    def __sub__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__sub__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(self.h_array - other, dtype=self.dtype)

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

    def __mul__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__mul__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(self.h_array * other, dtype=self.dtype)

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

    def __div__(self, other):
        if(self.d_array):
            s = afnumpy.arrayfire.__div__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=pu.InvTypeMap[s.type()], af_array=s)
        else:
            return array(self.h_array / other, dtype=self.dtype)

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
        if(self.d_array):
            s = afnumpy.arrayfire.__eq__(self.d_array, pu.raw(other))
            return ndarray(self.shape, dtype=numpy.bool, af_array=s)
        else:
            return array(self.h_array == other, dtype=self.dtype)

    def __ne__(self, other):
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
    def size(self):
        return numpy.prod(self.shape)

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        if(self.d_array is None):
            raise IndexError('too many indices for array')
        idx = self.__convert_dim__(args)
        if None in idx:
            # one of the indices is empty
            return ndarray(self.__index_shape__(idx), dtype=self.dtype)

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
        shape = list(shape)
        if isinstance(args, tuple):
            while(len(shape) < len(args)):
                shape = [1]+shape
            while(len(args) < len(shape)):
                args = args+(slice(None),)


        # ISSUE: Looks like afnumpy contracts dimensions in certain
        # cases and not in others. This should be checked out
        
        # Remove dimensions corresponding to non slices
        Tracer()()
        if(isinstance(args, tuple)):
            new_shape = []
            for axis in range(0,len(args)):
#                if(isinstance(args[axis], slice)):
                if not isinstance(args[axis], numbers.Number):
                    new_shape.append(shape[axis])
            if(new_shape != list(shape)):
                array = array.reshape(new_shape)
        return array

    def __slice_to_seq__(self, idx, axis):
        maxlen = self.shape[axis]
        if(isinstance(idx, numbers.Number)):
            if idx < 0:
                idx = maxlen + idx
            if(idx >= maxlen):
                raise IndexError('index %d is out of bounds for axis %d with size %d' % (idx, axis, maxlen))
            #return afnumpy.arrayfire.seq(float(idx), float(idx), float(1))
            return idx        

        if(isinstance(idx, ndarray)):
            return idx.d_array

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
            if(start < 0):
                start += maxlen
        if idx.stop is None:
            if step < 0:
                end = 0
            else:
                end = maxlen-1
        else:
            end = idx.stop
            if(end < 0):
                end += maxlen
            if step < 0:
                end += 1
            else:
                end -= 1
        # arrayfire doesn't like other steps in this case
        if(start == end):
            step = 1
        
        if((start-end > 0 and step > 0) or
           (start-end < 0 and step < 0)):
            return None           
        return  afnumpy.arrayfire.seq(float(start),
                              float(end),
                              float(step))

    def __convert_dim__(self, idx):
        # Convert numpy style indexing arguments to arrayfire style
        # Always returns a list
        maxlen = self.shape[0]
        if(isinstance(idx, ndarray)):
            return [afnumpy.arrayfire.index(idx.d_array)]
        if(isinstance(idx, slice)):
            idx = (idx,)
        if(isinstance(idx, numbers.Number)):
            idx = (idx,)
        if(isinstance(idx, tuple)):
            idx = list(idx)
            while len(idx) < len(self.shape):
                idx.append(slice(None,None,None))
            ret = [0]*len(self.shape)
            for axis in range(0,len(self.shape)):
                seq = self.__slice_to_seq__(idx[axis],axis)
                if(seq is not None):
                    ret[pu.c2f(self.shape,axis)] = afnumpy.arrayfire.index(seq)
                else:
                    ret[pu.c2f(self.shape,axis)] = None
            return ret
        raise NotImplementedError('indexing with %s not implemented' % (type(idx)))

    def __index_shape__(self, idx):
        shape = []
        for i in range(0,len(idx)):
            if(idx[i] is None):
                shape.append(0)
            elif(idx[i].isspan()):
                shape.append(self.shape[i])
            else:
                af_idx = idx[i].get()
                if(af_idx.isBatch):
                    raise ValueError
                if(af_idx.isSeq):
                    shape.append(afnumpy.arrayfire.seq(af_idx.seq()).size)
                else:
                    shape.append(af_idx.arr_elements())
        return pu.c2f(shape)
        
    def __expand_dim__(self, value, idx):
        # reshape value, adding size 1 dimensions, such that the dimensions of value match idx
        idx_shape = self.__index_shape__(idx)
        value_shape = list(value.shape)        
        past_one_dims = False
        needs_reshape = False
        for i in range(0, len(idx_shape)):
            if(len(value_shape) <= i or value_shape[i] != idx_shape[i]):
                if(idx_shape[i] != 1):
                    raise ValueError
                else:
                    value_shape.insert(i, 1)
                    # We only need to reshape if we are insert
                    # a dimension after any dimension of length > 1
                    if(past_one_dims):
                        needs_reshape = True
            elif(value_shape[i] != 1):
                past_one_dims = True
        
        if(len(idx_shape) != len(value_shape)):
            raise ValueError        

        if(needs_reshape):
            return value.reshape(value_shape)
        else:
            return value
        
    def __setitem__(self, idx, value):
        if(self.d_array is None):
            raise IndexError('too many indices for array')
        idx = self.__convert_dim__(idx)
        if None in idx:
            # one of the indices is empty
            return
            
        if(isinstance(value, ndarray)):
            if(value.dtype != self.dtype):
                raise TypeError('left hand side must have same dtype as right hand side')
            if(isinstance(idx,list)):
                # There must be a better way to do this!
                value = self.__expand_dim__(value, idx)
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
        return afnumpy.reshape(self, shape, order)

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

#    def __getattr__(self,name):
#        print name
#        raise AttributeError
        

    


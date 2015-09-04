import arrayfire
import sys
from IPython.core.debugger import Tracer
import afnumpy
import private_utils as pu
import numbers
import numpy
import math

def __slice_len__(idx, shape, axis):
    maxlen = shape[axis]
    if idx.step is None:
        step = 1
    else:
        step = idx.step
    if idx.start is None:
        if step < 0:
            start = maxlen
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
            end = maxlen
    else:
        end = idx.stop
        if(end < 0 and step > 0):
            end += maxlen
    if(start == end):
        return 0

    if((start-end > 0 and step > 0) or
       (start-end < 0 and step < 0)):
        return 0
    return int(math.ceil(float(end-start)/step))

def __slice_to_seq__(shape, idx, axis):
    maxlen = shape[axis]
    if(isinstance(idx, numbers.Number)):
        if idx < 0:
            idx = maxlen + idx
        if(idx >= maxlen):
            raise IndexError('index %d is out of bounds for axis %d with size %d' % (idx, axis, maxlen))
        return idx        

    if(isinstance(idx, afnumpy.ndarray)):
        return idx.d_array

    if not isinstance(idx, slice):
        return afnumpy.array(idx).d_array

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
    return  arrayfire.seq(float(start),
                                  float(end),
                                  float(step))
    
def __npidx_to_afidx__(idx, dim_len):
    if(isinstance(idx, numbers.Number)):
        return idx
    if(isinstance(idx, slice)):
        start = idx.start
        stop = idx.stop
        step = idx.step
        if(start is not None and start < 0):
            start += dim_len
        if(stop is not None and stop < 0):
            stop += dim_len
        if idx.step is not None and idx.step < 0:
            if idx.start is None:
                start = dim_len-1
            if idx.stop is None:
                stop = -1
        ret = slice(start,stop,step)
        if  __slice_len__(ret, [dim_len], 0) <= 0:
            return None
        return ret
    if(isinstance(idx, afnumpy.ndarray)):
        return idx.d_array
    return afnumpy.array(idx).d_array
        

def __convert_dim__(shape, idx):
    # Convert numpy style indexing arguments to arrayfire style
    # Always returns a list
    # Should also return the shape of the result

    # If it's an array just return the array
    if(isinstance(idx, afnumpy.ndarray)):
        return [idx.d_array], idx.shape
    # Otherwise turns thing into a tuple
    if not isinstance(idx, tuple):
        idx = (idx,)
    idx = list(idx)

    # According to http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    # newaxis is an alias for 'None', and 'None' can be used in place of this with the same result.
    newaxis = None
    # Check for Ellipsis. Expand it to ':' such that idx shape matches array shape, ignoring any newaxise
    
    # We have to do this because we don't want to trigger comparisons
    if any(e is Ellipsis for e in idx):
        for axis in range(0, len(idx)):
            if(idx[axis] is Ellipsis):
                i = axis
                break
        idx.pop(i)
        if any(e is Ellipsis for e in idx):
            raise IndexError('Only a single Ellipsis allowed')
        while len(idx)-idx.count(newaxis) < len(shape):
            idx.insert(i, slice(None,None,None))

    # Check and remove newaxis. Store their location for final reshape
    newaxes = []
    while newaxis in idx:
        newaxes.append(idx.index(newaxis))
        idx.remove(newaxis)

    # Append enough ':' to match the dimension of the aray
    while len(idx) < len(shape):
        idx.append(slice(None,None,None))

    ret = [0]*len(shape)
    for axis in range(0,len(shape)):
        af_idx = __npidx_to_afidx__(idx[axis], shape[axis])
        ret[pu.c2f(shape,axis)] = af_idx

    ret_shape = __index_shape__(shape, ret)

    # Insert new dimensions start from the end so we don't perturb other insertions
    for n in newaxes[::-1]:
        ret_shape.insert(n,1)
    return ret, tuple(ret_shape)

def __index_shape__(A_shape, idx, del_singleton=True):
    shape = []
    for i in range(0,len(idx)):
        if(idx[i] is None):
            shape.append(0)
        elif(isinstance(idx[i],numbers.Number)):
            if del_singleton:
                # Remove dimensions indexed with a scalar
                continue
            else:
                shape.append(1)
        elif(isinstance(idx[i],arrayfire.index.Seq)):
            if(idx[i].s == arrayfire.af_span):
                shape.append(A_shape[i])
            else:
                shape.append(idx[i].size)
        elif(isinstance(idx[i],slice)):
            shape.append(__slice_len__(idx[i], pu.c2f(A_shape), i))
        elif(isinstance(idx[i], arrayfire.Array)):
            shape.append(idx[i].elements())
        elif(isinstance(idx[i],arrayfire.index)):
            if(idx[i].isspan()):
                shape.append(A_shape[i])
            else:
                af_idx = idx[i].get()
                if(af_idx.isBatch):
                    raise ValueError
                if(af_idx.isSeq):
                    shape.append(arrayfire.seq(af_idx.seq()).size)
                else:
                    shape.append(af_idx.arr_elements())
        else:
            raise ValueError
    return pu.c2f(shape)

def __expand_dim__(shape, value, idx):
    # reshape value, adding size 1 dimensions, such that the dimensions of value match idx
#    Tracer()()
    idx_shape = __index_shape__(shape, idx, False)
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

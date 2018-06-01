import arrayfire
import sys
import afnumpy
from . import private_utils as pu
import numbers
import numpy
import math

def __slice_len__(idx, shape, axis):
    maxlen = shape[axis]
    # Out of bounds slices should be converted to None
    if(idx.stop is not None and idx.stop >= maxlen):
        idx.stop = None
    if(idx.start is not None and idx.start >= maxlen):
        idx.start = None

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
            end = -1
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
        # Out of bounds slices should be converted to None
        if(stop is not None and stop >= dim_len):
            stop = None
        if(start is not None and start >= dim_len):
            start = None

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

    if(not isinstance(idx, afnumpy.ndarray)):
        idx = afnumpy.array(idx)

    if(afnumpy.safe_indexing):
        # Check if we're going out of bounds
        max_index = afnumpy.arrayfire.max(idx.d_array)
        min_index = afnumpy.arrayfire.min(idx.d_array)
        if max_index >= dim_len:
            raise IndexError('index %d is out of bounds for axis with size %d' % (max_index, dim_len))
        if min_index < 0:
            # Transform negative indices in positive ones
            idx.d_array[idx.d_array < 0] += dim_len
    return idx.d_array


def __convert_dim__(shape, idx):
    # Convert numpy style indexing arguments to arrayfire style
    # Always returns a list
    # Should also return the shape of the result
    # Also returns the shape that the input should be reshaped to
    input_shape = list(shape)

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
        while __idx_ndims__(idx)-idx.count(newaxis) < len(shape):
            idx.insert(i, slice(None,None,None))

    # Check and remove newaxis. Store their location for final reshape
    newaxes = []
    while newaxis in idx:
        newaxes.append(idx.index(newaxis))
        idx.remove(newaxis)

    # Append enough ':' to match the dimension of the aray
    while __idx_ndims__(idx) < len(shape):
        idx.append(slice(None,None,None))

    # ret = [0]*len(idx)
    ret = []

    # Check for the number of ndarrays. Raise error if there are multiple
    arrays_in_idx = []
    for axis in range(0,len(idx)):
        if isinstance(idx[axis], afnumpy.ndarray):
            arrays_in_idx.append(axis)
        if isinstance(idx[axis], numpy.ndarray):
            idx[axis] = afnumpy.array(idx[axis])
            arrays_in_idx.append(axis)
    if len(arrays_in_idx) > 1:
        # This will fail because while multiple arrays
        # as indices in numpy treat the values given by
        # the arrays as the coordinates of the hyperslabs
        # to keep, arrayfire does things differently.
        # In arrayfire each entry of each array gets combined
        # with all entries of all other arrays to define the coordinate
        # In numpy each entry only gets combined with the corresponding
        # entry in the other arrays.
        # For example if one has [0,1],[0,1] as the two arrays for numpy
        # this would mean that the coordinates retrieved would be [0,0],
        # [1,1] while for arrayfire it would be [0,0], [0,1], [1,0], [1,1].
        raise NotImplementedError('Fancy indexing with multiple arrays is not implemented')
        # bcast_arrays = afnumpy.broadcast_arrays(*[idx[axis] for axis in arrays_in_idx])
        # for axis,bcast_array in zip(arrays_in_idx, bcast_arrays):
        #     idx[axis] = bcast_array

    for axis in range(0,len(idx)):
        # Handle boolean arrays indexes which require a reshape
        # of the input array
        if(isinstance(idx[axis], afnumpy.ndarray) and
           idx[axis].ndim > 1):
            # Flatten the extra dimensions
            extra_dims = 1
            for i in range(1,idx[axis].ndim):
                extra_dims *= input_shape.pop(axis+1)
            input_shape[axis] *= extra_dims
        af_idx = __npidx_to_afidx__(idx[axis], shape[axis])
        ret.insert(0,af_idx)
#        ret[pu.c2f(shape,axis)] = af_idx

    ret_shape = __index_shape__(shape, ret)

    # Insert new dimensions start from the end so we don't perturb other insertions
    for n in newaxes[::-1]:
        ret_shape.insert(n,1)
    return ret, tuple(ret_shape), tuple(input_shape)

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
            if idx[i].dtype() is arrayfire.Dtype.b8:
                shape.append(int(arrayfire.sum(idx[i])))
            else:
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

def __idx_ndims__(idx):
    ndims = 0
    for i in range(0,len(idx)):
        if isinstance(idx[i], afnumpy.ndarray):
            ndims += idx[i].ndim
        else:
            ndims += 1
    return ndims


def __expand_dim__(shape, value, idx):
    # reshape value, adding size 1 dimensions, such that the dimensions of value match idx
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

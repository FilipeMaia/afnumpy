import numpy as np
import afnumpy as afnp
import arrayfire
from .. import private_utils as pu
from ..decorators import *

def _slice_at_axis(shape, sl, axis):
    """
    Construct a slice tuple the length of shape, with sl at the specified axis
    """
    slice_tup = (slice(None),)
    return slice_tup * axis + (sl,) + slice_tup * (len(shape) - axis - 1)


def _slice_first(shape, n, axis):
    """ Construct a slice tuple to take the first n elements along axis """
    return _slice_at_axis(shape, slice(0, n), axis=axis)


def _slice_last(shape, n, axis):
    """ Construct a slice tuple to take the last n elements along axis """
    dim = shape[axis]  # doing this explicitly makes n=0 work
    return _slice_at_axis(shape, slice(dim - n, dim), axis=axis)

def _do_prepend(arr, pad_chunk, axis):
    return afnp.concatenate(
        (pad_chunk.astype(arr.dtype, copy=False), arr), axis=axis)

def _do_append(arr, pad_chunk, axis):
    return afnp.concatenate(
        (arr, pad_chunk.astype(arr.dtype, copy=False)), axis=axis)

def _prepend_const(arr, pad_amt, val, axis=-1):
    """
    Prepend constant `val` along `axis` of `arr`.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape.
    pad_amt : int
        Amount of padding to prepend.
    val : scalar
        Constant value to use. For best results should be of type `arr.dtype`;
        if not `arr.dtype` will be cast to `arr.dtype`.
    axis : int
        Axis along which to pad `arr`.

    Returns
    -------
    padarr : ndarray
        Output array, with `pad_amt` constant `val` prepended along `axis`.

    """
    if pad_amt == 0:
        return arr
    padshape = tuple(x if i != axis else pad_amt
                     for (i, x) in enumerate(arr.shape))
    return _do_prepend(arr, afnp.full(padshape, val, dtype=arr.dtype), axis)


def _append_const(arr, pad_amt, val, axis=-1):
    """
    Append constant `val` along `axis` of `arr`.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape.
    pad_amt : int
        Amount of padding to append.
    val : scalar
        Constant value to use. For best results should be of type `arr.dtype`;
        if not `arr.dtype` will be cast to `arr.dtype`.
    axis : int
        Axis along which to pad `arr`.

    Returns
    -------
    padarr : ndarray
        Output array, with `pad_amt` constant `val` appended along `axis`.

    """
    if pad_amt == 0:
        return arr
    padshape = tuple(x if i != axis else pad_amt
                     for (i, x) in enumerate(arr.shape))
    return _do_append(arr, afnp.full(padshape, val, dtype=arr.dtype), axis)


def _prepend_edge(arr, pad_amt, axis=-1):
    """
    Prepend `pad_amt` to `arr` along `axis` by extending edge values.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape.
    pad_amt : int
        Amount of padding to prepend.
    axis : int
        Axis along which to pad `arr`.

    Returns
    -------
    padarr : ndarray
        Output array, extended by `pad_amt` edge values appended along `axis`.

    """
    if pad_amt == 0:
        return arr

    edge_slice = _slice_first(arr.shape, 1, axis=axis)
    edge_arr = arr[edge_slice]
    return _do_prepend(arr, edge_arr.repeat(pad_amt, axis=axis), axis)


def _append_edge(arr, pad_amt, axis=-1):
    """
    Append `pad_amt` to `arr` along `axis` by extending edge values.

    Parameters
    ----------
    arr : ndarray
        Input array of arbitrary shape.
    pad_amt : int
        Amount of padding to append.
    axis : int
        Axis along which to pad `arr`.

    Returns
    -------
    padarr : ndarray
        Output array, extended by `pad_amt` edge values prepended along
        `axis`.

    """
    if pad_amt == 0:
        return arr

    edge_slice = _slice_last(arr.shape, 1, axis=axis)
    edge_arr = arr[edge_slice]
    return _do_append(arr, edge_arr.repeat(pad_amt, axis=axis), axis)


def _normalize_shape(ndarray, shape, cast_to_int=True):
    """
    Private function which does some checks and normalizes the possibly
    much simpler representations of 'pad_width', 'stat_length',
    'constant_values', 'end_values'.

    Parameters
    ----------
    narray : ndarray
        Input ndarray
    shape : {sequence, array_like, float, int}, optional
        The width of padding (pad_width), the number of elements on the
        edge of the narray used for statistics (stat_length), the constant
        value(s) to use when filling padded regions (constant_values), or the
        endpoint target(s) for linear ramps (end_values).
        ((before_1, after_1), ... (before_N, after_N)) unique number of
        elements for each axis where `N` is rank of `narray`.
        ((before, after),) yields same before and after constants for each
        axis.
        (constant,) or val is a shortcut for before = after = constant for
        all axes.
    cast_to_int : bool, optional
        Controls if values in ``shape`` will be rounded and cast to int
        before being returned.

    Returns
    -------
    normalized_shape : tuple of tuples
        val                               => ((val, val), (val, val), ...)
        [[val1, val2], [val3, val4], ...] => ((val1, val2), (val3, val4), ...)
        ((val1, val2), (val3, val4), ...) => no change
        [[val1, val2], ]                  => ((val1, val2), (val1, val2), ...)
        ((val1, val2), )                  => ((val1, val2), (val1, val2), ...)
        [[val ,     ], ]                  => ((val, val), (val, val), ...)
        ((val ,     ), )                  => ((val, val), (val, val), ...)

    """
    ndims = ndarray.ndim

    # Shortcut shape=None
    if shape is None:
        return ((None, None), ) * ndims

    # Convert any input `info` to a NumPy array
    shape_arr = np.asarray(shape)

    try:
        shape_arr = np.broadcast_to(shape_arr, (ndims, 2))
    except ValueError:
        fmt = "Unable to create correctly shaped tuple from %s"
        raise ValueError(fmt % (shape,))

    # Cast if necessary
    if cast_to_int is True:
        shape_arr = np.round(shape_arr).astype(int)

    # Convert list of lists to tuple of tuples
    return tuple(tuple(axis) for axis in shape_arr.tolist())

def _validate_lengths(narray, number_elements):
    """
    Private function which does some checks and reformats pad_width and
    stat_length using _normalize_shape.

    Parameters
    ----------
    narray : ndarray
        Input ndarray
    number_elements : {sequence, int}, optional
        The width of padding (pad_width) or the number of elements on the edge
        of the narray used for statistics (stat_length).
        ((before_1, after_1), ... (before_N, after_N)) unique number of
        elements for each axis.
        ((before, after),) yields same before and after constants for each
        axis.
        (constant,) or int is a shortcut for before = after = constant for all
        axes.

    Returns
    -------
    _validate_lengths : tuple of tuples
        int                               => ((int, int), (int, int), ...)
        [[int1, int2], [int3, int4], ...] => ((int1, int2), (int3, int4), ...)
        ((int1, int2), (int3, int4), ...) => no change
        [[int1, int2], ]                  => ((int1, int2), (int1, int2), ...)
        ((int1, int2), )                  => ((int1, int2), (int1, int2), ...)
        [[int ,     ], ]                  => ((int, int), (int, int), ...)
        ((int ,     ), )                  => ((int, int), (int, int), ...)

    """
    normshp = _normalize_shape(narray, number_elements)
    for i in normshp:
        chk = [1 if x is None else x for x in i]
        chk = [1 if x >= 0 else -1 for x in chk]
        if (chk[0] < 0) or (chk[1] < 0):
            fmt = "%s cannot contain negative values."
            raise ValueError(fmt % (number_elements,))
    return normshp

def pad(array, pad_width, mode, **kwargs):
    if not np.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')

    narray = afnp.array(array)
    pad_width = _validate_lengths(narray, pad_width)

    allowedkwargs = {
        'constant': ['constant_values'],
        'edge': [],
        'linear_ramp': ['end_values'],
        'maximum': ['stat_length'],
        'mean': ['stat_length'],
        'median': ['stat_length'],
        'minimum': ['stat_length'],
        'reflect': ['reflect_type'],
        'symmetric': ['reflect_type'],
        'wrap': [],
        }

    kwdefaults = {
        'stat_length': None,
        'constant_values': 0,
        'end_values': 0,
        'reflect_type': 'even',
        }

    if isinstance(mode, np.compat.basestring):
        # Make sure have allowed kwargs appropriate for mode
        for key in kwargs:
            if key not in allowedkwargs[mode]:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                 (key, allowedkwargs[mode]))

        # Set kwarg defaults
        for kw in allowedkwargs[mode]:
            kwargs.setdefault(kw, kwdefaults[kw])

        # Need to only normalize particular keywords.
        for i in kwargs:
            if i == 'stat_length':
                kwargs[i] = _validate_lengths(narray, kwargs[i])
            if i in ['end_values', 'constant_values']:
                kwargs[i] = _normalize_shape(narray, kwargs[i],
                                             cast_to_int=False)
    else:
        return afnumpy.array(np.pad(np.array(array),pad_width,mode,**kwargs))

    # If we get here, use new padding method
    newmat = narray.copy()

    # API preserved, but completely new algorithm which pads by building the
    # entire block to pad before/after `arr` with in one step, for each axis.

    if mode == 'constant':
        for axis, ((pad_before, pad_after), (before_val, after_val)) \
                in enumerate(zip(pad_width, kwargs['constant_values'])):
            newmat = _prepend_const(newmat, pad_before, before_val, axis)
            newmat = _append_const(newmat, pad_after, after_val, axis)
    else:
        # For the other modes we'll go with the slow mode
        return afnumpy.array(np.pad(np.array(array),pad_width,mode,**kwargs))

    return newmat

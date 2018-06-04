import afnumpy
import numpy
import collections

def copy(a, order='K'):
    return afnumpy.array(a, order=order, copy=True)

def meshgrid(*xi, **kwargs):
    ndim = len(xi)

    copy_ = kwargs.pop('copy', True)
    sparse = kwargs.pop('sparse', False)
    indexing = kwargs.pop('indexing', 'xy')

    if kwargs:
        raise TypeError("meshgrid() got an unexpected keyword argument '%s'"
                        % (list(kwargs)[0],))

    if indexing not in ['xy', 'ij']:
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim

    output = [afnumpy.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1::])
              for i, x in enumerate(xi)]

    shape = [int(x.size) for x in output]

    if indexing == 'xy' and ndim > 1:
        # switch first and second axis
        output[0].shape = (1, -1) + (1,)*(ndim - 2)
        output[1].shape = (-1, 1) + (1,)*(ndim - 2)
        shape[0], shape[1] = shape[1], shape[0]

    if sparse:
        if copy_:
            return [x.copy() for x in output]
        else:
            return output
    else:
        # Return the full N-D matrix (not only the 1-D vector)
        if copy_:
            # Numpy uses dtype=int but Arrayfire does not support int64 in all functions
            mult_fact = afnumpy.ones(shape, dtype=numpy.int32)
            return [x * mult_fact for x in output]
        else:
            return afnumpy.broadcast_arrays(*output)

def angle(z, deg=0):
    """
    Return the angle of the complex argument.

    Parameters
    ----------
    z : array_like
        A complex number or sequence of complex numbers.
    deg : bool, optional
        Return angle in degrees if True, radians if False (default).

    Returns
    -------
    angle : {ndarray, scalar}
        The counterclockwise angle from the positive real axis on
        the complex plane, with dtype as numpy.float64.

    See Also
    --------
    arctan2
    absolute



    Examples
    --------
    >>> np.angle([1.0, 1.0j, 1+1j])               # in radians
    array([ 0.        ,  1.57079633,  0.78539816])
    >>> np.angle(1+1j, deg=True)                  # in degrees
    45.0

    """
    if deg:
        fact = 180/pi
    else:
        fact = 1.0
    z = afnumpy.asarray(z)
    if numpy.issubdtype(z.dtype, numpy.complexfloating):
        zimag = z.imag
        zreal = z.real
    else:
        zimag = 0
        zreal = z
    return afnumpy.arctan2(zimag, zreal) * fact

def percentile(a, q, axis=None, out=None,
               overwrite_input=False, interpolation='linear', keepdims=False):
    if interpolation is not 'linear':
        raise ValueError('Only linear interpolation is supported')
    if out is not None:
        raise ValueError('"out" parameter is not supported')
    if isinstance(axis, collections.Sequence):
        raise ValueError('axis sequences not supported')
        
    if axis is None:
        input = a.flatten()
        axis = 0
    else: 
        input = a
    s = afnumpy.sort(input,axis=axis)
    low_idx = numpy.floor(q*(s.shape[axis]-1)/100.0)
    high_idx = numpy.ceil(q*(s.shape[axis]-1)/100.0)
    u = (q*(s.shape[axis]-1)/100.0) - low_idx
    low = afnumpy.take(s, low_idx, axis=axis)
    high = afnumpy.take(s, high_idx, axis=axis)
    ret = u*high+(1-u)*low
    if keepdims is True:
        ret.reshape(ret.shape[:axis]+(1,)+ret.shape[axis:])
    return ret
    
        

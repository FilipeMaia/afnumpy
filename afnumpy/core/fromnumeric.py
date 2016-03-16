import numpy
import afnumpy
import arrayfire
from .. import private_utils as pu
from ..decorators import *

def all(a, axis=None, out=None, keepdims=False):
    try:
        return a.all(axis, out, keepdims)
    except AttributeError:
        return numpy.all(a, axis, out, keepdims)

def any(a, axis=None, out=None, keepdims=False):
    try:
        return a.any(axis, out, keepdims)
    except AttributeError:
        return numpy.any(a, axis, out, keepdims)

def round(a, decimals=0, out=None):
    try:
        return a.round(decimals, out)
    except AttributeError:
        return numpy.round(a, decimals, out)

def take(a, indices, axis=None, out=None, mode='raise'):
    try:
        return a.take(indices, axis, out, mode)
    except AttributeError:
        return numpy.take(a, indices, axis, out, mode)

def amin(a, axis=None, out=None, keepdims=False):
    try:
        return a.min(axis, out, keepdims)
    except AttributeError:
        return numpy.amin(a, axis, out, keepdims)

min = amin

def amax(a, axis=None, out=None, keepdims=False):
    try:
        return a.max(axis, out, keepdims)
    except AttributeError:
        return numpy.amax(a, axis, out, keepdims)

max = amax

def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    try:
        return a.prod(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    except AttributeError:
        return numpy.prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    try:
        return a.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    except AttributeError:
        return numpy.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    try:
        return a.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    except AttributeError:
        return numpy.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

@outufunc
def sqrt(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.sqrt(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.sqrt(x)

def transpose(a, axes=None):
    try:
        return a.transpose(axes)
    except AttributeError:
        return numpy.tranpose(a, axes)

def squeeze(a, axis=None):
    try:
        return a.squeeze(axis)
    except AttributeError:
        return numpy.squeeze(a, axis)

def argmax(a, axis=None):
    try:
        return a.argmax(axis)
    except AttributeError:
        return numpy.argmax(a, axis)

def argmin(a, axis=None):
    try:
        return a.argmin(axis)
    except AttributeError:
        return numpy.argmin(a, axis)

def argsort(a, axis=-1, kind='quicksort', order=None):
    try:
        return a.argsort(axis, kind, order)
    except AttributeError:
        return numpy.argsort(a, axis, kind, order)

def ravel(a, order='C'):
    """
    Return a flattened array.

    A 1-D array, containing the elements of the input, is returned.  A copy is
    made only if needed.

    Parameters
    ----------
    a : array_like
        Input array.  The elements in `a` are read in the order specified by
        `order`, and packed as a 1-D array.
    order : {'C','F', 'A', 'K'}, optional
        The elements of `a` are read using this index order. 'C' means to
        index the elements in C-like order, with the last axis index changing
        fastest, back to the first axis index changing slowest.   'F' means to
        index the elements in Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that the 'C'
        and 'F' options take no account of the memory layout of the underlying
        array, and only refer to the order of axis indexing.  'A' means to read
        the elements in Fortran-like index order if `a` is Fortran *contiguous*
        in memory, C-like order otherwise.  'K' means to read the elements in
        the order they occur in memory, except for reversing the data when
        strides are negative.  By default, 'C' index order is used.

    Returns
    -------
    1d_array : ndarray
        Output of the same dtype as `a`, and of shape ``(a.size,)``.

    See Also
    --------
    ndarray.flat : 1-D iterator over an array.
    ndarray.flatten : 1-D array copy of the elements of an array
                      in row-major order.

    Notes
    -----
    In C-like (row-major) order, in two dimensions, the row index varies the
    slowest, and the column index the quickest.  This can be generalized to
    multiple dimensions, where row-major order implies that the index along the
    first axis varies slowest, and the index along the last quickest.  The
    opposite holds for Fortran-like, or column-major, index ordering.

    Examples
    --------
    It is equivalent to ``reshape(-1, order=order)``.

    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> print np.ravel(x)
    [1 2 3 4 5 6]

    >>> print x.reshape(-1)
    [1 2 3 4 5 6]

    >>> print np.ravel(x, order='F')
    [1 4 2 5 3 6]

    When ``order`` is 'A', it will preserve the array's 'C' or 'F' ordering:

    >>> print np.ravel(x.T)
    [1 4 2 5 3 6]
    >>> print np.ravel(x.T, order='A')
    [1 2 3 4 5 6]

    When ``order`` is 'K', it will preserve orderings that are neither 'C'
    nor 'F', but won't reverse axes:

    >>> a = np.arange(3)[::-1]; a
    array([2, 1, 0])
    >>> a.ravel(order='C')
    array([2, 1, 0])
    >>> a.ravel(order='K')
    array([2, 1, 0])

    >>> a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a
    array([[[ 0,  2,  4],
            [ 1,  3,  5]],
           [[ 6,  8, 10],
            [ 7,  9, 11]]])
    >>> a.ravel(order='C')
    array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])
    >>> a.ravel(order='K')
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

    """
    return afnumpy.asarray(a).ravel(order)

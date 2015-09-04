import afnumpy
import arrayfire
from numpy.core import complexfloating, Inf, longdouble
from afnumpy import asarray, sqrt, abs
from afnumpy.lib import asfarray

def isComplexType(t):
    return issubclass(t, complexfloating)

def vdot(a, b):
    s = arrayfire.dot(arrayfire.conjg(a.flat.d_array), b.flat.d_array)
    return afnumpy.ndarray((), dtype=a.dtype, af_array=s)[()]

def dot(a, b):
    s = arrayfire.dot((a.flat.d_array), b.flat.d_array)
    return afnumpy.ndarray((), dtype=a.dtype, af_array=s)[()]

def norm(x, ord=None, axis=None, keepdims=False):
    x = asarray(x)

    # Check the default case first and handle it immediately.
    if ord is None and axis is None:
        ndim = x.ndim
        x = x.ravel(order='K')
        if isComplexType(x.dtype.type):
            sqnorm = dot(x.real, x.real) + dot(x.imag, x.imag)
        else:
            sqnorm = dot(x, x)
        ret = sqrt(sqnorm)
        if keepdims:
            ret = ret.reshape(ndim*[1])
        return ret

    # Normalize the `axis` argument to a tuple.
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except:
            raise TypeError("'axis' must be None, an integer or a tuple of integers")
        axis = (axis,)

    if len(axis) == 1:
        if ord == Inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -Inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            # Zero norm
            return (x != 0).sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            # special case for speedup
            return afnumpy.sum(abs(x), axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            # special case for speedup
            s = (x.conj() * x).real
            return sqrt(afnumpy.sum(s, axis=axis, keepdims=keepdims))
        else:
            try:
                ord + 1
            except TypeError:
                raise ValueError("Invalid norm order for vectors.")
            if x.dtype.type is longdouble:
                # Convert to a float type, so integer arrays give
                # float results.  Don't apply asfarray to longdouble arrays,
                # because it will downcast to float64.
                absx = abs(x)
            else:
                absx = x if isComplexType(x.dtype.type) else asfarray(x)
                if absx.dtype is x.dtype:
                    absx = abs(absx)
                else:
                    # if the type changed, we can safely overwrite absx
                    abs(absx, out=absx)
            absx **= ord
            return afnumpy.sum(absx, axis=axis, keepdims=keepdims) ** (1.0 / ord)
    elif len(axis) == 2:
        row_axis, col_axis = axis
        if not (-nd <= row_axis < nd and -nd <= col_axis < nd):
            raise ValueError('Invalid axis %r for an array with shape %r' %
                             (axis, x.shape))
        if row_axis % nd == col_axis % nd:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            ret =  _multi_svd_norm(x, row_axis, col_axis, amax)
        elif ord == -2:
            ret = _multi_svd_norm(x, row_axis, col_axis, amin)
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = afnumpy.sum(abs(x), axis=row_axis).max(axis=col_axis)
        elif ord == Inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = afnumpy.sum(abs(x), axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = afnumpy.sum(abs(x), axis=row_axis).min(axis=col_axis)
        elif ord == -Inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = afnumpy.sum(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            ret = sqrt(afnumpy.sum((x.conj() * x).real, axis=axis))
        else:
            raise ValueError("Invalid norm order for matrices.")
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")
    

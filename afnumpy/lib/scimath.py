import numpy
import afnumpy
import arrayfire
from afnumpy import private_utils as pu
from afnumpy.decorators import *

@outufunc
def arccos(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.acos(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.arccos(x)

@outufunc
def arcsin(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.asin(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.arcsin(x)

@outufunc
def arctan(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.atan(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.arctan(x)

@outufunc
def arctan2(x1, x2):
    if isinstance(x1, afnumpy.ndarray) and isinstance(x2, afnumpy.ndarray):
        s = arrayfire.atan2(x1.d_array, x2.d_array)
        return afnumpy.ndarray(x1.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.arctan(x1, x2)

@outufunc
def arccosh(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.acosh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.arccosh(x)

@outufunc
def arcsinh(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.asinh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.arcsinh(x)

@outufunc
def arctanh(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.atanh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.arctanh(x)


@outufunc
def cos(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.cos(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.cos(x)

@outufunc
def sin(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.sin(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.sin(x)

@outufunc
def tan(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.tan(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.tan(x)

@outufunc
def cosh(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.cosh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.cosh(x)

@outufunc
def sinh(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.sinh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.sinh(x)

@outufunc
def tanh(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.tanh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.tanh(x)

@outufunc
def exp(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.exp(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.exp(x)

@outufunc
def log(x):
    if isinstance(x, afnumpy.ndarray):
        s = arrayfire.log(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.dtype()), af_array=s)
    else:
        return numpy.log(x)


@outufunc
def multiply(x1, x2):
    return x1*x2

@outufunc
def subtract(x1, x2):
    return x1-x2

@outufunc
def add(x1, x2):
    return x1+x2

@outufunc
def divide(x1, x2):
    return x1/x2

        
        
inf = numpy.inf
Inf = numpy.Inf
Infinity = numpy.Infinity
pi = numpy.pi
e = numpy.e


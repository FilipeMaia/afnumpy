import numpy
import afnumpy
from afnumpy import private_utils as pu
from afnumpy.decorators import *

@outufunc
def arccos(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.acos(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.arccos(x)

@outufunc
def arcsin(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.asin(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.arcsin(x)

@outufunc
def arctan(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.atan(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.arctan(x)

@outufunc
def arccosh(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.acosh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.arccosh(x)

@outufunc
def arcsinh(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.asinh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.arcsinh(x)

@outufunc
def arctanh(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.atanh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.arctanh(x)


@outufunc
def cos(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.cos(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.cos(x)

@outufunc
def sin(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.sin(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.sin(x)

@outufunc
def tan(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.tan(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.tan(x)

@outufunc
def cosh(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.cosh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.cosh(x)

@outufunc
def sinh(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.sinh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.sinh(x)

@outufunc
def tanh(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.tanh(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.tanh(x)

@outufunc
def exp(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.exp(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.exp(x)

@outufunc
def log(x):
    if isinstance(x, afnumpy.ndarray):
        s = afnumpy.arrayfire.log(x.d_array)
        return afnumpy.ndarray(x.shape, dtype=pu.typemap(s.type()), af_array=s)
    else:
        return numpy.log(x)

        
        
    

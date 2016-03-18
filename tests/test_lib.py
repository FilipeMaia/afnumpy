import afnumpy
import numpy
from asserts import *
import sys

def test_copy():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    c = afnumpy.copy(a)
    d = numpy.copy(b)
    a[:] = 0
    b[:] = 0
    iassert(c,d)

def test_meshgrid():
    nx, ny = (3, 2)
    x2 = numpy.linspace(0, 1, nx)
    y2 = numpy.linspace(0, 1, ny)
    x1 = afnumpy.array(x2)
    y1 = afnumpy.array(y2)

    iassert(afnumpy.meshgrid(x1, y1), numpy.meshgrid(x2, y2))

def test_broadcast_arrays():
    # Currently arrayfire is missing support for int64
    x2 = numpy.array([[1,2,3]], dtype=numpy.float32)
    y2 = numpy.array([[1],[2],[3]], dtype=numpy.float32)
    x1 = afnumpy.array(x2)
    y1 = afnumpy.array(y2)
    iassert(afnumpy.broadcast_arrays(x1, y1), numpy.broadcast_arrays(x2, y2))
    x1 = afnumpy.array([2])
    y1 = afnumpy.array(2)
    x2 = numpy.array([2])
    y2 = numpy.array(2)
    iassert(afnumpy.broadcast_arrays(x1, y1), numpy.broadcast_arrays(x2, y2))

def test_tile():
    # Currently arrayfire is missing support for int64
    b = numpy.array([0, 1, 2], dtype=numpy.float32)
    a = afnumpy.array(b)
    iassert(afnumpy.tile(a, 2), numpy.tile(b, 2))
    iassert(afnumpy.tile(a, (2,2)), numpy.tile(b, (2,2)))
    iassert(afnumpy.tile(a, (2,1,2)), numpy.tile(b, (2,1,2)))

def test_arccos():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.arccos(a), numpy.arccos(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.arccos(a, out=c), numpy.arccos(b, out=d))
    fassert(c, d)

def test_arcsin():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.arcsin(a), numpy.arcsin(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.arcsin(a, out=c), numpy.arcsin(b, out=d))
    fassert(c, d)

def test_arctan():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.arctan(a), numpy.arctan(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.arctan(a, out=c), numpy.arctan(b, out=d))
    fassert(c, d)

def test_arctan2():
    a1 = afnumpy.random.random((2,3))
    b1 = numpy.array(a1)
    a2 = afnumpy.random.random((2,3))
    b2 = numpy.array(a2)
    fassert(afnumpy.arctan2(a1,a2), numpy.arctan2(b1,b2))
    c = afnumpy.random.random((2,3))
    d = numpy.array(c)
    fassert(afnumpy.arctan2(a1,a2, out=c), numpy.arctan2(b1, b2, out=d))
    fassert(c, d)

def test_arccosh():
    # Domain for arccosh starts at 1
    a = afnumpy.random.random((2,3))+1
    b = numpy.array(a)
    fassert(afnumpy.arccosh(a), numpy.arccosh(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.arccosh(a, out=c), numpy.arccosh(b, out=d))
    fassert(c, d)

def test_arcsinh():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.arcsinh(a), numpy.arcsinh(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.arcsinh(a, out=c), numpy.arcsinh(b, out=d))
    fassert(c, d)

def test_arctanh():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.arctanh(a), numpy.arctanh(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.arctanh(a, out=c), numpy.arctanh(b, out=d))
    fassert(c, d)

def test_cos():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.cos(a), numpy.cos(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.cos(a, out=c), numpy.cos(b, out=d))
    fassert(c, d)

def test_sin():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.sin(a), numpy.sin(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.sin(a, out=c), numpy.sin(b, out=d))
    fassert(c, d)

def test_tan():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.tan(a), numpy.tan(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.tan(a, out=c), numpy.tan(b, out=d))
    fassert(c, d)

def test_cosh():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.cosh(a), numpy.cosh(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.cosh(a, out=c), numpy.cosh(b, out=d))
    fassert(c, d)

def test_sinh():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.sinh(a), numpy.sinh(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.sinh(a, out=c), numpy.sinh(b, out=d))
    fassert(c, d)

def test_tanh():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.tanh(a), numpy.tanh(b))
    c = afnumpy.random.random((2,3))
    d = numpy.array(a)
    fassert(afnumpy.tanh(a, out=c), numpy.tanh(b, out=d))
    fassert(c, d)

def test_exp():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.exp(a), numpy.exp(b))

def test_log():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.log(a), numpy.log(b))

def test_real():
    x = numpy.sqrt([1+0j, 0+1j])
    y = afnumpy.array(x)
    fassert(afnumpy.real(y), numpy.real(x))
    y.real[:] = 0
    x.real[:] = 0
    fassert(y, x)

def test_imag():
    x = numpy.sqrt([1+0j, 0+1j])
    y = afnumpy.array(x)
    fassert(afnumpy.imag(y), numpy.imag(x))
    y.real[:] = 0
    x.real[:] = 0
    fassert(y, x)

def test_multiply():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.multiply(a,a), numpy.multiply(b,b))
    a = afnumpy.array(2)
    ao = afnumpy.array(0)
    b = numpy.array(a)
    bo = numpy.array(0)
    fassert(afnumpy.multiply(a,a), numpy.multiply(b,b))
    fassert(afnumpy.multiply(a,a, out=ao), numpy.multiply(b,b, out = bo))
    fassert(ao, bo)

def test_subtract():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.subtract(a,a), numpy.subtract(b,b))
    a = afnumpy.array(2)
    ao = afnumpy.array(0)
    b = numpy.array(a)
    bo = numpy.array(0)
    fassert(afnumpy.subtract(a,a), numpy.subtract(b,b))
    fassert(afnumpy.subtract(a,a, out=ao), numpy.subtract(b,b, out = bo))
    fassert(ao, bo)

def test_add():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.add(a,a), numpy.add(b,b))
    a = afnumpy.array(2)
    ao = afnumpy.array(0)
    b = numpy.array(a)
    bo = numpy.array(0)
    fassert(afnumpy.add(a,a), numpy.add(b,b))
    fassert(afnumpy.add(a,a, out=ao), numpy.add(b,b, out = bo))
    fassert(ao, bo)

def test_divide():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.divide(a,a), numpy.divide(b,b))
    a = afnumpy.array(2)
    b = numpy.array(a)
    if sys.version_info >= (3, 0):
        ao = afnumpy.array(0.)
        bo = numpy.array(0.)
    else:
        ao = afnumpy.array(0)
        bo = numpy.array(0)
    fassert(afnumpy.divide(a,a), numpy.divide(b,b))
    fassert(afnumpy.divide(a,a, out=ao), numpy.divide(b,b, out = bo))
    fassert(ao, bo)


def test_true_divide():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.true_divide(a,a), numpy.true_divide(b,b))
    a = afnumpy.array(2)
    b = numpy.array(a)
    ao = afnumpy.array(0.)
    bo = numpy.array(0.)
    fassert(afnumpy.true_divide(a,a), numpy.true_divide(b,b))
    fassert(afnumpy.true_divide(a,a, out=ao), numpy.true_divide(b,b, out = bo))
    fassert(ao, bo)

def test_floor_divide():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.floor_divide(a,a), numpy.floor_divide(b,b))
    a = afnumpy.array(2)
    b = numpy.array(a)
    ao = afnumpy.array(0)
    bo = numpy.array(0)
    fassert(afnumpy.floor_divide(a,a), numpy.floor_divide(b,b))
    fassert(afnumpy.floor_divide(a,a, out=ao), numpy.floor_divide(b,b, out = bo))
    fassert(ao, bo)

def test_angle():
    a = afnumpy.random.random((2,3))+afnumpy.random.random((2,3))*1.0j
    b = numpy.array(a)
    fassert(afnumpy.angle(a), numpy.angle(b))

def test_conjugate():
    a = afnumpy.random.random((2,3))+afnumpy.random.random((2,3))*1.0j
    b = numpy.array(a)
    fassert(afnumpy.conjugate(a), numpy.conjugate(b))

def test_conj():
    a = afnumpy.random.random((2,3))+afnumpy.random.random((2,3))*1.0j
    b = numpy.array(a)
    fassert(afnumpy.conj(a), numpy.conj(b))

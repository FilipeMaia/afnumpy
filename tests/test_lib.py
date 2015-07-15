import afnumpy
import numpy
from asserts import *

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


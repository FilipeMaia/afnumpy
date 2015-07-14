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
    x2 = numpy.array([[1,2,3]])
    y2 = numpy.array([[1],[2],[3]])
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

import afnumpy
import numpy
from asserts import *

def test_floor():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(afnumpy.floor(a), numpy.floor(b))

    b = numpy.array([1,2,3])
    a = afnumpy.array(b)
    iassert(afnumpy.floor(a), numpy.floor(b))

def test_ceil():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(afnumpy.ceil(a), numpy.ceil(b))

    b = numpy.array([1,2,3])
    a = afnumpy.array(b)
    iassert(afnumpy.ceil(a), numpy.ceil(b))
    
def test_asanyarray():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(afnumpy.asanyarray(a), numpy.asanyarray(b))
    # zero dim arrays not supported
    iassert(afnumpy.asanyarray(1), numpy.asanyarray(1))
    iassert(afnumpy.asanyarray([1,2]), numpy.asanyarray([1,2]))
    iassert(afnumpy.asanyarray(b), numpy.asanyarray(b))

def test_reshape():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(a.reshape((3,2)), b.reshape((3,2)))
    iassert(a.reshape(6), b.reshape(6))
    iassert(a.reshape((-1,2)), b.reshape((-1,2)))

    b = numpy.random.random((1))
    a = afnumpy.array(b)
    # Some empty shape reshape
    iassert(a.reshape(()), b.reshape(()))
    iassert(a.reshape([]), b.reshape([]))

def test_abs():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    fassert(afnumpy.abs(a), numpy.abs(b))

def test_ones():
    a = afnumpy.ones(3)
    b = numpy.ones(3)
    iassert(a, b)

def test_roll():    
    b = numpy.random.random(3)
    a = afnumpy.array(b)
    fassert(afnumpy.roll(a, -1, 0), numpy.roll(b, -1, 0))

    b = numpy.random.random(3)
    a = afnumpy.array(b)
    fassert(afnumpy.roll(a, 1, 0), numpy.roll(b, 1, 0))

    b = numpy.random.random((2, 3))
    a = afnumpy.array(b)
    fassert(afnumpy.roll(a, 1, 0), numpy.roll(b, 1, 0))

    b = numpy.random.random((2, 3))
    a = afnumpy.array(b)
    fassert(afnumpy.roll(a, 1, 1), numpy.roll(b, 1, 1))

    b = numpy.random.random((2, 3))
    a = afnumpy.array(b)
    fassert(afnumpy.roll(a, 2), numpy.roll(b, 2))

def test_concatenate():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(afnumpy.concatenate(a), numpy.concatenate(b))
    iassert(afnumpy.concatenate((a,a)), numpy.concatenate((b,b)))
    iassert(afnumpy.concatenate((a,a),axis=1), numpy.concatenate((b,b),axis=1))

def test_round():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    fassert(afnumpy.round(a), numpy.round(b))

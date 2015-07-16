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

def test_take():
    a = [4, 3, 5, 7, 6, 8]
    indices = [0, 1, 4]
    iassert(afnumpy.take(a, indices), numpy.take(a, indices))
    b = numpy.array(a)
    a = afnumpy.array(a)
    iassert(afnumpy.take(a, indices), numpy.take(b, indices))

def test_ascontiguousarray():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.ascontiguousarray(a), numpy.ascontiguousarray(b))

def test_min():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.min(a), numpy.min(b))
    fassert(afnumpy.min(a,axis=0), numpy.min(b,axis=0))
    fassert(afnumpy.min(a,axis=1), numpy.min(b,axis=1))

def test_max():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.max(a), numpy.max(b))
    fassert(afnumpy.max(a,axis=0), numpy.max(b,axis=0))
    fassert(afnumpy.max(a,axis=1), numpy.max(b,axis=1))

def test_prod():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.prod(a), numpy.prod(b))
    fassert(afnumpy.prod(a,axis=0), numpy.prod(b,axis=0))
    fassert(afnumpy.prod(a,axis=1), numpy.prod(b,axis=1))

def test_mean():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.mean(a), numpy.mean(b))
    fassert(afnumpy.mean(a,axis=0), numpy.mean(b,axis=0))
    fassert(afnumpy.mean(a,axis=1), numpy.mean(b,axis=1))
    fassert(afnumpy.mean(a,axis=(0,1)), numpy.mean(b,axis=(0,1)))


def test_sqrt():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.sqrt(a), numpy.sqrt(b))


def test_dtypes():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(afnumpy.int32(a), numpy.int32(b))
    fassert(afnumpy.complex64(a), numpy.complex64(b))
    assert(afnumpy.float(a.sum()), numpy.float(b.sum()))

def test_transpose():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(a.transpose(), b.transpose())
    iassert(a.transpose((0,1)), b.transpose((0,1)))
    iassert(a.transpose((1,0)), b.transpose((1,0)))
    b = numpy.random.random((2))
    a = afnumpy.array(b)
    iassert(a.transpose(), b.transpose())
    b = numpy.random.random((2,3,4))
    a = afnumpy.array(b)
    iassert(a.transpose(), b.transpose())
    iassert(a.transpose((2,0,1)), b.transpose((2,0,1)))

import afnumpy
import afnumpy as af
import numpy
import numpy as np
from asserts import *
import pytest
xfail = pytest.mark.xfail

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
    assert a.d_array.device_ptr() == a.reshape((-1,2)).d_array.device_ptr()

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
    fassert(afnumpy.complex64(b), numpy.complex64(a))
    assert(type(afnumpy.complex64(b)), afnumpy.multiarray.ndarray)
    assert(type(afnumpy.complex64([1,2,3])), afnumpy.multiarray.ndarray)
    assert(type(afnumpy.bool8(True)), numpy.bool_)
    
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

def test_rollaxis():
    b = numpy.random.random((3,4,5,6))
    a = afnumpy.array(b)
    iassert(afnumpy.rollaxis(a, 3, 1),numpy.rollaxis(b, 3, 1))
    iassert(afnumpy.rollaxis(a, 2),numpy.rollaxis(b, 2))
    iassert(afnumpy.rollaxis(a, 1, 4),numpy.rollaxis(b, 1, 4))

def test_cross():
    x = [1, 2, 3]
    y = [4, 5, 6]
    iassert(afnumpy.cross(x, y), numpy.cross(x, y))

def test_linspace():
    iassert(afnumpy.linspace(0,10), numpy.linspace(0,10))

def test_squeeze():
    x = numpy.array([[[0], [1], [2]]])
    y = afnumpy.array(x)
    iassert(afnumpy.squeeze(y), numpy.squeeze(x))
    iassert(afnumpy.squeeze(y, axis=(2,)), numpy.squeeze(x, axis=(2,)))

def test_all():
    iassert(afnumpy.all([[True, False], [True, True]]),
            numpy.all([[True, False], [True, True]]))
    x = numpy.array([[True, False], [True, True]])
    y = afnumpy.array(x)
    iassert(y.all(axis=0),x.all(axis=0))

def test_any():
    iassert(afnumpy.any([[True, False], [True, True]]),
            numpy.any([[True, False], [True, True]]))
    x = numpy.array([[True, False], [True, True]])
    y = afnumpy.array(x)
    iassert(y.any(axis=0),x.any(axis=0))
    
def test_argmax():
    a = afnumpy.arange(6).reshape((2,3))
    b = numpy.array(a)
    iassert(afnumpy.argmax(a),numpy.argmax(b))
    iassert(afnumpy.argmax(a, axis=0),numpy.argmax(b, axis=0))
    b[0,1] = 5
    a[0,1] = 5
    iassert(afnumpy.argmax(a),numpy.argmax(b))

def test_argmin():
    a = afnumpy.arange(6).reshape((2,3))
    b = numpy.array(a)
    iassert(afnumpy.argmin(a),numpy.argmin(b))
    iassert(afnumpy.argmin(a, axis=0),numpy.argmin(b, axis=0))
    b[0,1] = 5
    a[0,1] = 5
    iassert(afnumpy.argmin(a),numpy.argmin(b))

def test_argsort():
    # Sort does not support 64bit int yet
    x = np.array([3, 1, 2], dtype=float)
    y = af.array(x)
    iassert(af.argsort(y), np.argsort(x))
    x = np.array([[0, 3], [2, 2]], dtype=float)    
    y = af.array(x)
    iassert(af.argsort(y), np.argsort(x))
    iassert(af.argsort(y, axis=1), np.argsort(x, axis=1))
    iassert(af.argsort(y, axis=None), np.argsort(x, axis=None))
    # Arrayfire at the moment can only sort along the last dimension
    # iassert(af.argsort(y, axis=0), np.argsort(x, axis=0))


def test_sort():
    # Sort does not support 64bit int yet
    x = np.array([3, 1, 2], dtype=np.int32)
    y = af.array(x)
    iassert(af.sort(y), np.sort(x))
    x.sort()
    y.sort()
    iassert(y,x)
    x = np.array([[0, 3], [2, 2]], dtype=float)    
    y = af.array(x)
    iassert(af.sort(y), np.sort(x))
    iassert(af.sort(y, axis=1), np.sort(x, axis=1))
    iassert(af.sort(y, axis=None), np.sort(x, axis=None))

    x.sort()
    y.sort()
    iassert(y,x)

    x = np.array([[0, 3], [2, 2]], dtype=float)    
    y = af.array(x)
    x.sort(axis=1)
    y.sort(axis=1)
    iassert(y,x)
    
    x = np.array([[0, 3], [2, 2]], dtype=float)    
    y = af.array(x)
    iassert(af.sort(y,axis=None),np.sort(x,axis=None))

@xfail
def test_sort_xfail():
    x = np.array([[0, 3], [2, 2]], dtype=float)    
    y = af.array(x)
    # Arrayfire at the moment can only sort along the last dimension
    iassert(af.argsort(y, axis=0), np.argsort(x, axis=0))
    
def test_isnan():
    b = 1.0*numpy.random.randint(0,2,(2,3))
    b[b == 0] = numpy.nan
    a = afnumpy.array(b)
    fassert(afnumpy.isnan(a), numpy.isnan(b))

def test_isinf():
    b = 1.0*numpy.random.randint(0,2,(2,3))
    b[b == 0] = numpy.inf
    a = afnumpy.array(b)
    fassert(afnumpy.isnan(a), numpy.isnan(b))

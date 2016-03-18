import afnumpy
import numpy
import afnumpy as af
import numpy as np
from asserts import *
import pytest
xfail = pytest.mark.xfail

def test_zeros():
    a = afnumpy.zeros(3)
    b = numpy.zeros(3)
    iassert(a, b)

def test_fromstring():
    iassert(afnumpy.fromstring('\x01\x02', dtype=numpy.uint8),numpy.fromstring('\x01\x02', dtype=numpy.uint8))

def test_ndarray_transpose():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(a.transpose(), b.transpose())
    iassert(a.transpose(0,1), b.transpose(0,1))
    iassert(a.transpose(1,0), b.transpose(1,0))
    b = numpy.random.random((2))
    a = afnumpy.array(b)
    iassert(a.transpose(), b.transpose())
    b = numpy.random.random((2,3,4))
    a = afnumpy.array(b)
    iassert(a.transpose(), b.transpose())
    iassert(a.transpose((2,0,1)), b.transpose((2,0,1)))
    iassert(a.transpose(2,0,1), b.transpose(2,0,1))

    

def test_where():
    a1 = afnumpy.array([1,2,3])
    b1 = numpy.array(a1)

    a2 = afnumpy.array([0,2,1])
    b2 = numpy.array(a2)
    
    # Test where with input as indices
    iassert(afnumpy.where(a2, a1, a2), numpy.where(b2, b1, b2))
    # Test where with input as indices
    iassert(afnumpy.where(a2), numpy.where(b2))
    # Test where with input as booleans
    iassert(afnumpy.where(a2 < 2, a1, a2), numpy.where(b2 < 2, b1, b2))
    # Test where with input as booleans
    iassert(afnumpy.where(a2 < 2), numpy.where(b2 < 2))

    # And now multidimensional
    a1 = afnumpy.array([[1,2,3],[4,5,6]])
    b1 = numpy.array(a1)

    a2 = afnumpy.array([[0,2,1],[1,0,1]])
    b2 = numpy.array(a2)

    # Test where with input as indices
    iassert(afnumpy.where(a2, a1, a2), numpy.where(b2, b1, b2))
    # Test where with input as indices
    iassert(afnumpy.where(a2), numpy.where(b2))

    # And now multidimensional
    b1 = numpy.random.random((3,3,3)) > 0.5
    a1 = afnumpy.array(b1)

    # Test where with input as indices
    iassert(afnumpy.where(a1), numpy.where(b1))


def test_array():
    a = afnumpy.array([3])
    b = numpy.array([3])
    iassert(a, b)

    a = afnumpy.array([1,2,3])
    b = numpy.array([1,2,3])
    iassert(a, b)

    a = afnumpy.array(numpy.array([1,2,3]))
    b = numpy.array([1,2,3])
    iassert(a, b)

    a = afnumpy.array(numpy.array([1.,2.,3.]))
    b = numpy.array([1.,2.,3.])
    iassert(a, b)

    # Try multidimensional arrays
    a = afnumpy.array(numpy.array([[1.,2.,3.],[4.,5.,6.]]))
    b = numpy.array(a)
    iassert(a, b)

    # Check for non contiguous input
    b = numpy.array([[1.,2.,3.],[4.,5.,6.]]).T
    a = afnumpy.array(b)
    iassert(a, b)


# For some strange reason this fails in Travis
def test_binary_arithmetic():
    a = afnumpy.random.rand(3)
    b = numpy.array(a)

    fassert(a+a, b+b)
    fassert(a+3, b+3)
    fassert(3+a, 3+b)

    fassert(a-a, b-b)
    fassert(a-3, b-3)
    fassert(3-a, 3-b)

    fassert(a*a, b*b)
    fassert(a*3, b*3)
    fassert(3*a, 3*b)

    fassert(a/a, b/b)
    fassert(a/3, b/3)
    fassert(3/a, 3/b)

    fassert(a**a, b**b)
    fassert(a**3, b**3)
    fassert(3**a, 3**b)

    fassert(a%a, b%b)
    fassert(a%3, b%3)
    fassert(3%a, 3%b)

    # Check for arguments of diffeernt types
    a = afnumpy.ones(3,dtype=numpy.uint32)
    b = numpy.array(a)
    fassert(a+3.0, b+3.0)
    # This is a tricky case we won't support for now
    # fassert(a+numpy.float32(3.0), b+numpy.float32(3.0))
    fassert(3.0+a, 3.0+b)

    fassert(a-3.0, b-3.0)
    fassert(3.0-a, 3.0-b)

    fassert(a*3.0, b*3.0)
    fassert(3.0*a, 3.0*b)

    fassert(a/3.0, b/3.0)
    fassert(3.0/a, 3.0/b)

    fassert(a**3.0, b**3.0)
    fassert(3.0**a, 3.0**b)

    fassert(a%3.0, b%3.0)
    fassert(3.0%a, 3.0%b)


def test_broadcast_binary_arithmetic():
    a = afnumpy.random.rand(2,3)
    b = afnumpy.random.rand(2,1)
    c = numpy.array(a)
    d = numpy.array(b)    
    fassert(a*b, c*d)
    a*=b
    c*=d
    fassert(a, c)
    fassert(a/b, c/d)
    a/=b
    c/=d
    fassert(a, c)
    fassert(a+b, c+d)
    a+=b
    c+=d
    fassert(a, c)
    fassert(a-b, c-d)
    a-=b
    c-=d
    fassert(a, c)

def test_augmented_assignment():
    a = afnumpy.random.rand(3)
    b = numpy.array(a)

    mem_before = a.d_array.device_ptr()
    a += a
    assert mem_before == a.d_array.device_ptr()
    b += b
    fassert(a, b)
    mem_before = a.d_array.device_ptr()
    a += 3
    assert mem_before == a.d_array.device_ptr()
    b += 3
    fassert(a, b)

    mem_before = a.d_array.device_ptr()
    a -= a
    assert mem_before == a.d_array.device_ptr()
    b -= b
    fassert(a, b)
    mem_before = a.d_array.device_ptr()
    a -= 3
    assert mem_before == a.d_array.device_ptr()
    b -= 3
    fassert(a, b)

    mem_before = a.d_array.device_ptr()
    a *= a
    assert mem_before == a.d_array.device_ptr()
    b *= b
    fassert(a, b)
    mem_before = a.d_array.device_ptr()
    a *= 3
    assert mem_before == a.d_array.device_ptr()
    b *= 3
    fassert(a, b)

    mem_before = a.d_array.device_ptr()
    a /= a
    assert mem_before == a.d_array.device_ptr()
    b /= b
    fassert(a, b)
    mem_before = a.d_array.device_ptr()
    a /= 3
    assert mem_before == a.d_array.device_ptr()
    b /= 3
    fassert(a, b)

def test_unary_operators():
    a = afnumpy.random.rand(3)
    b = numpy.array(a)
    fassert(-a, -b)
    fassert(+a, +b)
    b = numpy.random.randint(0,2,3).astype('bool')
    a = afnumpy.array(b)
    fassert(-a, -b)
    fassert(+a, +b)
    # fassert(~a, ~b)

def test_comparisons():
    a1 = afnumpy.random.rand(3)
    b1 = numpy.array(a1)

    a2 = afnumpy.random.rand(3)
    b2 = numpy.array(a2)

    iassert(a1 > a2, b1 > b2)
    iassert(a1 > 0.5, b1 > 0.5)
    iassert(0.5 > a1, 0.5 > b1)

    iassert(a1 >= a2, b1 >= b2)
    iassert(a1 >= 0.5, b1 >= 0.5)
    iassert(0.5 >= a1, 0.5 >= b1)

    iassert(a1 < a2, b1 < b2)
    iassert(a1 < 0.5, b1 < 0.5)
    iassert(0.5 < a1, 0.5 < b1)

    iassert(a1 <= a2, b1 <= b2)
    iassert(a1 <= 0.5, b1 <= 0.5)
    iassert(0.5 <= a1, 0.5 <= b1)

    iassert(a1 == a2, b1 == b2)
    iassert(a1 == 0.5, b1 == 0.5)
    iassert(0.5 == a1, 0.5 == b1)

    iassert(a1 != a2, b1 != b2)
    iassert(a1 != 0.5, b1 != 0.5)
    iassert(0.5 != a1, 0.5 != b1)

def test_ndarray_all():    
    b = numpy.random.randint(0,2,3).astype('bool')
    a = afnumpy.array(b)
    iassert(a.all(), b.all())
    iassert(a.all(axis=0), b.all(axis=0))

    b = numpy.random.randint(0,2,(3,2)).astype('bool')
    a = afnumpy.array(b)
    iassert(a.all(), b.all())
    iassert(a.all(axis=0), b.all(axis=0))
    iassert(a.all(keepdims=True), b.all(keepdims=True))

def test_sum():    
    b = numpy.random.random(3)
    a = afnumpy.array(b)
    fassert(afnumpy.sum(a), numpy.sum(b))
    fassert(afnumpy.sum(a,axis=0), numpy.sum(b,axis=0))
    fassert(afnumpy.sum(a,keepdims=True), numpy.sum(b,keepdims=True))

    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    fassert(afnumpy.sum(a), numpy.sum(b))
    fassert(afnumpy.sum(a,axis=0), numpy.sum(b,axis=0))

def test_max():    
    b = numpy.random.random(3)+numpy.random.random(3)*1.0j
    a = afnumpy.array(b)
    # Arrayfire uses the magnitude for max while numpy uses
    # the real part as primary key followed by the imaginary part
    # fassert(a.max(), b.max())
    b = numpy.random.random(3)
    a = afnumpy.array(b)
    fassert(a.max(), b.max())

def test_min():    
    b = numpy.random.random(3)+numpy.random.random(3)*1.0j
    a = afnumpy.array(b)
    # Arrayfire uses the magnitude for max while numpy uses
    # the real part as primary key followed by the imaginary part
    # fassert(a.min(), b.min())
    b = numpy.random.random(3)
    a = afnumpy.array(b)
    fassert(a.min(), b.min())

def test_ndarray_abs():    
    b = numpy.random.random(3)+numpy.random.random(3)*1.0j
    a = afnumpy.array(b)
    fassert(abs(a), abs(b))
    b = numpy.random.random(3)
    a = afnumpy.array(b)
    fassert(abs(a), abs(b))

def test_getitem():
    b = numpy.random.random((3))
    a = afnumpy.array(b)
    iassert(a[0], b[0])
    iassert(a[2], b[2])
    iassert(a[:], b[:])
    iassert(a[0:], b[0:])
    iassert(a[:-1], b[:-1])
    iassert(a[0:-1], b[0:-1])
    iassert(a[1:-1], b[1:-1])
    iassert(a[1:2], b[1:2])
    iassert(a[...,0], b[...,0])
    # This will return an empty array, which is not yet supported
    # iassert(a[1:1], b[1:1])
    iassert(a[-2:], b[-2:])
    iassert(a[-3:-1], b[-3:-1])
    iassert(a[1:-1:1], b[1:-1:1])
    iassert(a[1:-1:2], b[1:-1:2])
    iassert(a[::2], b[::2])
    iassert(a[::3], b[::3])
    iassert(a[::-1], b[::-1])
    iassert(a[::-2], b[::-2])
    iassert(a[-1::-1], b[-1::-1])
    iassert(a[-1:1:-1], b[-1:1:-1])
    iassert(a[-2::-1], b[-2::-1])
    iassert(a[-2:0:-1], b[-2:0:-1])
    iassert(a[-2::-2], b[-2::-2])
    iassert(a[-2::2], b[-2::2])
    iassert(a[([0],)], b[([0],)])
    
    # Now multidimensional!
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
           
    iassert(a[:], b[:])
    iassert(a[0], b[0])
    iassert(a[:,2], b[:,2])
    iassert(a[1,:], b[1,:])
    iassert(a[:,::-1], b[:,::-1])
    
    # Boolean indexing
    d = numpy.random.random((2)) > 0.5
    c = afnumpy.array(d)
    iassert(a[c,:], b[d,:])


    b = numpy.random.random((2,3,1))
    a = afnumpy.array(b)
    iassert(a[:], b[:])

    b = numpy.random.random((2,3,1,2))
    a = afnumpy.array(b)
    iassert(a[:], b[:])
    iassert(a[1,:,:,:], b[1,:,:,:])
    iassert(a[1,:,0,:], b[1,:,0,:])
    iassert(a[1,1,:,:], b[1,1,:,:])
    d = numpy.array([0,2],dtype=numpy.int32)
    c = afnumpy.array(d)
    iassert(a[1,c,0,:], b[1,d,0,:])

    # Boolean indexing
    d = b > 0.5
    c = afnumpy.array(d)
    iassert(a[c], b[d])

    d = numpy.random.random((2,3)) > 0.5
    c = afnumpy.array(d)
    iassert(a[c,:], b[d,:])

    # Zero dimensional
    b = numpy.ones(())
    a = afnumpy.array(b)
    iassert(a[()],b[()])

    # Slices that extend outside the array
    b = numpy.ones((3))
    a = afnumpy.array(b)
    iassert(a[1:4],b[1:4])
    iassert(a[3::-1],b[3::-1])

    # Partial boolean indexing
    b = numpy.ones((3,3))
    a = afnumpy.array(b)
    d = numpy.ones((3)) > 0
    c = afnumpy.array(d)
    iassert(a[c],b[d])

    # Partial array indexing
    b = numpy.ones((3,3))
    a = afnumpy.array(b)
    d = numpy.array([0,1])
    c = afnumpy.array(d)
    iassert(a[c],b[d])


def test_getitem_multi_array():
    # Multidimensional array indexing
    b = numpy.random.random((2,2))
    a = afnumpy.array(b)
    d = numpy.array([0,1])
    c = afnumpy.array(d)
    iassert(a[c,c], b[d,d])

def test_newaxis():
    b = numpy.random.random((3))
    a = afnumpy.array(b)
    # iassert(a[afnumpy.newaxis,:], b[numpy.newaxis,:])

def test_setitem():
    b = numpy.random.random((3))
    a = afnumpy.array(b)
    mem_before = a.d_array.device_ptr()
    a[0] = 1;
    b[0] = 1;
    iassert(a, b)
    assert mem_before == a.d_array.device_ptr()
    a[:] = 2;
    b[:] = 2;
    assert mem_before == a.d_array.device_ptr()
    iassert(a, b)
    d = numpy.array([0,1],dtype=numpy.int32)
    c = afnumpy.array(d)
    a[c] = 3;
    b[d] = 3;
    assert mem_before == a.d_array.device_ptr()

    # Multidimensional
    # 2D
    b1 = numpy.random.random((2,2))
    b2 = numpy.random.random(2)
    a1 = afnumpy.array(b1)
    a2 = afnumpy.array(b2)
    mem_before = a1.d_array.device_ptr()
    a1[:] = 1
    b1[:] = 1
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_ptr()
    a1[:,0] = a2[:]
    b1[:,0] = b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_ptr()
    a1[c,0] = -a2[:]
    b1[d,0] = -b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_ptr()
    a1[0,c] = a2[:]
    b1[0,d] = b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_ptr()
    a1[0] = a2[:]
    b1[0] = b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_ptr()

    # 3D
    b1 = numpy.random.random((2,3,1))
    b2 = numpy.random.random((3,1))
    a1 = afnumpy.array(b1)
    a2 = afnumpy.array(b2)
    mem_before = a1.d_array.device_ptr()
    a1[0,:,:] = a2[:]
    b1[0,:,:] = b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_ptr()

    a1[0] = a2[:]
    b1[0] = b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_ptr()

    # 4D
    b1 = numpy.random.random((2,3,2,2))
    b2 = numpy.random.random((2,2))
    a1 = afnumpy.array(b1)
    a2 = afnumpy.array(b2)
    d = numpy.array([0,1],dtype=numpy.int32)
    c = afnumpy.array(d)
    mem_before = a1.d_array.device_ptr()
    a1[:,0,0,c] = a2
    b1[:,0,0,d] = b2
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_ptr()

    a1[1,2] = a2
    b1[1,2] = b2
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_ptr()
    
    # Boolean indexing
    d = b > 0.5
    c = afnumpy.array(d)
    a[c] = 1
    b[d] = 1
    iassert(a, b)
    a[a < 0.3] = 1
    b[b < 0.3] = 1
    iassert(a, b)

    # Multidimensional Boolean
    a1[a1 < 0.3] = 1
    b1[b1 < 0.3] = 1
    iassert(a1, b1)

def test_setitem_multi_array():
    # Multidimensional array indexing
    b = numpy.random.random((2,2))
    a = afnumpy.array(b)
    d = numpy.array([0,1])
    c = afnumpy.array(d)
    # This will fail because while multiple arrays
    # as indices in numpy treat the values given by
    # the arrays as the coordinates of the hyperslabs
    # to keep arrayfire does things differently.
    # In arrayfire each entry of each array gets combined
    # with all entries of all other arrays to define the coordinate
    # In numpy each entry only gets combined with the corresponding
    # entry in the other arrays.
    # For example if one has [0,1],[0,1] as the two arrays for numpy
    # this would mean that the coordinates retrieved would be [0,0], 
    # [1,1] while for arrayfire it would be [0,0], [0,1], [1,0], [1,1].
    a[c,c] = c 
    b[d,d] = d
    iassert(a, b)

def test_views():
    b = numpy.random.random((3,3))
    a = afnumpy.array(b)
    a[0] = 1
    b[0] = 1
    c = a[0]
    d = b[0]
    c[:] = 0
    d[:] = 0
    iassert(a,b)

    assert a[0,:].d_array.device_ptr() == a[0,:].d_array.device_ptr()
    # There is currently no way to get views with stride[0] > 1
    # assert a[:,0].d_array.device_ptr() == a[:,0].d_array.device_ptr()

    b = numpy.random.random((3))
    a = afnumpy.array(b)
    c = a[...,0]
    assert a.d_array.device_ptr() == c.d_array.device_ptr()
    d = b[...,0]
    c[()] = 0
    d[()] = 0
    iassert(a,b)

def test_ndarray_astype():
    b = numpy.random.random(3)
    a = afnumpy.array(b)
    iassert(a.astype(numpy.uint8),b.astype(numpy.uint8))
    iassert(a.astype(numpy.complex128),b.astype(numpy.complex128))

def test_ndarray_len():
    b = numpy.random.random(3)
    a = afnumpy.array(b)
    assert(len(a) == len(b))
    b = numpy.random.random((3,3))
    a = afnumpy.array(b)
    assert(len(a) == len(b))


def test_vstack():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(afnumpy.vstack(a), numpy.vstack(b))
    iassert(afnumpy.vstack((a,a)), numpy.vstack((b,b)))

def test_hstack():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(afnumpy.hstack(a), numpy.hstack(b))
    iassert(afnumpy.hstack((a,a)), numpy.hstack((b,b)))


def test_empty_ndarray():
    a = afnumpy.zeros(())
    b = numpy.zeros(())
    iassert(a,b)
    a = afnumpy.ndarray(0)
    b = numpy.ndarray(0)
    iassert(a,b)
    a = afnumpy.ndarray((0,))
    b = numpy.ndarray((0,))
    iassert(a,b)
    a = afnumpy.zeros(3)
    b = numpy.zeros(3)
    iassert(a[0:0],b[0:0])
    


def test_arange():
    iassert(afnumpy.arange(10), numpy.arange(10))
    iassert(afnumpy.arange(1,10), numpy.arange(1,10))
    iassert(afnumpy.arange(10,1,-1), numpy.arange(10,1,-1))
    iassert(afnumpy.arange(10,1,-1,dtype=numpy.int32), numpy.arange(10,1,-1,dtype=numpy.int32))

def test_ndarray_shape():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    a.shape = (3,2)
    b.shape = (3,2)
    fassert(a,b)


def test_ndarray_round():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    fassert(a.round(), b.round())

def test_ndarray_take():
    b = numpy.array([4, 3, 5, 7, 6, 8])
    a = afnumpy.array(b)
    indices = [0, 1, 4]
    iassert(a.take(indices), b.take(indices))
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(a.take([0,1],axis=1), b.take([0,1],axis=1))
    iassert(a.take([0,1]), b.take([0,1]))

def test_ndarray_min():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(a.min(), b.min())
    fassert(a.min(axis=1), b.min(axis=1))
    fassert(a.min(axis=1, keepdims=True), b.min(axis=1, keepdims=True))

def test_ndarray_max():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(a.max(), b.max())
    fassert(a.max(axis=1), b.max(axis=1))
    fassert(a.max(axis=1, keepdims=True), b.max(axis=1, keepdims=True))

def test_ndarray_sum():
    a = afnumpy.random.random((2,3))
    b = numpy.array(a)
    fassert(a.sum(), b.sum())
    fassert(a.sum(axis=1), b.sum(axis=1))
    fassert(a.sum(axis=1, keepdims=True), b.sum(axis=1, keepdims=True))
    fassert(a.sum(axis=(0,1), keepdims=True), b.sum(axis=(0,1), keepdims=True))
    fassert(a.sum(axis=(0,1)), b.sum(axis=(0,1)))
    a = afnumpy.random.random(())
    b = afnumpy.array(a)
    fassert(a.sum(), b.sum())

def test_ndarray_conj():
    # The weird astype is because of issue #914 in arrayfire
    a =afnumpy.random.random((2,3)).astype(numpy.complex64)+1.0j
    b = numpy.array(a)
    fassert(a.conj(), b.conj())

def test_empty():
    a = afnumpy.empty((2,3))
    b = numpy.array(a)
    a[:] = 1
    b[:] = 1
    fassert(a,b)

def test_ndarray_T():
    x = numpy.array([[1.,2.],[3.,4.]])
    y = afnumpy.array(x)
    fassert(y.T,x.T)
    x = numpy.array([1.,2.,3.,4.])
    y = afnumpy.array(x)
    fassert(y.T,x.T)

def test_ndarray_any():
    x = numpy.array([[True, False], [True, True]])
    y = afnumpy.array(x)
    iassert(y.any(),x.any())
    iassert(y.any(axis=0),x.any(axis=0))

def test_ndarray_real():
    x = np.sqrt([1+0j, 0+1j])
    y = af.array(x)
    fassert(y.real, x.real)
    y.real[:] = 0
    x.real[:] = 0
    fassert(y, x)

def test_ndarray_imag():
    x = np.sqrt([1+0j, 0+1j])
    y = af.array(x)
    fassert(y.imag, x.imag)
    y.imag[:] = 0
    x.imag[:] = 0
    fassert(y, x)

def test_ndarray_strides():
    a = afnumpy.random.random((4,3))
    b = numpy.array(a)
    iassert(a.strides, b.strides)
    iassert(a[:,:].strides, b[:,:].strides)
    iassert(a[1:,:].strides, b[1:,:].strides)
    iassert(a[:,1:].strides, b[:,1:].strides)
    # The following cases fails for arrayfire < 3.3 as the stride
    # hack requires at least 2 elements per dimension
    iassert(a[3:,:].strides, b[3:,:].strides)
    iassert(a[2:,2:].strides, b[2:,2:].strides)
    iassert(a[3,:2].strides, b[3,:2].strides)

@xfail
def test_ndarray_strides_xfail():
    # The following case fails as arrayfire always drops
    # leading dimensions of size 1 and so the stride
    # information is missing
    a = afnumpy.random.random((4,3))
    b = numpy.array(a)
    iassert(a[3:,:2].strides, b[3:,:2].strides)

def test_ndarray_copy():
    b = numpy.random.random((3,3))
    a = afnumpy.array(b)
    iassert(a.copy(), b.copy())

def test_ndarray_nonzero():
    b = numpy.random.random((3,3,3)) > 0.5
    a = afnumpy.array(b)    
    iassert(a.nonzero(), b.nonzero())

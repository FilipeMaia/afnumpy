import afnumpy
import numpy
from IPython.core.debugger import Tracer
from asserts import *

def test_zeros():
    a = afnumpy.zeros(3)
    b = numpy.zeros(3)
    iassert(a, b)

def test_fromstring():
    iassert(numpy.fromstring('\x01\x02', dtype=numpy.uint8),afnumpy.fromstring('\x01\x02', dtype=numpy.uint8))

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

    mem_before = a.d_array.device_f32()
    a += a
    assert mem_before == a.d_array.device_f32()
    b += b
    fassert(a, b)
    mem_before = a.d_array.device_f32()
    a += 3
    assert mem_before == a.d_array.device_f32()
    b += 3
    fassert(a, b)

    mem_before = a.d_array.device_f32()
    a -= a
    assert mem_before == a.d_array.device_f32()
    b -= b
    fassert(a, b)
    mem_before = a.d_array.device_f32()
    a -= 3
    assert mem_before == a.d_array.device_f32()
    b -= 3
    fassert(a, b)

    mem_before = a.d_array.device_f32()
    a *= a
    assert mem_before == a.d_array.device_f32()
    b *= b
    fassert(a, b)
    mem_before = a.d_array.device_f32()
    a *= 3
    assert mem_before == a.d_array.device_f32()
    b *= 3
    fassert(a, b)

    mem_before = a.d_array.device_f32()
    a /= a
    assert mem_before == a.d_array.device_f32()
    b /= b
    fassert(a, b)
    mem_before = a.d_array.device_f32()
    a /= 3
    assert mem_before == a.d_array.device_f32()
    b /= 3
    fassert(a, b)

def test_unary_operators():
    a = afnumpy.random.rand(3)
    b = numpy.array(a)
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

def test_all():    
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

    # Zero dimensional
    b = numpy.ones(())
    a = afnumpy.array(b)
    iassert(a[()],b[()])



def test_newaxis():
    b = numpy.random.random((3))
    a = afnumpy.array(b)
    # iassert(a[afnumpy.newaxis,:], b[numpy.newaxis,:])

def test_setitem():
    b = numpy.random.random((3))
    a = afnumpy.array(b)
    mem_before = a.d_array.device_f32()
    a[0] = 1;
    b[0] = 1;
    iassert(a, b)
    assert mem_before == a.d_array.device_f32()
    a[:] = 2;
    b[:] = 2;
    assert mem_before == a.d_array.device_f32()
    iassert(a, b)
    d = numpy.array([0,1],dtype=numpy.int32)
    c = afnumpy.array(d)
    a[c] = 3;
    b[d] = 3;
    assert mem_before == a.d_array.device_f32()

    # Multidimensional
    # 2D
    b1 = numpy.random.random((2,2))
    b2 = numpy.random.random(2)
    a1 = afnumpy.array(b1)
    a2 = afnumpy.array(b2)
    mem_before = a1.d_array.device_f32()
    a1[:] = 1
    b1[:] = 1
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_f32()
    a1[:,0] = a2[:]
    b1[:,0] = b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_f32()
    a1[c,0] = -a2[:]
    b1[d,0] = -b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_f32()
    a1[0,c] = a2[:]
    b1[0,d] = b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_f32()
    a1[0] = a2[:]
    b1[0] = b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_f32()

    # 3D
    b1 = numpy.random.random((2,3,1))
    b2 = numpy.random.random((3,1))
    a1 = afnumpy.array(b1)
    a2 = afnumpy.array(b2)
    mem_before = a1.d_array.device_f32()
    a1[0,:,:] = a2[:]
    b1[0,:,:] = b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_f32()

    a1[0] = a2[:]
    b1[0] = b2[:]
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_f32()

    # 4D
    b1 = numpy.random.random((2,3,2,2))
    b2 = numpy.random.random((2,2))
    a1 = afnumpy.array(b1)
    a2 = afnumpy.array(b2)
    d = numpy.array([0,1],dtype=numpy.int32)
    c = afnumpy.array(d)
    mem_before = a1.d_array.device_f32()
    a1[:,0,0,c] = a2
    b1[:,0,0,d] = b2
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_f32()

    a1[1,2] = a2
    b1[1,2] = b2
    iassert(a1,b1)
    assert mem_before == a1.d_array.device_f32()
    
def test_views():
    b = numpy.random.random((3,3))
    a = afnumpy.array(b)
    c = a[0]
    d = b[0]
    c[:] = 0
    d[:] = 0
    iassert(a,b)

    b = numpy.random.random((3))
    a = afnumpy.array(b)
    c = a[...,0]
    assert a.d_array.device_f32() == c.d_array.device_f32()
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

def test_ndarray_strides():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(a.strides, b.strides)

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

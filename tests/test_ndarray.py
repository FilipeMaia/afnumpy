import afnumpy
import numpy
from numpy.testing import assert_allclose as fassert
from IPython.core.debugger import Tracer

def iassert(af_a, np_a):
    assert numpy.all(numpy.array(af_a) == np_a)

def fassert(af_a, np_a):
    numpy.testing.assert_allclose(numpy.array(af_a), np_a)

def test_ones():
    a = afnumpy.ones(3)
    b = numpy.ones(3)
    iassert(a, b)

def test_zeros():
    a = afnumpy.zeros(3)
    b = numpy.zeros(3)
    iassert(a, b)

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
    iassert(afnumpy.all(a), numpy.all(b))
    iassert(afnumpy.all(a,axis=0), numpy.all(b,axis=0))

    b = numpy.random.randint(0,2,(3,2)).astype('bool')
    a = afnumpy.array(b)
    iassert(afnumpy.all(a), numpy.all(b))
    iassert(afnumpy.all(a,axis=0), numpy.all(b,axis=0))
    # Not implemented
    # iassert(afnumpy.all(a,keepdims=True), numpy.all(b,keepdims=True))

def test_sum():    
    b = numpy.random.random(3)
    a = afnumpy.array(b)
    fassert(afnumpy.sum(a), numpy.sum(b))
    fassert(afnumpy.sum(a,axis=0), numpy.sum(b,axis=0))
    # Not implemented
    # fassert(afnumpy.sum(a,keepdims=True), numpy.all(b,keepdims=True))

    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    fassert(afnumpy.sum(a), numpy.sum(b))
    fassert(afnumpy.sum(a,axis=0), numpy.sum(b,axis=0))

def test_vdot():    
    b = numpy.random.random(3)+numpy.random.random(3)*1.0j
    a = afnumpy.array(b)
    fassert(afnumpy.vdot(a,a), numpy.vdot(b,b))

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
    fassert(a.max(), b.max())

def test_abs():    
    b = numpy.random.random(3)+numpy.random.random(3)*1.0j
    a = afnumpy.array(b)
    fassert(abs(a), abs(b))
    b = numpy.random.random(3)
    a = afnumpy.array(b)
    fassert(abs(a), abs(b))

        
def test_reshape():
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
    iassert(a.reshape((3,2)), b.reshape((3,2)))
    iassert(a.reshape(6), b.reshape(6))

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
    iassert(a[1:1], b[1:1])
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
    
    # Now multidimensional!
    b = numpy.random.random((2,3))
    a = afnumpy.array(b)
            
    iassert(a[:], b[:])
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
    # This shows the error
    #iassert(a[1,:,0,:], b[1,:,0,:])
    # This shows the error
    iassert(a[1,1,:,:], b[1,1,:,:])


def test_setitem():
    b = numpy.random.random((3))
    a = afnumpy.array(b)
    mem_before = a.d_array.device_f32()
    a[0] = 1;
    b[0] = 1;
    iassert(a, b)
    assert mem_before == a.d_array.device_f32()
    a[:] = 1;
    b[:] = 1;
    assert mem_before == a.d_array.device_f32()
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

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

    a += a
    b += b
    fassert(a, b)
    a += 3
    b += 3
    fassert(a, b)

    a -= a
    b -= b
    fassert(a, b)
    a -= 3
    b -= 3
    fassert(a, b)

    a *= a
    b *= b
    fassert(a, b)
    a *= 3
    b *= 3
    fassert(a, b)

    a /= a
    b /= b
    fassert(a, b)
    a /= 3
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

    

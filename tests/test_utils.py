import afnumpy
import numpy
from numpy.testing import assert_allclose as fassert

def iassert(af_a, np_a):
    assert numpy.all(numpy.array(af_a) == np_a)

def fassert(af_a, np_a):
    numpy.testing.assert_allclose(numpy.array(af_a), np_a)

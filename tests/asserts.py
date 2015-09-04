import afnumpy
import numpy
from numpy.testing import assert_allclose
import numbers

def massert(af_a, np_a):
    # Assert the metadata of the arrays

    if isinstance(af_a, tuple):
        assert(af_a == np_a)
    elif isinstance(af_a, afnumpy.ndarray):
        assert isinstance(np_a, numpy.ndarray)
        # I will not strictly enforce float32 vs float64
        # I will not strictly enforce uint32 vs int64
        assert af_a.dtype == np_a.dtype or (af_a.dtype == numpy.float32 and np_a.dtype == numpy.float64) or (af_a.dtype == numpy.uint32 and np_a.dtype == numpy.int64)
        assert (af_a.shape == np_a.shape)
    elif isinstance(af_a, numbers.Number):
        assert isinstance(af_a, numbers.Number)
        assert isinstance(np_a, numbers.Number)
    elif isinstance(af_a, numpy.number):
        assert isinstance(af_a, numpy.number)
        assert isinstance(np_a, numpy.number)
    else:
        assert type(af_a) == type(np_a)

def iassert(af_a, np_a):
    if not isinstance(af_a, tuple) and not isinstance(af_a, list):
        af_a = (af_a,)
        np_a = (np_a,)
    for a,b in zip(af_a,np_a):
        assert numpy.all(numpy.array(a) == b)
        massert(a, b)

def fassert(af_a, np_a):
    numpy.testing.assert_allclose(numpy.array(af_a), np_a)
    massert(af_a, np_a)

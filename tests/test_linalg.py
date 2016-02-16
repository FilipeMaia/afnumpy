import afnumpy
import numpy
from asserts import *
import pytest
xfail = pytest.mark.xfail


# For some strange reason this fails in Travis
@xfail
def test_norm():
    a = afnumpy.random.random((3))
    b = numpy.array(a)
    fassert(afnumpy.linalg.norm(a), numpy.linalg.norm(b))
    fassert(afnumpy.linalg.norm(a, ord=0), numpy.linalg.norm(b, ord=0))
    fassert(afnumpy.linalg.norm(a, ord=1), numpy.linalg.norm(b, ord=1))
    fassert(afnumpy.linalg.norm(a, ord=-1), numpy.linalg.norm(b, ord=-1))
    fassert(afnumpy.linalg.norm(a, ord=2), numpy.linalg.norm(b, ord=2))
    fassert(afnumpy.linalg.norm(a, ord=-2), numpy.linalg.norm(b, ord=-2))
    fassert(afnumpy.linalg.norm(a, ord=3), numpy.linalg.norm(b, ord=3))
    fassert(afnumpy.linalg.norm(a, ord=numpy.Inf), numpy.linalg.norm(b, ord=numpy.Inf))
    fassert(afnumpy.linalg.norm(a, ord=-numpy.Inf), numpy.linalg.norm(b, ord=-numpy.Inf))

def test_vdot():    
    b = numpy.random.random(3)+numpy.random.random(3)*1.0j
    a = afnumpy.array(b)
    fassert(afnumpy.vdot(a,a), numpy.vdot(b,b))


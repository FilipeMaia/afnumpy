import afnumpy
import afnumpy.fft
import numpy
import numpy.fft
from IPython.core.debugger import Tracer
from test_utils import *

def test_fft2():    
    b = numpy.random.random((3,3))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fft2(a), numpy.fft.fft2(b))

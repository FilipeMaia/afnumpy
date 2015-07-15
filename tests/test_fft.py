import afnumpy
import afnumpy.fft
import numpy
import numpy.fft
from IPython.core.debugger import Tracer
from test_utils import *

def test_fft():    
    b = numpy.random.random((3,3))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fft(a), numpy.fft.fft(b))

    b = numpy.random.random((3,2))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fft(a), numpy.fft.fft(b))

    b = numpy.random.random((5,3,2))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fft(a), numpy.fft.fft(b))

    b = numpy.random.random((5,3,2))+numpy.random.random((5,3,2))*1.0j
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fft(a), numpy.fft.fft(b))

def test_ifft():    
    # Real to complex inverse fft not implemented in arrayfire
    # b = numpy.random.random((3,3))
    # a = afnumpy.array(b)
    # fassert(afnumpy.fft.ifft(a), numpy.fft.ifft(b))

    # b = numpy.random.random((3,2))
    # a = afnumpy.array(b)
    # fassert(afnumpy.fft.ifft(a), numpy.fft.ifft(b))

    # b = numpy.random.random((5,3,2))
    # a = afnumpy.array(b)
    # fassert(afnumpy.fft.ifft(a), numpy.fft.ifft(b))

    b = numpy.random.random((5,3,2))+numpy.random.random((5,3,2))*1.0j
#    b = numpy.ones((3,3))+numpy.zeros((3,3))*1.0j
    a = afnumpy.array(b)
    fassert(afnumpy.fft.ifft(a), numpy.fft.ifft(b))

def test_fft2():    
    b = numpy.random.random((3,3))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fft2(a), numpy.fft.fft2(b))

    b = numpy.random.random((3,2))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fft2(a), numpy.fft.fft2(b))

    b = numpy.random.random((5,3,2))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fft2(a), numpy.fft.fft2(b))

    b = numpy.random.random((5,3,2))+numpy.random.random((5,3,2))*1.0j
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fft2(a), numpy.fft.fft2(b))

def test_ifft2():    
    # Real to complex inverse fft not implemented in arrayfire
    # b = numpy.random.random((3,3))
    # a = afnumpy.array(b)
    # fassert(afnumpy.fft.ifft2(a), numpy.fft.ifft2(b))

    # b = numpy.random.random((3,2))
    # a = afnumpy.array(b)
    # fassert(afnumpy.fft.ifft2(a), numpy.fft.ifft2(b))

    # b = numpy.random.random((5,3,2))
    # a = afnumpy.array(b)
    # fassert(afnumpy.fft.ifft2(a), numpy.fft.ifft2(b))

    b = numpy.random.random((5,3,2))+numpy.random.random((5,3,2))*1.0j
#    b = numpy.ones((3,3))+numpy.zeros((3,3))*1.0j
    a = afnumpy.array(b)
    fassert(afnumpy.fft.ifft2(a), numpy.fft.ifft2(b))

def test_fftn():    
    b = numpy.random.random((3,3))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fftn(a), numpy.fft.fftn(b))

    b = numpy.random.random((3,2))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fftn(a), numpy.fft.fftn(b))

    b = numpy.random.random((5,3,2))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fftn(a), numpy.fft.fftn(b))

    b = numpy.random.random((5,3,2))+numpy.random.random((5,3,2))*1.0j
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fftn(a), numpy.fft.fftn(b))

def test_ifftn():    
    # Real to complex inverse fft not implemented in arrayfire
    # b = numpy.random.random((3,3))
    # a = afnumpy.array(b)
    # fassert(afnumpy.fft.ifftn(a), numpy.fft.ifftn(b))

    # b = numpy.random.random((3,2))
    # a = afnumpy.array(b)
    # fassert(afnumpy.fft.ifftn(a), numpy.fft.ifftn(b))

    # b = numpy.random.random((5,3,2))
    # a = afnumpy.array(b)
    # fassert(afnumpy.fft.ifftn(a), numpy.fft.ifftn(b))

    b = numpy.random.random((5,3,2))+numpy.random.random((5,3,2))*1.0j
#    b = numpy.ones((3,3))+numpy.zeros((3,3))*1.0j
    a = afnumpy.array(b)
    fassert(afnumpy.fft.ifftn(a), numpy.fft.ifftn(b))

def test_fftshift():
    b = numpy.random.random((3))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fftshift(a), numpy.fft.fftshift(b))
    b = numpy.random.random((3,3))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fftshift(a), numpy.fft.fftshift(b))
    b = numpy.random.random((3,3,3))
    a = afnumpy.array(b)
    fassert(afnumpy.fft.fftshift(a), numpy.fft.fftshift(b))
    fassert(afnumpy.fft.fftshift(a,axes=0), numpy.fft.fftshift(b,axes=0))
    fassert(afnumpy.fft.fftshift(a,axes=1), numpy.fft.fftshift(b,axes=1))
    fassert(afnumpy.fft.fftshift(a,axes=2), numpy.fft.fftshift(b,axes=2))
    fassert(afnumpy.fft.fftshift(a,axes=(1,2)), numpy.fft.fftshift(b,axes=(1,2)))

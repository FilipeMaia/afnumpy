import afnumpy
import arrayfire
import numpy

def test_fft2(benchmark):
    a = afnumpy.ones((256,256),dtype=numpy.complex64)
    benchmark(afnumpy.fft.fftn, a)

def test_ifft2(benchmark):
    a = afnumpy.ones((256,256),dtype=numpy.complex64)
    benchmark(afnumpy.fft.ifftn, a)

def test_fftshift(benchmark):
    a = afnumpy.ones((256,256),dtype=numpy.complex64)
    benchmark(afnumpy.fft.fftshift, a)

def test_ifftshift(benchmark):
    a = afnumpy.ones((256,256),dtype=numpy.complex64)
    benchmark(afnumpy.fft.ifftshift, a)

def test_array_nocopy(benchmark):
    a = afnumpy.ones((2048,2048),dtype=numpy.complex64)
    benchmark(afnumpy.array, a, copy=False)
        

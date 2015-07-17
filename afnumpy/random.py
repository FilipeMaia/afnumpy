import numpy.random
from . import ndarray

def rand(*args):
    a = numpy.random.rand(*args)
    return ndarray(a.shape, a.dtype, buffer=a)


def random(size = None):
    a = numpy.random.random(size)
    return ndarray(a.shape, a.dtype, buffer=a)

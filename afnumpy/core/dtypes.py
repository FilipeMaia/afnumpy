import numpy
import afnumpy
import numbers


class int64(numpy.int64):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.int64(x)
        else:
            return afnumpy.array(x).astype(cls)

class int32(numpy.int32):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.int32(x)
        else:
            return afnumpy.array(x).astype(cls)

class int16(numpy.intc):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.int16(x)
        else:
            return afnumpy.array(x).astype(cls)

class int8(numpy.int8):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.int8(x)
        else:
            return afnumpy.array(x).astype(cls)


class uint64(numpy.uint64):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.uint64(x)
        else:
            return afnumpy.array(x).astype(cls)

class uint32(numpy.uint32):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.uint32(x)
        else:
            return afnumpy.array(x).astype(cls)

class uint16(numpy.uint16):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.uint16(x)
        else:
            return afnumpy.array(x).astype(cls)

class uint8(numpy.uint8):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.uint8(x)
        else:
            return afnumpy.array(x).astype(cls)


class intc(numpy.intc):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.intc(x)
        else:
            return afnumpy.array(x).astype(cls)

class intp(numpy.intp):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.intp(x)
        else:
            return afnumpy.array(x).astype(cls)

class int_(numpy.int_):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.int_(x)
        else:
            return afnumpy.array(x).astype(cls)

class bool_(numpy.bool_):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.bool_(x)
        else:
            return afnumpy.array(x).astype(cls)

class float_(numpy.float_):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.float_(x)
        else:
            return afnumpy.array(x).astype(cls)

class float16(numpy.float16):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.float16(x)
        else:
            return afnumpy.array(x).astype(cls)

class float32(numpy.float32):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.float32(x)
        else:
            return afnumpy.array(x).astype(cls)

class float64(numpy.float64):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.float64(x)
        else:
            return afnumpy.array(x).astype(cls)

# removed for now
#class float128(numpy.float64):
#    def __new__(cls, x=0):
#        if isinstance(x, afnumpy.ndarray):
#            raise NotImplementedError('Arrayfire does not support 128 bit floats')
#        elif isinstance(x, numbers.Number):
#            return numpy.float64(x)
#        else:
#            raise NotImplementedError('Arrayfire does not support 128 bit floats')

class complex_(numpy.complex_):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.complex_(x)
        else:
            return afnumpy.array(x).astype(cls)

class complex64(numpy.complex64):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.complex64(x)
        else:
            return afnumpy.array(x).astype(cls)


class complex128(numpy.complex128):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        elif isinstance(x, numbers.Number):
            return numpy.complex128(x)
        else:
            return afnumpy.array(x).astype(cls)

float = float
complex = complex
bool = bool
int = int
long = int
bool8 = bool_

promote_types = numpy.promote_types

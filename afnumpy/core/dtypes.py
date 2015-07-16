import numpy
import afnumpy


class int64(numpy.int64):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.int64(x)

class int32(numpy.int32):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.int32(x)

class int16(numpy.intc):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.int16(x)

class int8(numpy.intc):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.int8(x)


class uint64(numpy.uint64):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.uint64(x)

class uint32(numpy.uint32):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.uint32(x)

class uint16(numpy.uintc):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.uint16(x)

class uint8(numpy.uintc):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.uint8(x)


class intc(numpy.intc):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.intc(x)

class intp(numpy.intp):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.intp(x)

class int_(numpy.int_):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.int_(x)


class bool_(numpy.bool_):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.bool_(x)


class float_(numpy.float_):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.float_(x)

class float16(numpy.float16):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.float16(x)

class float32(numpy.float32):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.float32(x)

class float64(numpy.float64):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.float64(x)


class complex_(numpy.complex_):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.complex_(x)

class complex64(numpy.complex64):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.complex64(x)

class complex128(numpy.complex128):
    def __new__(cls, x=0):
        if isinstance(x, afnumpy.ndarray):
            return x.astype(cls)
        else:
            return numpy.complex128(x)

float = float
complex = complex
bool = bool
int = int
long = long

promote_types = numpy.promote_types

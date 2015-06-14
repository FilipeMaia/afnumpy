import numpy
import arrayfire
import numbers
import multiarray

dim_t = numpy.int64

TypeMap = { float: arrayfire.f64,
                numpy.float32: arrayfire.f32,
                numpy.dtype('float32'): arrayfire.f32,
                numpy.float64: arrayfire.f64,
                numpy.dtype('float64'): arrayfire.f64,
                numpy.int8: arrayfire.b8,
                numpy.dtype('int8'): arrayfire.b8,
                numpy.uint8: arrayfire.u8,
                numpy.dtype('uint8'): arrayfire.u8,
                numpy.bool: arrayfire.b8,
                numpy.dtype('bool'): arrayfire.b8,
                numpy.int64: arrayfire.s64,
                numpy.dtype('int64'): arrayfire.s64,
                numpy.uint64: arrayfire.u64,
                numpy.dtype('uint64'): arrayfire.u64,
                numpy.uint32: arrayfire.u32,
                numpy.dtype('uint32'): arrayfire.u32,
                numpy.int32: arrayfire.s32,
                numpy.dtype('int32'): arrayfire.s32,
                numpy.complex128: arrayfire.c64,
                numpy.dtype('complex128'): arrayfire.c64,
                numpy.complex64: arrayfire.c32,
                numpy.dtype('complex64'): arrayfire.c32,
            }

InvTypeMap = {arrayfire.f64: numpy.float64,
                  arrayfire.f32: numpy.float32,
                  arrayfire.c64: numpy.complex128,
                  arrayfire.c32: numpy.complex64,
                  arrayfire.s32: numpy.int32,
                  arrayfire.s64: numpy.int64,
                  arrayfire.u32: numpy.uint32,
                  arrayfire.u64: numpy.uint64,
                  }

TypeToString = { arrayfire.f64: 'f64',
                     arrayfire.f32: 'f32',
                     arrayfire.u32: 'u32',
                     arrayfire.s32: 's32',
                     arrayfire.u64: 'u64',
                     arrayfire.s64: 's64',
                     arrayfire.c32: 'c32',
                     arrayfire.c64: 'c64',
                     }

dummy = object()

def af_shape(af_array):
    shape = ()
    for i in range(0,af_array.numdims()):
        shape = (af_array.dims(i),)+shape
    return shape

def raw(x):
    if(isinstance(x,multiarray.ndarray)):
        return x.d_array
    else:
        return x

def c2f(shape, dim = None):
    if isinstance(shape,numbers.Number):
        return shape
    if(dim is None):
        return shape[::-1]
    else:
        return len(shape)-dim-1



import numpy
import numbers
import afnumpy

dim_t = numpy.int64

TypeMap = { float: afnumpy.arrayfire.f64,
                numpy.float32: afnumpy.arrayfire.f32,
                numpy.dtype('float32'): afnumpy.arrayfire.f32,
                numpy.float64: afnumpy.arrayfire.f64,
                numpy.dtype('float64'): afnumpy.arrayfire.f64,
                numpy.int8: afnumpy.arrayfire.b8,
                numpy.dtype('int8'): afnumpy.arrayfire.b8,
                numpy.uint8: afnumpy.arrayfire.u8,
                numpy.dtype('uint8'): afnumpy.arrayfire.u8,
                numpy.bool: afnumpy.arrayfire.b8,
                numpy.dtype('bool'): afnumpy.arrayfire.b8,
                numpy.int64: afnumpy.arrayfire.s64,
                numpy.dtype('int64'): afnumpy.arrayfire.s64,
                numpy.uint64: afnumpy.arrayfire.u64,
                numpy.dtype('uint64'): afnumpy.arrayfire.u64,
                numpy.uint32: afnumpy.arrayfire.u32,
                numpy.dtype('uint32'): afnumpy.arrayfire.u32,
                numpy.int32: afnumpy.arrayfire.s32,
                numpy.dtype('int32'): afnumpy.arrayfire.s32,
                numpy.complex128: afnumpy.arrayfire.c64,
                numpy.dtype('complex128'): afnumpy.arrayfire.c64,
                numpy.complex64: afnumpy.arrayfire.c32,
                numpy.dtype('complex64'): afnumpy.arrayfire.c32,
            }

InvTypeMap = {afnumpy.arrayfire.f64: numpy.float64,
                  afnumpy.arrayfire.f32: numpy.float32,
                  afnumpy.arrayfire.c64: numpy.complex128,
                  afnumpy.arrayfire.c32: numpy.complex64,
                  afnumpy.arrayfire.s32: numpy.int32,
                  afnumpy.arrayfire.s64: numpy.int64,
                  afnumpy.arrayfire.u32: numpy.uint32,
                  afnumpy.arrayfire.u64: numpy.uint64,
                  }

TypeToString = { afnumpy.arrayfire.f64: 'f64',
                     afnumpy.arrayfire.f32: 'f32',
                     afnumpy.arrayfire.u32: 'u32',
                     afnumpy.arrayfire.s32: 's32',
                     afnumpy.arrayfire.u64: 'u64',
                     afnumpy.arrayfire.s64: 's64',
                     afnumpy.arrayfire.c32: 'c32',
                     afnumpy.arrayfire.c64: 'c64',
                     }

dummy = object()

def af_shape(af_array):
    shape = ()
    for i in range(0,af_array.numdims()):
        shape = (af_array.dims(i),)+shape
    return shape

def raw(x):
    if(isinstance(x,afnumpy.ndarray)):
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



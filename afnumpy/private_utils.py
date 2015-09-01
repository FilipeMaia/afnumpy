import numpy
import numbers
import afnumpy
import arrayfire_python

dim_t = numpy.int64

TypeToString = { arrayfire_python.f64.value: 'f64',
                     arrayfire_python.f32.value: 'f32',
                     arrayfire_python.u32.value: 'u32',
                     arrayfire_python.s32.value: 's32',
                     arrayfire_python.u64.value: 'u64',
                     arrayfire_python.s64.value: 's64',
                     arrayfire_python.c32.value: 'c32',
                     arrayfire_python.c64.value: 'c64',
                     arrayfire_python.b8.value: 'b8',
                     }

dummy = object()

def af_shape(af_array):
    shape = ()
    for i in range(0,af_array.numdims()):
        shape = (af_array.dims()[i],)+shape
    return shape

def raw(x):
    if(isinstance(x,afnumpy.ndarray)):
        return x.d_array
#    elif(isinstance(x,complex)):
#        return arrayfire_python.af_cdouble(x.real, x.imag)
    else:
        return x

def c2f(shape, dim = None):
    if isinstance(shape,numbers.Number):
        return shape
    if(dim is None):
        return shape[::-1]
    else:
        return len(shape)-dim-1


def typemap(dtype):
    InvTypeMap = {arrayfire_python.f64.value: numpy.float64,
                  arrayfire_python.f32.value: numpy.float32,
                  arrayfire_python.c64.value: numpy.complex128,
                  arrayfire_python.c32.value: numpy.complex64,
                  arrayfire_python.s32.value: numpy.int32,
                  arrayfire_python.s64.value: numpy.int64,
                  arrayfire_python.u32.value: numpy.uint32,
                  arrayfire_python.u64.value: numpy.uint64,
                  arrayfire_python.b8.value: numpy.bool,
              }
    TypeMap = {numpy.dtype('float32'): arrayfire_python.f32.value,
               numpy.dtype('float64'): arrayfire_python.f64.value,
               numpy.dtype('int8'): arrayfire_python.b8.value,
               numpy.dtype('uint8'): arrayfire_python.u8.value,
               numpy.dtype('bool'): arrayfire_python.b8.value,
               numpy.dtype('int64'): arrayfire_python.s64.value,
               numpy.dtype('uint64'): arrayfire_python.u64.value,
               numpy.dtype('uint32'): arrayfire_python.u32.value,
               numpy.dtype('int32'): arrayfire_python.s32.value,
               numpy.dtype('complex128'): arrayfire_python.c64.value,
               numpy.dtype('complex64'): arrayfire_python.c32.value,
    }
    if(dtype in InvTypeMap):
        return InvTypeMap[dtype]
    else:
        dtype = numpy.dtype(dtype)
        return TypeMap[dtype]

        


// Requires SWIG 3!

%module arrayfire
%include "typemaps.i"
%include "numpy.i"
// For typemaps for std::exception
%include "exception.i"
%include "std_complex.i"

#undef __cplusplus
#define __cplusplus 201103L

 // Ignore attributes to prevent compilation errors
#define __attribute__(x) 

 %{
 /* Includes the header in the wrapper code */
#include "af/algorithm.h"
#include "af/arith.h"
#include "af/array.h"
#include "af/blas.h"
#include "af/constants.h"
#include "af/complex.h"
#include "af/data.h"
#include "af/defines.h"
#include "af/device.h"
#include "af/exception.h"
#include "af/features.h"
#include "af/gfor.h"
#include "af/graphics.h"
#include "af/image.h"
#include "af/index.h"
#include "af/lapack.h"
#include "af/seq.h"
#include "af/signal.h"
#include "af/statistics.h"
#include "af/timing.h"
#include "af/util.h"
#include "af/version.h"
#include "af/vision.h"
 %}

%rename(astype) af::array::array_proxy::as(dtype type) const;
// as is a python keyword
%rename(astype) af::array::as(dtype type) const;
%rename(__getitem__) af::array::operator();

// For some reason I couldn't make this work properly when it was
// overloaded with other stuff
%rename(array_from_handle) af::array::array(const af_array handle);
   
%typemap(in) void * {
  $1 = (void *)PyInt_AsLong($input);
}


%typemap(in, numinputs=0) af_array *OUTPUT (af_array temp) { 
  $1 = &temp;
 }


%typemap(argout) af_array * OUTPUT {
    PyObject *o, *o2, *o3;
    o =  PyInt_FromLong((long)*$1);
    if ((!$result) || ($result == Py_None)) {
        $result = o;
    } else {
        if (!PyTuple_Check($result)) {
            PyObject *o2 = $result;
            $result = PyTuple_New(1);
            PyTuple_SetItem($result,0,o2);
        }
        o3 = PyTuple_New(1);
        PyTuple_SetItem(o3,0,o);
        o2 = $result;
        $result = PySequence_Concat(o2,o3);
        Py_DECREF(o2);
        Py_DECREF(o3);
    }
}


%typemap(out) af::array::array_proxy {
  // This looks really strange but it's necessary due to the overloading of & and *
  $result = SWIG_NewPointerObj(new af::array((af::array)(*(&$1))), SWIGTYPE_p_af__array, SWIG_POINTER_OWN |  0 );
}

%typemap(in) dim_t *  {
  $1 = (dim_t *)PyInt_AsLong($input);
}

/* %typemap(in) const dim_t *  { */
/*   $1 = (const dim_t *)PyInt_AsLong($input); */
/* } */

/* %typemap(in)  const dim_t * const  { */
/*   $1(( dim_t *)PyInt_AsLong($input)); */
/* } */

%apply af_array * OUTPUT { af_array *arr };
%apply af_array * OUTPUT { af_array *out };
%feature("flatnested") af::array::array_proxy;

%ignore af::operator+(const dim4& first, const dim4& second);
%ignore af::operator-(const dim4& first, const dim4& second);
%ignore af::operator*(const dim4& first, const dim4& second);
%ignore operator<<(std::ostream &s, const exception &e);
%define IGNORE(func)
%ignore func(dim_type const,dim_type const);
%ignore func(dim_type const,dim_type const,dim_type const);
%ignore func(dim_type const,dim_type const,dim_type const,dim_type const);
%ignore func(dim_type,dim_type);
%ignore func(dim_type,dim_type,dim_type);
%ignore func(dim_type,dim_type,dim_type,dim_type);
%enddef
IGNORE(af::randu)
IGNORE(af::randn)
IGNORE(af::identity)
IGNORE(af::array::array)

%define TYPE_IGNORE(func, type)
%ignore af::func(type, dim_type const,dim_type const);
%ignore af::func(type, dim_type const,dim_type const,dim_type const);
%ignore af::func(type, dim_type const,dim_type const,dim_type const,dim_type const);
%enddef
TYPE_IGNORE(constant, float)
TYPE_IGNORE(constant, double)
TYPE_IGNORE(constant, int)
TYPE_IGNORE(constant, unsigned int)
TYPE_IGNORE(constant, long)
TYPE_IGNORE(constant, unsigned long)
TYPE_IGNORE(constant, long long)
TYPE_IGNORE(constant, unsigned long long)
TYPE_IGNORE(constant, char)
TYPE_IGNORE(constant, unsigned char)
TYPE_IGNORE(constant, bool)
TYPE_IGNORE(constant, cdouble)
TYPE_IGNORE(constant, cfloat)

%ignore af::array::array_proxy::unlock() const;
%ignore af::array::unlock() const;

// These ones are missing compatible.h in the header
%ignore af::setintersect(const array &first, const array &second, const bool is_unique=false);
%ignore af::setunion(const array &first, const array &second, const bool is_unique=false);
%ignore af::setunique(const array &in, const bool is_sorted=false);

// These ones have on implementation that I could find
%ignore af::operator+(const af::cfloat &lhs, const double &rhs);
%ignore af::operator+(const af::cdouble &lhs, const double &rhs);

// Something wrong with these ones also
%ignore af::array::array_proxy::col(int) const;
%ignore af::array::array_proxy::cols(int, int) const;
%ignore af::array::array_proxy::row(int) const;
%ignore af::array::array_proxy::rows(int, int) const;
%ignore af::array::array_proxy::slice(int) const;
%ignore af::array::array_proxy::slices(int, int) const;



%ignore operator+(double, seq);
%ignore operator-(double, seq);
%ignore operator*(double, seq);
%rename(proxy_asarray) af::array::array_proxy::operator array();
%rename(seq_asarray) af::seq::operator array() const;
%rename(as_const_array) af::array::array_proxy::operator array() const;
%rename(g_afDevice) ::afDevice;
%rename(g_afHost) ::afHost;

%rename(logical_or) af::operator||;
%rename(logical_and) af::operator&&;
%rename(copy_on_write) af::array::operator=;
%rename(copy_on_write) af::array::array_proxy::operator=;

%rename(logical_not) af::array::operator!;
%rename(__getitem__) af::dim4::operator[];
%rename(copy) af::seq::operator=;
%rename(pprint) af::print;
%rename(copy) af::features::operator=;

%rename(copy) af::features::operator=;

// Try to handle exceptions
%exception {
try {
  $function
    }
catch (const std::exception & e) {
  PyErr_SetString(PyExc_RuntimeError, e.what());
  return NULL;
}
}

%typemap(out) af::af_cfloat {
  $result = PyComplex_FromDoubles(af::real($1),af::imag($1));
 }

%typemap(out) af::af_cdouble  {
  $result = PyComplex_FromDoubles(af::real($1),af::imag($1));
 }

%include "af/defines.h"
%include "af/index.h"
%include "af/complex.h"
%include "af/algorithm.h"
%include "af/arith.h"
%include "af/array.h"
%include "af/blas.h"
%include "af/constants.h"
%include "af/data.h"
%include "af/device.h"
%include "af/exception.h"
%include "af/features.h"
%include "af/gfor.h"
%include "af/graphics.h"
%include "af/image.h"
%include "af/lapack.h"
%include "af/seq.h"
%include "af/signal.h"
%include "af/statistics.h"
%include "af/timing.h"
%include "af/util.h"
%include "af/version.h"
%include "af/vision.h"


%extend af::array {
  void setValue(const af::index &s0, const af::array &value){
    ((*self)(s0)) = value;
  }
  void setValue(const af::index &s0, double value){
    ((*self)(s0)) = value;
  }
  void setValue(const af::index &s0, std::complex<double> value){
    ((*self)(s0)) = af::af_cdouble(value.real(),value.imag());
  }
};

%template(max_f32) af::max<float>;
%template(max_f64) af::max<double>;
%template(max_s32) af::max<int32_t>;
%template(max_u32) af::max<uint32_t>;
//%template(max_s64) af::max<int64_t>(const array &in);
//%template(max_u64) af::max<uint64_t>;
%template(max_c32) af::max<af::cfloat>;
%template(max_c64) af::max<af::cdouble>;


%template(min_f32) af::min<float>;
%template(min_f64) af::min<double>;
%template(min_s32) af::min<int32_t>;
%template(min_u32) af::min<uint32_t>;
//%template(min_s64) af::min<int64_t>(const array &in);
//%template(min_u64) af::min<uint64_t>;
%template(min_c32) af::min<af::cfloat>;
%template(min_c64) af::min<af::cdouble>;



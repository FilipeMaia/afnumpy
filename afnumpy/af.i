// Requires SWIG 3!

%module arrayfire
%include "typemaps.i"
%include "numpy.i"

#undef __cplusplus
#define __cplusplus 201103L

 // Ignore attributes to prevent compilation errors
#define __attribute__(x) 
// This needs to be defined only for Mac OS X
#define __APPLE__
#define __MACH__

 %{
 /* Includes the header in the wrapper code */
 #include "af/defines.h"
 #include "af/dim4.hpp"
 #include "af/seq.h"
 #include "af/array.h"
 #include "af/data.h"
 #include "af/compatible.h"
 #include "af/algorithm.h"
 #include "af/arith.h"
 #include "af/blas.h"
 #include "af/device.h"
 #include "af/exception.h"
 #include "af/features.h"
 #include "af/gfor.h"
 #include "af/image.h"
 #include "af/index.h"
 #include "af/signal.h"
 #include "af/statistics.h"
 #include "af/timing.h"
 #include "af/util.h"
 %}


%rename(astype) af::array::array_proxy::as(dtype type) const;
// as is a python keyword
%rename(astype) af::array::as(dtype type) const;
%rename(__getitem__) af::array::operator();
   
%typemap(in) void * {
  $1 = (void *)PyInt_AsLong($input);
}
%typemap(in) dim_type *  {
  $1 = (dim_type *)PyInt_AsLong($input);
}
%apply af_array *OUTPUT { af_array *arr };
//%nodefaultctor af::array::array_proxy; 
//%feature("valuewrapper") af::array::array_proxy;
%feature("flatnested") af::array::array_proxy;

%ignore af::seqElements(const af_seq & seq);
%ignore af::isSpan(const af_seq & seq);
%ignore af::calcDim(const af_seq &seq, const dim_type &parentDim);
%ignore af::toDims(const std::vector<af_seq>& seqs, af::dim4 parentDims);
%ignore af::toOffset(const std::vector<af_seq>& seqs, af::dim4 parentDims);
%ignore af::toStride(const std::vector<af_seq>& seqs, af::dim4 parentDims);
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

%ignore operator+(double, seq);
%ignore operator-(double, seq);
%ignore operator*(double, seq);
%rename(asarray) af::array::array_proxy::operator array();
%rename(asarray) af::seq::operator array() const;
%rename(as_const_array) af::array::array_proxy::operator array() const;
%rename(g_afDevice) ::afDevice;
%rename(g_afHost) ::afHost;

 %include "af/defines.h"
 %include "af/dim4.hpp"
 %include "af/seq.h"
 %include "af/index.h"
 %include "af/array.h"
 %include "af/data.h"
 %include "af/compatible.h"
 %include "af/algorithm.h"
 %include "af/arith.h"
 %include "af/blas.h"
 %include "af/device.h"
 %include "af/exception.h"
 %include "af/features.h"
 %include "af/gfor.h"
 %include "af/image.h"
 %include "af/signal.h"
 %include "af/statistics.h"
 %include "af/timing.h"
 %include "af/util.h"


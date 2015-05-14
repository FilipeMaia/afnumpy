// Requires SWIG 3!

%module arrayfire
%include "typemaps.i"
%include "numpy.i"

#undef __cplusplus
#define __cplusplus 1

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
%ignore af::toDims(const std::vector<af_seq>& seqs, af::dim4 parentDims);
%ignore af::toOffset(const std::vector<af_seq>& seqs, af::dim4 parentDims);
%ignore af::toStride(const std::vector<af_seq>& seqs, af::dim4 parentDims);
%ignore af::operator+(const dim4& first, const dim4& second);
%ignore af::operator-(const dim4& first, const dim4& second);
%ignore af::operator*(const dim4& first, const dim4& second);

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


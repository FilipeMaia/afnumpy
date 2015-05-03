%module arrayfire
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
  // #include "arrayfire.h"

 %}
 
 /* Parse the header file to generate wrappers */
 %include "af/defines.h"
  // Ignore attributes to prevent compilation errors
 #define __attribute__(x) 
 %ignore af::seqElements(const af_seq & seq);
 %ignore af::isSpan(const af_seq & seq);
 %ignore af::toDims(const std::vector<af_seq>& seqs, af::dim4 parentDims);
 %ignore af::toOffset(const std::vector<af_seq>& seqs, af::dim4 parentDims);
 %ignore af::toStride(const std::vector<af_seq>& seqs, af::dim4 parentDims);
 %ignore af::operator+(const dim4& first, const dim4& second);
 %ignore af::operator-(const dim4& first, const dim4& second);
 %ignore af::operator*(const dim4& first, const dim4& second);
 %include "af/dim4.hpp"
 %include "af/seq.h"
  // as is a python keyword
 %rename(astype) af::array::as(dtype type) const;
  // for some reason these are missing from the precompiled library
 %ignore af::array::isbool() const;
 %ignore af::array::unlock() const;
 %ignore af::constant(bool val, const dim_type d0, dtype ty=b8);
 %ignore af::constant(bool val, const dim_type d0, const dim_type, dtype ty=b8);
 %ignore af::constant(bool val, const dim_type d0, const dim_type, 
		      const dim_type, dtype ty=b8);
 %ignore af::constant(bool val, const dim_type d0, const dim_type, 
		      const dim_type, const dim_type, dtype ty=b8);
 %typemap(in) void * {
   $1 = (void *)PyInt_AsLong($input);
  }
 %include "af/array.h"
 %include "af/data.h"
 %include "af/compatible.h"
 %ignore af::setintersect(const array &first, const array &second,
			  bool is_unique=false);
 %ignore af::setunion(const array &first, const array &second,
		      bool is_unique=false);
 %ignore af::setunique(const array &in, bool is_sorted=false);
 %include "af/algorithm.h"
 %include "af/arith.h"
 %include "af/blas.h"
 %include "af/device.h"
  // %include "arrayfire.h"


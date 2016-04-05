import arrayfire

# We're going to test several arrayfire behaviours that we rely on


from asserts import *
import afnumpy as af
import numpy as np

def test_cast():
    a = afnumpy.random.rand(2,3)
    afnumpy.arrayfire.sync()
    # Check that device_ptr does not cause a copy
    assert a.d_array.device_ptr() == a.d_array.device_ptr()
    # Check that cast does not cause a copy
    assert arrayfire.cast(a.d_array, a.d_array.dtype()).device_ptr() == a.d_array.device_ptr()

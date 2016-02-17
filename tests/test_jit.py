import afnumpy
from asserts import *

def test_conditionals():
    a = afnumpy.arange(10, dtype="float32") - 5.
    b = afnumpy.ones((10), dtype="float32")
    afnumpy.arrayfire.backend.get().af_eval(a.d_array.arr)
    a_mask = a < 0.
    a_sum = a_mask.sum()
    a -= b
    assert(a_sum == a_mask.sum())
